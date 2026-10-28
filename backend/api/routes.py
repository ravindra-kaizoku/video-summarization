import os
import sys
import re
import uuid
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yt_dlp

from backend.services.pipeline import VideoSummarizationPipeline
from backend.utils.ffmpeg_utils import get_ffmpeg_cmd

router = APIRouter()
pipeline = VideoSummarizationPipeline()

tasks_db = {}
results_db = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class UrlProcessRequest(BaseModel):
    url: str

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    tasks_db[task_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "File uploaded successfully.",
        "file_path": str(save_path),
        "filename": file.filename,
        "video_url": f"/uploads/{save_path.name}"
    }

    return {
        "task_id": task_id,
        "filename": file.filename,
        "video_url": f"/uploads/{save_path.name}",
        "message": "Upload complete"
    }

def extract_youtube_id(url: str):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def get_video_info(url: str):
    """Fast extraction of video title, duration, thumbnail, and channel without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        }
    }
    yt_id = extract_youtube_id(url)
    default_thumb = f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg" if yt_id else ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            dur = info.get("duration", 0) or 0
            return {
                "title": info.get("title", "YouTube Video"),
                "thumbnail": info.get("thumbnail") or default_thumb,
                "duration": dur,
                "duration_formatted": f"{int(dur // 60)}:{int(dur % 60):02d}",
                "uploader": info.get("uploader") or info.get("channel") or "Unknown Creator",
                "youtube_id": yt_id
            }
    except Exception as e:
        print(f"Warning extracting quick video info: {e}")
        return {
            "title": "YouTube Video" if yt_id else "Video Stream",
            "thumbnail": default_thumb,
            "duration": 0,
            "duration_formatted": "Unknown",
            "uploader": "YouTube" if yt_id else "Video Source",
            "youtube_id": yt_id
        }

def download_video_url(url: str, output_path: str, progress_callback=None):
    """Downloads YouTube video or direct video link using yt-dlp capped at 480p/720p with progress hook."""
    ffmpeg_exe = get_ffmpeg_cmd()
    print(f"  --> [FFmpeg Path]: {ffmpeg_exe}")
    
    base_stem = str(Path(output_path).with_suffix(''))

    def yt_hook(d):
        if progress_callback and d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded = d.get('downloaded_bytes') or 0
            speed = d.get('speed') or 0
            speed_mb = speed / (1024 * 1024) if speed else 0
            if total > 0:
                pct = int((downloaded / total) * 100)
                msg = f"Downloading YouTube stream: {pct}% ({speed_mb:.1f} MB/s)"
                mapped_prog = int(10 + (pct * 0.15))
                progress_callback(msg, mapped_prog)
            else:
                progress_callback("Downloading video stream...", 15)

    ydl_opts = {
        # Fast 480p/720p mp4 stream selection for rapid download & full compatibility
        'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=480]/best[height<=720]/best',
        'outtmpl': base_stem + '.%(ext)s',
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_exe,
        'quiet': False,
        'no_warnings': False,
        'noplaylist': True,
        'progress_hooks': [yt_hook],
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        }
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    
    # Locate actual downloaded file and ensure it is saved at output_path
    actual_file = None
    for ext in ['.mp4', '.mkv', '.webm', '.mov']:
        cand = Path(base_stem + ext)
        if cand.exists():
            actual_file = cand
            break

    target_path = Path(output_path)
    if actual_file and actual_file.resolve() != target_path.resolve():
        if target_path.exists():
            target_path.unlink()
        actual_file.rename(target_path)
    
    return info

def run_pipeline_task(task_id: str, file_path: str, url: str = None):
    tasks_db[task_id]["status"] = "processing"

    def progress_callback(stage: str, percentage: int):
        tasks_db[task_id]["progress"] = percentage
        tasks_db[task_id]["message"] = stage
        print(f"[{percentage}%] {stage}")
        sys.stdout.flush()

    try:
        video_title = None
        if url and not os.path.exists(file_path):
            print(f"[3] Video download started for URL: {url}")
            progress_callback("Connecting to YouTube video stream...", 10)
            info = download_video_url(url, file_path, progress_callback=progress_callback)
            if info:
                video_title = info.get("title")
                tasks_db[task_id]["title"] = video_title
                tasks_db[task_id]["thumbnail"] = info.get("thumbnail")
                tasks_db[task_id]["duration"] = info.get("duration")
            print(f"[4] Video download completed. Saved to: {file_path}")

        result = pipeline.process_long_video(file_path, progress_callback=progress_callback)
        result["video_url"] = tasks_db[task_id]["video_url"]
        if video_title:
            result["title"] = video_title
        if tasks_db[task_id].get("youtube_id"):
            result["youtube_id"] = tasks_db[task_id]["youtube_id"]
        if tasks_db[task_id].get("thumbnail"):
            result["thumbnail"] = tasks_db[task_id]["thumbnail"]
            
        results_db[task_id] = result
        
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["message"] = "Summarization completed successfully."
        print(f"[13] Final summary generated successfully for Task ID: {task_id}")
        sys.stdout.flush()

    except Exception as e:
        err_msg = str(e)
        tb_str = traceback.format_exc()
        print(f"\n================ BACKEND TERMINAL ERROR TRACEBACK ================", file=sys.stderr)
        print(f"Task ID: {task_id}", file=sys.stderr)
        print(f"Error Message: {err_msg}", file=sys.stderr)
        print(tb_str, file=sys.stderr)
        print(f"==================================================================\n", file=sys.stderr)
        sys.stderr.flush()

        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = err_msg
        tasks_db[task_id]["traceback"] = tb_str
        tasks_db[task_id]["message"] = f"Error: {err_msg}"

@router.post("/video-info")
async def get_url_video_info(req: UrlProcessRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Invalid video URL")
    return get_video_info(req.url.strip())

@router.post("/process-url")
async def process_video_url(req: UrlProcessRequest, background_tasks: BackgroundTasks):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Invalid video URL")

    url_str = req.url.strip()
    print(f"\n[1] URL received: {url_str}")

    task_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{task_id}.mp4"
    yt_id = extract_youtube_id(url_str)
    initial_thumb = f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg" if yt_id else ""
    print(f"[2] Task created: {task_id} (YouTube ID: {yt_id})")

    tasks_db[task_id] = {
        "status": "processing",
        "progress": 5,
        "message": "Initializing YouTube video processing...",
        "file_path": str(save_path),
        "filename": url_str,
        "video_url": f"/uploads/{save_path.name}",
        "youtube_id": yt_id,
        "thumbnail": initial_thumb,
        "source_url": url_str
    }

    background_tasks.add_task(run_pipeline_task, task_id, str(save_path), url_str)

    return {
        "task_id": task_id,
        "status": "processing_started",
        "video_url": f"/uploads/{save_path.name}",
        "youtube_id": yt_id,
        "thumbnail": initial_thumb,
        "message": "URL video processing started"
    }

@router.post("/process/{task_id}")
async def process_video(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task ID not found")

    file_path = tasks_db[task_id]["file_path"]
    background_tasks.add_task(run_pipeline_task, task_id, file_path)

    return {"task_id": task_id, "status": "processing_started"}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return tasks_db[task_id]

@router.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in results_db:
        if task_id in tasks_db:
            return {
                "status": tasks_db[task_id]["status"],
                "message": tasks_db[task_id]["message"],
                "error": tasks_db[task_id].get("error", None),
                "traceback": tasks_db[task_id].get("traceback", None)
            }
        raise HTTPException(status_code=404, detail="Result not ready or task not found")
    return results_db[task_id]
