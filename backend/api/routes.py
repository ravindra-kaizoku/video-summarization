import os
import sys
import re
import uuid
import traceback
import gc
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yt_dlp

from backend.utils.ffmpeg_utils import get_ffmpeg_cmd

router = APIRouter()

_pipeline = None

def get_pipeline():
    """Lazily load the deep learning pipeline on first request to conserve memory on startup."""
    global _pipeline
    if _pipeline is None:
        print("[BACKEND] Initializing VideoSummarizationPipeline lazily...")
        from backend.services.pipeline import VideoSummarizationPipeline
        _pipeline = VideoSummarizationPipeline()
    return _pipeline

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

    import shutil
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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

def ensure_cookie_file():
    cookie_file = os.environ.get("YOUTUBE_COOKIEFILE", "cookies.txt")
    cookies_content = os.environ.get("YOUTUBE_COOKIES_CONTENT")
    if cookies_content and not os.path.exists(cookie_file):
        try:
            with open(cookie_file, "w", encoding="utf-8") as f:
                f.write(cookies_content)
        except Exception as e:
            print(f"Warning writing cookie file: {e}")
    if os.path.exists(cookie_file):
        return cookie_file
    return None


def fetch_youtube_transcript(video_id: str) -> list:
    """
    Fetch YouTube captions using youtube-transcript-api.
    Returns transcript as a list of {'start': float, 'end': float, 'text': str} segments.
    This bypasses YouTube's bot detection since it uses the caption API, not the video stream.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        
        transcript_segments = []
        for snippet in fetched:
            start = float(snippet.start)
            duration = float(snippet.duration)
            text = str(snippet.text).strip()
            if text:
                transcript_segments.append({
                    "start": round(start, 2),
                    "end": round(start + duration, 2),
                    "text": text
                })
        
        print(f"[TRANSCRIPT-API] Fetched {len(transcript_segments)} caption segments for {video_id}")
        return transcript_segments
    except Exception as e:
        print(f"[TRANSCRIPT-API] Failed to fetch transcript for {video_id}: {e}")
        raise


def run_transcript_only_pipeline(task_id: str, video_id: str, transcript_segments: list, progress_callback):
    """
    Text-only summarization pipeline for YouTube videos when video download is blocked.
    Uses captions from youtube-transcript-api instead of downloading the video file.
    Skips audio extraction, whisper, and frame extraction — goes straight to 
    temporal alignment (with zero visual features), importance scoring, and T5 summarization.
    """
    import numpy as np

    progress_callback("Building text-only segments from YouTube captions...", 30)

    if not transcript_segments:
        raise RuntimeError("No transcript/captions available for this YouTube video.")

    # Calculate total duration from transcript timestamps
    total_duration = max(seg["end"] for seg in transcript_segments) if transcript_segments else 60.0
    duration_minutes = total_duration / 60.0

    # Create video metadata (no actual video file)
    metadata = {
        "duration_seconds": total_duration,
        "duration_minutes": round(duration_minutes, 1),
        "fps": 0,
        "width": 0,
        "height": 0,
        "total_frames": 0,
        "source": "youtube_transcript_api"
    }

    # Temporal alignment: chunk transcript into time windows (no visual features)
    progress_callback("Aligning transcript into temporal chunks...", 40)
    
    pipeline_instance = get_pipeline()
    chunk_duration = pipeline_instance.config.get("system", {}).get("chunk_duration_seconds", 180)
    num_chunks = max(1, int(np.ceil(total_duration / chunk_duration)))
    
    chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min(total_duration, (i + 1) * chunk_duration)
        
        # Filter transcript segments within this chunk
        chunk_texts = [
            seg["text"] for seg in transcript_segments
            if seg["start"] < end_time and seg["end"] >= start_time
        ]
        chunk_transcript = " ".join(chunk_texts).strip()
        if not chunk_transcript:
            chunk_transcript = f"Caption segment from {int(start_time//60):02d}:{int(start_time%60):02d} to {int(end_time//60):02d}:{int(end_time%60):02d}."
        
        chunks.append({
            "chunk_id": i + 1,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "transcript": chunk_transcript,
            "visual_information": ["No video frames available (transcript-only mode)."],
            "visual_features": [0.0] * 512,
            "frame_count": 0
        })
    
    print(f"[TRANSCRIPT-PIPELINE] Created {len(chunks)} text-only chunks from {len(transcript_segments)} caption segments")

    # Importance scoring (text features only, visual features zeroed)
    progress_callback("Scoring segment importance using AI model...", 60)
    top_k = pipeline_instance.config.get("model", {}).get("top_k_chunks", 15)
    scored_chunks, selected_chunks = pipeline_instance.importance_service.score_chunks(chunks, top_k=top_k)
    print(f"[TRANSCRIPT-PIPELINE] Selected {len(selected_chunks)} important chunks")

    # Unload importance model to free RAM for summarizer
    pipeline_instance.importance_service.unload_model()
    gc.collect()

    # Summarization
    total_sel = len(selected_chunks)
    chunk_summaries = []
    for i, c in enumerate(selected_chunks):
        pct = int(75 + ((i + 1) / max(1, total_sel)) * 20)
        progress_callback(f"Summarizing chapter {i + 1} of {total_sel}...", pct)
        chunk_summaries.append(pipeline_instance.summarizer.generate_chunk_summary(c))

    progress_callback("Finalizing executive summary...", 98)
    final_output = pipeline_instance.summarizer.generate_final_summary(
        video_metadata=metadata,
        selected_chunks=selected_chunks,
        chunk_summaries=chunk_summaries
    )

    # Unload summarizer to free RAM
    pipeline_instance.summarizer.unload_model()
    gc.collect()

    final_output["metadata"] = metadata
    final_output["total_chunks_processed"] = len(chunks)
    final_output["selected_chunks_count"] = len(selected_chunks)
    final_output["source_mode"] = "transcript_only"

    return final_output

def get_video_info(url: str):
    """Fast extraction of video title, duration, thumbnail, and channel without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios'],
                'player_skip': ['webpage', 'configs']
            }
        }
    }
    cookie_path = ensure_cookie_file()
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path

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
                msg = f"Downloading stream: {pct}% ({speed_mb:.1f} MB/s)"
                mapped_prog = int(10 + (pct * 0.15))
                progress_callback(msg, mapped_prog)
            else:
                progress_callback("Downloading video stream...", 15)

    ydl_opts = {
        # Format 18 is pre-muxed 360p mp4 available directly from the Android/iOS client without bot block
        'format': '18/best[height<=480][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': base_stem + '.%(ext)s',
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_exe,
        'quiet': False,
        'no_warnings': False,
        'noplaylist': True,
        'progress_hooks': [yt_hook],
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios'],
                'player_skip': ['webpage', 'configs']
            }
        }
    }
    cookie_path = ensure_cookie_file()
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path
    
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
        yt_id = tasks_db[task_id].get("youtube_id")
        use_transcript_fallback = False

        if url and not os.path.exists(file_path):
            print(f"[3] Video download started for URL: {url}")
            progress_callback("Connecting to YouTube video stream...", 10)
            try:
                info = download_video_url(url, file_path, progress_callback=progress_callback)
                if info:
                    video_title = info.get("title")
                    tasks_db[task_id]["title"] = video_title
                    tasks_db[task_id]["thumbnail"] = info.get("thumbnail")
                    tasks_db[task_id]["duration"] = info.get("duration")
                print(f"[4] Video download completed. Saved to: {file_path}")
            except Exception as dl_err:
                dl_msg = str(dl_err).lower()
                # If YouTube blocks us (bot detection), fall back to transcript-only mode
                if yt_id and ("bot" in dl_msg or "sign in" in dl_msg or "confirm" in dl_msg):
                    print(f"[4-FALLBACK] YouTube download blocked. Switching to transcript-only mode.")
                    progress_callback("YouTube blocked video download. Fetching captions instead...", 15)
                    use_transcript_fallback = True
                else:
                    raise  # Re-raise non-bot errors

        # -- Transcript-only path for blocked YouTube URLs --
        if use_transcript_fallback and yt_id:
            progress_callback("Fetching YouTube captions via transcript API...", 20)
            transcript_segments = fetch_youtube_transcript(yt_id)
            result = run_transcript_only_pipeline(task_id, yt_id, transcript_segments, progress_callback)

            result["video_url"] = tasks_db[task_id].get("video_url", "")
            if video_title:
                result["title"] = video_title
            result["youtube_id"] = yt_id
            if tasks_db[task_id].get("thumbnail"):
                result["thumbnail"] = tasks_db[task_id]["thumbnail"]

            results_db[task_id] = result
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["progress"] = 100
            tasks_db[task_id]["message"] = "Summarization completed successfully (transcript-only mode)."
            print(f"[13] Summary generated via transcript-only fallback for Task ID: {task_id}")
            sys.stdout.flush()
            return

        # -- Standard full video pipeline (local uploads or successful YouTube downloads) --
        pipeline_instance = get_pipeline()
        result = pipeline_instance.process_long_video(file_path, progress_callback=progress_callback)
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

        user_facing_msg = f"Error: {err_msg}"
        if "bot" in err_msg.lower() or "sign in" in err_msg.lower():
            user_facing_msg = (
                "YouTube blocked this video download. "
                "The transcript-based fallback also failed. "
                "Please upload your local video file using the Upload Video tab."
            )

        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = err_msg
        tasks_db[task_id]["traceback"] = tb_str
        tasks_db[task_id]["message"] = user_facing_msg

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
