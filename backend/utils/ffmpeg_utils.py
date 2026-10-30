import re
import shutil
import subprocess
from pathlib import Path

def get_ffmpeg_cmd():
    """Returns the executable command for ffmpeg, trying system PATH or imageio_ffmpeg binary."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")

def get_media_duration(file_path: str) -> float:
    """Extracts duration in seconds from any media file (video or audio) using ffmpeg."""
    try:
        ffmpeg_cmd = get_ffmpeg_cmd()
        cmd = [ffmpeg_cmd, "-i", str(file_path)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
        if m:
            h, mins, s = m.groups()
            return round(int(h) * 3600 + int(mins) * 60 + float(s), 2)
    except Exception as e:
        print(f"Warning parsing media duration with ffmpeg: {e}")
    return 60.0

def extract_audio(video_path: str, output_wav_path: str) -> str:
    """Extracts audio from video or audio file to 16kHz mono WAV format for Whisper processing."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Command to extract 16kHz mono audio WAV
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        # Check if video simply has no audio stream
        if "does not contain any stream" in result.stderr or "Output file #0 does not contain any stream" in result.stderr:
            print(f"Warning: File {video_path.name} has no audio stream.")
            return ""
        print(f"FFmpeg error on audio extraction: {result.stderr}")
        return ""
    
    return str(output_wav_path)
