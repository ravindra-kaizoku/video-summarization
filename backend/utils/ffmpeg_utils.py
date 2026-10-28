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

def extract_audio(video_path: str, output_wav_path: str) -> str:
    """Extracts audio from video file to 16kHz mono WAV format for Whisper processing."""
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
            print(f"Warning: Video {video_path.name} has no audio stream.")
            return ""
        # Create an empty or silent WAV file fallback if needed
        print(f"FFmpeg error on audio extraction: {result.stderr}")
        return ""
    
    return str(output_wav_path)
