import os
import cv2
from pathlib import Path
from backend.utils.ffmpeg_utils import get_media_duration

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma', '.opus', '.aiff', '.alac'}

class VideoProcessor:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def extract_metadata(self, media_path: str) -> dict:
        media_path = str(Path(media_path))
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found at: {media_path}")

        path_obj = Path(media_path)
        ext = path_obj.suffix.lower()

        # Check if the uploaded file is purely an audio file
        if ext in AUDIO_EXTENSIONS:
            duration_seconds = get_media_duration(media_path)
            duration_minutes = round(duration_seconds / 60.0, 2)
            return {
                "file_path": media_path,
                "filename": path_obj.name,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_minutes,
                "fps": 0,
                "total_frames": 0,
                "resolution": "Audio Recording",
                "width": 0,
                "height": 0,
                "is_audio_only": True
            }

        # Otherwise attempt video capture via OpenCV
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            # Fallback to ffmpeg duration if OpenCV fails (e.g. audio container or unsupported codec)
            duration_seconds = get_media_duration(media_path)
            duration_minutes = round(duration_seconds / 60.0, 2)
            return {
                "file_path": media_path,
                "filename": path_obj.name,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_minutes,
                "fps": 0,
                "total_frames": 0,
                "resolution": "Audio/Media Stream",
                "width": 0,
                "height": 0,
                "is_audio_only": True
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration_seconds = total_frames / fps if fps > 0 else get_media_duration(media_path)
        duration_minutes = duration_seconds / 60.0

        return {
            "file_path": media_path,
            "filename": path_obj.name,
            "duration_seconds": round(duration_seconds, 2),
            "duration_minutes": round(duration_minutes, 2),
            "fps": fps,
            "total_frames": total_frames,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "is_audio_only": False
        }
