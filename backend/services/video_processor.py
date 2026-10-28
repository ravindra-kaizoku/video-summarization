import cv2
import os
from pathlib import Path

class VideoProcessor:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def extract_metadata(self, video_path: str) -> dict:
        video_path = str(Path(video_path))
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()

        duration_seconds = total_frames / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60.0

        return {
            "file_path": video_path,
            "filename": Path(video_path).name,
            "duration_seconds": round(duration_seconds, 2),
            "duration_minutes": round(duration_minutes, 2),
            "fps": fps,
            "total_frames": total_frames,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height
        }
