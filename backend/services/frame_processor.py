import os
import json
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from backend.services.visual_features import VisualFeatureExtractor
from backend.utils.ffmpeg_utils import get_ffmpeg_cmd, get_media_duration

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma', '.opus', '.aiff', '.alac'}

class FrameProcessor:
    def __init__(self, config: dict):
        self.config = config
        visual_cfg = config.get("visual", {})
        sys_cfg = config.get("system", {})
        self.frame_interval = sys_cfg.get("frame_interval_seconds", 15)
        self.feature_extractor = VisualFeatureExtractor(
            feature_dim=visual_cfg.get("feature_dim", 512),
            device=sys_cfg.get("device", "cpu")
        )

    def process_video_frames(self, video_path: str, cache_dir: str = "cache", progress_callback=None) -> list:
        video_name = Path(video_path).stem
        ext = Path(video_path).suffix.lower()

        # If it's an audio recording, skip frame extraction gracefully
        if ext in AUDIO_EXTENSIONS:
            print(f"[FRAME] Audio file detected ({ext}). Skipping visual frame extraction.")
            if progress_callback:
                progress_callback("Audio file detected — processing acoustic stream...", 60)
            return []

        frames_dir = os.path.join(cache_dir, "frames", video_name)
        features_json_path = os.path.join(cache_dir, "visual_features", f"{video_name}.json")

        if os.path.exists(features_json_path):
            with open(features_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        os.makedirs(frames_dir, exist_ok=True)

        # Use fast FFmpeg keyframe seeking instead of slow OpenCV cap.set()
        # OpenCV cap.set() decodes intermediate frames sequentially in CPU, taking 10-15 mins on long videos.
        # FFmpeg keyframe seeking (-ss before -i) jumps directly to the nearest keyframe in 0.05 seconds!
        dur = get_media_duration(video_path) or 60.0
        num_frames = min(15, max(3, int(dur // 60))) if dur > 180 else min(10, max(2, int(dur // 15)))
        timestamps = [round((i + 0.5) * (dur / num_frames), 2) for i in range(num_frames)]
        if not timestamps:
            timestamps = [1.0]

        total_targets = len(timestamps)
        extracted_frames = []
        ffmpeg_cmd = get_ffmpeg_cmd()

        print(f"[FRAME] Fast keyframe extraction starting for {total_targets} frames (duration: {dur:.1f}s)...")

        for idx, ts in enumerate(timestamps):
            frame_filename = f"frame_{ts:.1f}s.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            # Fast keyframe extraction + immediate 224x224 downscaling in FFmpeg
            cmd = [
                ffmpeg_cmd, "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", "scale=224:224",
                "-q:v", "3",
                str(frame_path)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                try:
                    pil_img = Image.open(frame_path).convert("RGB")
                    features = self.feature_extractor.extract_features(pil_img)
                    description = self.feature_extractor.generate_visual_description(pil_img)

                    extracted_frames.append({
                        "timestamp": ts,
                        "frame_path": frame_path,
                        "description": description,
                        "features": features.tolist() if hasattr(features, 'tolist') else list(features)
                    })
                except Exception as img_err:
                    print(f"[FRAME] Warning analyzing frame at {ts}s: {img_err}")

            if progress_callback:
                mapped_pct = int(50 + ((idx + 1) / total_targets) * 15)
                progress_callback(f"Analyzing visual scenes: {idx + 1} / {total_targets} frames", mapped_pct)

        print(f"[FRAME] Fast extraction complete: extracted {len(extracted_frames)} frames in seconds!")

        Path(features_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(features_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_frames, f, indent=2)

        return extracted_frames

    def unload_model(self):
        if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'unload_model'):
            self.feature_extractor.unload_model()
