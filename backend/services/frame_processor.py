import cv2
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from backend.services.visual_features import VisualFeatureExtractor

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
        frames_dir = os.path.join(cache_dir, "frames", video_name)
        features_json_path = os.path.join(cache_dir, "visual_features", f"{video_name}.json")

        if os.path.exists(features_json_path):
            with open(features_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(fps * self.frame_interval)
        if frame_step < 1:
            frame_step = 30

        target_indices = list(range(0, total_video_frames, frame_step))
        if not target_indices:
            target_indices = [0]
            
        max_frames = 25
        if len(target_indices) > max_frames:
            step = len(target_indices) / max_frames
            target_indices = [target_indices[int(i * step)] for i in range(max_frames)]

        total_targets = len(target_indices)
        extracted_frames = []

        for idx, frame_idx in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            timestamp = round(frame_idx / fps, 2)
            frame_filename = f"frame_{timestamp:.1f}s.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            
            # Convert OpenCV BGR to PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            features = self.feature_extractor.extract_features(pil_img)
            description = self.feature_extractor.generate_visual_description(pil_img)

            extracted_frames.append({
                "timestamp": timestamp,
                "frame_path": frame_path,
                "description": description,
                "features": features.tolist()
            })

            if progress_callback and (idx % 5 == 0 or idx == total_targets - 1):
                mapped_pct = int(50 + ((idx + 1) / total_targets) * 15)
                progress_callback(f"Analyzing visual scenes: {idx + 1} / {total_targets} frames", mapped_pct)

        cap.release()

        Path(features_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(features_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_frames, f, indent=2)

        return extracted_frames

    def unload_model(self):
        if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'unload_model'):
            self.feature_extractor.unload_model()
