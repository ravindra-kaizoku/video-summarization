import os
from pathlib import Path
from backend.utils.ffmpeg_utils import extract_audio
from backend.services.whisper_service import WhisperService

class AudioProcessor:
    def __init__(self, config: dict):
        self.config = config
        audio_cfg = config.get("audio", {})
        self.whisper_service = WhisperService(
            model_size=audio_cfg.get("whisper_model", "tiny"),
            device=audio_cfg.get("whisper_device", "cpu"),
            compute_type=audio_cfg.get("compute_type", "int8")
        )

    def process_audio(self, video_path: str, cache_dir: str = "cache", progress_callback=None, total_duration: float = None) -> tuple:
        video_name = Path(video_path).stem
        wav_output_path = os.path.join(cache_dir, "audio", f"{video_name}.wav")
        transcript_json_path = os.path.join(cache_dir, "transcripts", f"{video_name}.json")

        # 1. Extract audio WAV if not cached
        if not os.path.exists(wav_output_path):
            extracted_path = extract_audio(video_path, wav_output_path)
        else:
            extracted_path = wav_output_path

        # 2. Transcribe audio to timestamped transcript with live progress
        transcript = self.whisper_service.transcribe(
            extracted_path,
            cache_json_path=transcript_json_path,
            progress_callback=progress_callback,
            total_duration=total_duration
        )

        return extracted_path, transcript
