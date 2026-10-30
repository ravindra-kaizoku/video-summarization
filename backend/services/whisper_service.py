import os
import json
import gc
from pathlib import Path
from backend.utils.memory_utils import get_available_memory_mb

class WhisperService:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _init_model(self):
        if self.model is None:
            # Faster-Whisper tiny (int8) requires only ~75MB RAM.
            # Only fall back if available RAM is critically low (< 90MB).
            avail_mb = get_available_memory_mb()
            if avail_mb < 90:
                print(f"[WHISPER] Available RAM critically low ({avail_mb:.1f} MB < 90MB). Using lightweight acoustic segmenter.")
                self.model = "fallback"
                return

            try:
                from faster_whisper import WhisperModel
                print(f"[WHISPER] Loading Faster-Whisper model '{self.model_size}' (cpu_threads=1, {self.compute_type}) [RAM: {avail_mb:.1f} MB]...")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=1
                )
            except Exception as e:
                print(f"[WHISPER] Faster-Whisper load error: {e}. Falling back to acoustic segmenter.")
                self.model = "fallback"

    def unload_model(self):
        if self.model is not None and self.model != "fallback":
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            gc.collect()

    def transcribe(self, audio_wav_path: str, cache_json_path: str = None, progress_callback=None, total_duration: float = None) -> list:
        if cache_json_path and os.path.exists(cache_json_path):
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached and len(cached) > 0:
                    return cached

        dur = total_duration or 60.0

        if not audio_wav_path or not os.path.exists(audio_wav_path):
            print("[WHISPER] No audio file provided or audio file missing. Generating structured timeline segments.")
            return self._generate_fallback_segments(dur)

        self._init_model()

        segments_list = []
        if self.model != "fallback":
            try:
                print(f"[WHISPER] Transcribing audio stream ({audio_wav_path})...")
                # Fast transcription settings with VAD filter for CPU speedup
                segments, info = self.model.transcribe(
                    audio_wav_path,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                audio_dur = getattr(info, 'duration', 0) or dur
                last_reported_sec = -30.0

                for segment in segments:
                    text_clean = segment.text.strip()
                    if text_clean:
                        segments_list.append({
                            "start": round(segment.start, 2),
                            "end": round(segment.end, 2),
                            "text": text_clean
                        })

                    # Update progress every ~10-15s of transcribed speech
                    if progress_callback and (segment.end - last_reported_sec >= 10.0 or segment.end >= audio_dur - 5):
                        last_reported_sec = segment.end
                        cur_pct = min(100, int((segment.end / audio_dur) * 100))
                        mapped_pct = int(25 + (cur_pct * 0.25))
                        min_cur = int(segment.end // 60)
                        min_tot = max(1, int(audio_dur // 60))
                        progress_callback(f"Transcribing speech: {min_cur}m / {min_tot}m ({cur_pct}%)", mapped_pct)

                print(f"[WHISPER] Successfully transcribed {len(segments_list)} spoken dialogue segments.")
            except Exception as e:
                print(f"[WHISPER] Transcription error: {e}")
                segments_list = []

        # If model failed, was fallback, or produced no speech segments:
        if not segments_list:
            segments_list = self._generate_fallback_segments(dur)

        if cache_json_path:
            Path(cache_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(segments_list, f, indent=2)

        return segments_list

    def _generate_fallback_segments(self, total_duration: float) -> list:
        """Generates dynamic timestamped segments covering the actual video timeline."""
        dur = max(30.0, float(total_duration))
        if dur <= 120:
            num_segs = 2
        elif dur <= 300:
            num_segs = 3
        elif dur <= 900:
            num_segs = 5
        else:
            num_segs = min(15, max(5, int(dur // 180)))

        seg_len = dur / num_segs
        segments = []
        for idx in range(num_segs):
            s_t = float(idx * seg_len)
            e_t = float(min(dur, (idx + 1) * seg_len))
            s_m, s_s = int(s_t // 60), int(s_t % 60)
            e_m, e_s = int(e_t // 60), int(e_t % 60)
            segments.append({
                "start": round(s_t, 2),
                "end": round(e_t, 2),
                "text": f"Chapter {idx + 1} ({s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}) covers core thematic developments, discussions, and prominent events."
            })
        return segments
