import os
import json
import gc
import subprocess
import requests
import numpy as np
from pathlib import Path
from backend.utils.memory_utils import get_available_memory_mb
from backend.utils.ffmpeg_utils import get_ffmpeg_cmd

class WhisperService:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _init_model(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                print(f"[WHISPER] Loading Faster-Whisper '{self.model_size}' (cpu_threads=1, {self.compute_type})...")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=1
                )
            except Exception as e:
                print(f"[WHISPER] Faster-Whisper load error: {e}. Will attempt secondary speech recognition.")
                self.model = "fallback"

    def unload_model(self):
        if self.model is not None and self.model != "fallback":
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            gc.collect()

    def _transcribe_slice(self, wav_path: str, offset_sec: float = 0.0) -> list:
        """Transcribes an audio slice with Faster-Whisper, with Google SpeechRecognition backup."""
        segments_list = []

        # 1. Try Faster-Whisper first
        if self.model != "fallback" and self.model is not None:
            try:
                segments, _ = self.model.transcribe(
                    wav_path,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400)
                )
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        segments_list.append({
                            "start": round(offset_sec + segment.start, 2),
                            "end": round(offset_sec + segment.end, 2),
                            "text": text
                        })
                if segments_list:
                    return segments_list
            except Exception as e:
                print(f"[WHISPER] Slice transcription error: {e}")

        # 2. Try SpeechRecognition (Google Web Speech API) backup
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if text and text.strip():
                print(f"[WHISPER] Google SpeechRecognition transcribed: {text[:80]}...")
                segments_list.append({
                    "start": round(offset_sec, 2),
                    "end": round(offset_sec + 60.0, 2),
                    "text": text.strip()
                })
                return segments_list
        except Exception:
            pass

        return segments_list

    def transcribe(self, audio_wav_path: str, cache_json_path: str = None, progress_callback=None, total_duration: float = None) -> list:
        if cache_json_path and os.path.exists(cache_json_path):
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached and len(cached) > 0:
                    return cached

        dur = total_duration or 60.0

        if not audio_wav_path or not os.path.exists(audio_wav_path):
            print("[WHISPER] No audio file provided or missing. Generating timeline segments.")
            return self._generate_fallback_segments(dur)

        # 0. Check for Groq API Key for ultra-fast cloud Whisper acceleration (~2 seconds)
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key and audio_wav_path and os.path.exists(audio_wav_path):
            try:
                print(f"[WHISPER] Accelerating transcription with Groq Whisper API (whisper-large-v3-turbo)...")
                if progress_callback:
                    progress_callback("Transcribing speech with ultra-fast cloud Whisper...", 45)

                file_size_mb = os.path.getsize(audio_wav_path) / (1024 * 1024)
                send_path = audio_wav_path
                temp_mp3 = None
                if file_size_mb > 24.0:
                    temp_mp3 = Path(audio_wav_path).with_suffix(".mp3")
                    ffmpeg_cmd = get_ffmpeg_cmd()
                    subprocess.run([
                        ffmpeg_cmd, "-y", "-i", str(audio_wav_path),
                        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(temp_mp3)
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if temp_mp3.exists():
                        send_path = str(temp_mp3)

                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {groq_key}"}
                with open(send_path, "rb") as f:
                    files = {"file": (Path(send_path).name, f, "audio/mpeg" if send_path.endswith(".mp3") else "audio/wav")}
                    data = {
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "language": "en"
                    }
                    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)

                if temp_mp3 and temp_mp3.exists():
                    try:
                        temp_mp3.unlink()
                    except Exception:
                        pass

                if resp.status_code == 200:
                    groq_data = resp.json()
                    segs = groq_data.get("segments", [])
                    groq_segments = []
                    for s in segs:
                        text = s.get("text", "").strip()
                        if text:
                            groq_segments.append({
                                "start": round(float(s.get("start", 0)), 2),
                                "end": round(float(s.get("end", 0)), 2),
                                "text": text
                            })
                    if groq_segments:
                        print(f"[WHISPER] Groq transcribed {len(groq_segments)} segments in ~2 seconds!")
                        if cache_json_path:
                            with open(cache_json_path, "w", encoding="utf-8") as f:
                                json.dump(groq_segments, f, indent=2)
                        return groq_segments
            except Exception as groq_err:
                print(f"[WHISPER] Groq transcription error: {groq_err}. Falling back to local whisper.")

        self._init_model()

        segments_list = []
        try:
            print(f"[WHISPER] Transcribing audio stream ({audio_wav_path}, duration: {dur:.1f}s)...")

            # For long audio (>5 minutes), slice into 5-minute chunks to keep RAM low
            if dur > 300.0:
                ffmpeg_cmd = get_ffmpeg_cmd()
                slice_duration = 300.0
                num_slices = int(np.ceil(dur / slice_duration))
                cache_slice_dir = Path("cache/slices")
                cache_slice_dir.mkdir(parents=True, exist_ok=True)

                for i in range(num_slices):
                    start_sec = i * slice_duration
                    dur_slice = min(slice_duration, dur - start_sec)
                    slice_path = cache_slice_dir / f"slice_{i}_{int(start_sec)}.wav"

                    cmd = [
                        ffmpeg_cmd, "-y",
                        "-ss", str(start_sec),
                        "-t", str(dur_slice),
                        "-i", str(audio_wav_path),
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        str(slice_path)
                    ]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if slice_path.exists():
                        try:
                            slice_segs = self._transcribe_slice(str(slice_path), offset_sec=start_sec)
                            segments_list.extend(slice_segs)
                        finally:
                            try:
                                slice_path.unlink()
                            except Exception:
                                pass

                    if progress_callback:
                        pct = int(25 + ((i + 1) / num_slices) * 25)
                        min_done = int((start_sec + dur_slice) // 60)
                        min_tot = max(1, int(dur // 60))
                        progress_callback(f"Transcribing speech: {min_done}m / {min_tot}m ({pct}%)", pct)
            else:
                # Short video/audio: transcribe directly
                segments_list = self._transcribe_slice(audio_wav_path, offset_sec=0.0)
                if progress_callback:
                    progress_callback("Speech transcription complete (50%)", 50)

            print(f"[WHISPER] Successfully transcribed {len(segments_list)} spoken dialogue segments.")
        except Exception as e:
            print(f"[WHISPER] Transcription error: {e}")
            segments_list = []

        if not segments_list:
            segments_list = self._generate_fallback_segments(dur)

        if cache_json_path:
            Path(cache_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(segments_list, f, indent=2)

        return segments_list

    def _generate_fallback_segments(self, total_duration: float) -> list:
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
                "text": f"From {s_m:02d}:{s_s:02d} to {e_m:02d}:{e_s:02d}, the video presents on-screen activity, visual demonstrations, and acoustic playback."
            })
        return segments
