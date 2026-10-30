import os
import sys
import gc
from pathlib import Path
from backend.services.video_processor import VideoProcessor
from backend.services.audio_processor import AudioProcessor
from backend.services.frame_processor import FrameProcessor
from backend.services.temporal_alignment import TemporalAligner
from backend.services.importance_model import VideoImportanceService
from backend.services.summarizer import HierarchicalSummarizer
from backend.utils.config_loader import load_config

class VideoSummarizationPipeline:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.video_processor = VideoProcessor(self.config)
        self.audio_processor = AudioProcessor(self.config)
        self.frame_processor = FrameProcessor(self.config)

        sys_cfg = self.config.get("system", {})
        chunk_sec = sys_cfg.get("chunk_duration_seconds", 180)
        self.temporal_aligner = TemporalAligner(chunk_duration_seconds=chunk_sec)

        self.importance_service = VideoImportanceService(
            model_path=self.config.get("paths", {}).get("model_save_path", "models/video_importance_model.pt"),
            device=sys_cfg.get("device", "cpu")
        )

        self.summarizer = HierarchicalSummarizer(self.config)

    def process_long_video(self, video_path: str, progress_callback=None) -> dict:
        video_path = str(Path(video_path).resolve())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        def notify(stage: str, percentage: int):
            if progress_callback:
                progress_callback(stage, percentage)

        # 1. Video Metadata Extraction
        notify("Extracting video metadata", 15)
        metadata = self.video_processor.extract_metadata(video_path)
        dur_sec = metadata.get("duration_seconds", 0)

        # 2 & 3. Audio Extraction & Whisper Transcription
        print(f"[5] Audio extraction started")
        notify("Extracting audio stream...", 22)
        wav_path, transcript = self.audio_processor.process_audio(
            video_path,
            progress_callback=notify,
            total_duration=dur_sec
        )
        print(f"[11] Audio processing and transcription completed. Sentences count: {len(transcript)}")

        # -- Unload Whisper model after transcription to free ~40 MB RAM --
        self.audio_processor.whisper_service.unload_model()
        gc.collect()
        print("[MEM] Unloaded Whisper model")

        # 4. Frame Sampling & Visual Feature Extraction
        print(f"[7] Video feature extraction started")
        notify("Sampling video frames and analyzing visual scenes...", 50)
        visual_frames = self.frame_processor.process_video_frames(
            video_path,
            progress_callback=notify
        )

        # -- Unload ResNet-18 after frame processing to free ~50 MB RAM --
        self.frame_processor.unload_model()
        gc.collect()
        print("[MEM] Unloaded ResNet-18 visual feature model")

        # 5. Multimodal Temporal Alignment into chunks
        notify("Performing multimodal temporal alignment", 65)
        chunks = self.temporal_aligner.align_and_chunk(
            duration_seconds=metadata["duration_seconds"],
            transcript=transcript,
            visual_frames=visual_frames
        )

        # 6. Segment Selection via Trained Video Importance Model
        print(f"[8] Importance model loaded")
        notify("Scoring segments using trained PyTorch Importance Model", 80)
        top_k = self.config.get("model", {}).get("top_k_chunks", 15)
        scored_chunks, selected_chunks = self.importance_service.score_chunks(chunks, top_k=top_k)
        print(f"[9] Importance prediction completed. Selected {len(selected_chunks)} chunks.")

        # -- Unload SentenceTransformer + BiLSTM to free ~140 MB RAM --
        self.importance_service.unload_model()
        gc.collect()
        print("[MEM] Unloaded importance scoring models")

        # 7. Hierarchical LLM Summarization with progressive updates
        print(f"[12] Text summarization started")
        total_sel = len(selected_chunks)
        chunk_summaries = []
        for i, c in enumerate(selected_chunks):
            pct = int(85 + ((i + 1) / max(1, total_sel)) * 12)
            notify(f"Summarizing chapter {i + 1} of {total_sel}...", pct)
            chunk_summaries.append(self.summarizer.generate_chunk_summary(c))

        notify("Finalizing executive summary...", 98)
        final_output = self.summarizer.generate_final_summary(
            video_metadata=metadata,
            selected_chunks=selected_chunks,
            chunk_summaries=chunk_summaries
        )

        # -- Unload T5 summarizer to free ~60 MB RAM --
        self.summarizer.unload_model()
        gc.collect()
        print("[MEM] Unloaded T5 summarizer model")

        # Attach execution details
        final_output["metadata"] = metadata
        final_output["total_chunks_processed"] = len(chunks)
        final_output["selected_chunks_count"] = len(selected_chunks)

        notify("Summarization complete!", 100)
        return final_output
