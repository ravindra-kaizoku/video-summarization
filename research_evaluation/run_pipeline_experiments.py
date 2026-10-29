"""
research_evaluation/run_pipeline_experiments.py
===============================================
Executes:
- EXP-9: End-to-End Pipeline Profiling across real long-format videos.
- EXP-10: Text Summarization Performance and Compression Analysis.
"""

import os
import sys
import time
import json
import numpy as np

# Force offline mode for HuggingFace / Transformers to load instantly from local disk cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.video_processor import VideoProcessor
from backend.services.temporal_alignment import TemporalAligner
from backend.services.importance_model import VideoImportanceService
from backend.services.summarizer import HierarchicalSummarizer
from backend.utils.config_loader import load_config

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def exp9_end_to_end_profiling():
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: End-to-End Pipeline Profiling on Real Video Assets")
    print("=" * 60)

    config = load_config()
    vp = VideoProcessor(config)
    aligner = TemporalAligner(chunk_duration_seconds=180)
    importance_service = VideoImportanceService(device="cpu")
    summarizer = HierarchicalSummarizer(config)

    test_vids = [
        os.path.join(PROJECT_ROOT, "uploads", "test_video_30min.mp4"),
        os.path.join(PROJECT_ROOT, "uploads", "diagnostic_test_001.mp4"),
        os.path.join(PROJECT_ROOT, "uploads", "e8b793cf-28a5-4eb1-a6b6-d2e4eb75e9c4.mp4"),
    ]

    profiling_records = []

    for vpath in test_vids:
        if not os.path.exists(vpath):
            continue

        vname = os.path.splitext(os.path.basename(vpath))[0]
        transcript_cache = os.path.join(PROJECT_ROOT, "cache", "transcripts", f"{vname}.json")
        features_cache = os.path.join(PROJECT_ROOT, "cache", "visual_features", f"{vname}.json")

        if not os.path.exists(transcript_cache) or not os.path.exists(features_cache):
            print(f"  Skipping {vname} (cache not found)")
            continue

        print(f"\n  Profiling video: {vname}")
        t0 = time.perf_counter()

        # Step 1: Metadata
        t_meta0 = time.perf_counter()
        meta = vp.extract_metadata(vpath)
        t_meta = time.perf_counter() - t_meta0

        # Step 2 & 3: Load transcript (simulating cached audio STT)
        t_stt0 = time.perf_counter()
        with open(transcript_cache, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        t_stt = time.perf_counter() - t_stt0

        # Step 4: Load visual features
        t_vis0 = time.perf_counter()
        with open(features_cache, "r", encoding="utf-8") as f:
            visual_frames = json.load(f)
        t_vis = time.perf_counter() - t_vis0

        # Step 5: Temporal Alignment into Chunks
        t_align0 = time.perf_counter()
        chunks = aligner.align_and_chunk(
            duration_seconds=meta["duration_seconds"],
            transcript=transcript,
            visual_frames=visual_frames
        )
        t_align = time.perf_counter() - t_align0

        # Step 6: Importance Scoring & Selection
        t_score0 = time.perf_counter()
        scored_chunks, selected_chunks = importance_service.score_chunks(chunks, top_k=15)
        t_score = time.perf_counter() - t_score0

        # Step 7: Hierarchical Summarization (run on selected chunks)
        t_sum0 = time.perf_counter()
        chunk_summaries = []
        for c in selected_chunks:
            chunk_summaries.append(summarizer.generate_chunk_summary(c))

        final_summary = summarizer.generate_final_summary(
            video_metadata=meta,
            selected_chunks=selected_chunks,
            chunk_summaries=chunk_summaries
        )
        t_sum = time.perf_counter() - t_sum0

        total_time = time.perf_counter() - t0
        dur_sec = max(1.0, meta["duration_seconds"])
        rtf = total_time / dur_sec

        rec = {
            "video_name": vname,
            "duration_minutes": meta["duration_minutes"],
            "duration_seconds": meta["duration_seconds"],
            "resolution": meta["resolution"],
            "total_frames_extracted": len(visual_frames),
            "transcript_segments": len(transcript),
            "total_chunks": len(chunks),
            "selected_chunks": len(selected_chunks),
            "stage_latency_seconds": {
                "metadata_extraction": round(t_meta, 4),
                "stt_loading": round(t_stt, 4),
                "visual_feature_loading": round(t_vis, 4),
                "temporal_alignment": round(t_align, 4),
                "importance_scoring_and_selection": round(t_score, 4),
                "t5_summarization": round(t_sum, 4),
            },
            "total_latency_seconds": round(total_time, 4),
            "real_time_factor": round(rtf, 6),
            "summary_word_count": len(final_summary["summary"].split()),
            "key_points_count": len(final_summary["key_points"]),
            "sample_summary": final_summary["summary"][:200] + "...",
        }
        profiling_records.append(rec)

        print(f"    Duration: {meta['duration_minutes']:.1f} min | Chunks: {len(chunks)} -> Selected: {len(selected_chunks)}")
        print(f"    Importance Scoring Latency: {t_score*1000:.1f} ms")
        print(f"    T5 Summarization Latency: {t_sum:.2f} s")
        print(f"    Total Time: {total_time:.2f} s (RTF: {rtf:.4f})")

    out_file = os.path.join(OUTPUT_DIR, "exp9_pipeline_profiling.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(profiling_records, f, indent=2)
    print(f"\n  Results saved to: {out_file}")
    return profiling_records


def exp10_summarization_evaluation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Hierarchical Text Summarization Analysis")
    print("=" * 60)

    config = load_config()
    summarizer = HierarchicalSummarizer(config)

    # Test inputs with varied transcript lengths
    sample_transcripts = [
        "In this lecture, we introduce deep convolutional neural networks for computer vision. We explore convolutional filters, pooling layers, and residual connections. The architecture allows hierarchical feature extraction from low-level edges to high-level semantic representations.",
        "Today we cover database indexing and query optimization. B-trees provide logarithmic search guarantees, while hash indexes offer constant-time point lookups. We also discuss multi-column indexes and explain query execution plans.",
        "The experimental results demonstrate that our proposed multimodal fusion model reduces mean squared error significantly while maintaining high temporal coverage across 45-minute videos. We analyze ablation results and threshold sensitivity.",
    ]

    summaries = []
    generation_times = []

    for idx, text in enumerate(sample_transcripts):
        chunk = {
            "chunk_id": idx + 1,
            "start_time": idx * 180.0,
            "end_time": (idx + 1) * 180.0,
            "transcript": text
        }
        t0 = time.perf_counter()
        summary = summarizer.generate_chunk_summary(chunk)
        gen_time = time.perf_counter() - t0

        generation_times.append(gen_time)
        input_words = len(text.split())
        output_words = len(summary.split())
        compression = 1.0 - (output_words / max(1, input_words))

        summaries.append({
            "chunk_id": idx + 1,
            "input_words": input_words,
            "output_words": output_words,
            "compression_ratio": round(compression, 3),
            "generation_time_sec": round(gen_time, 3),
            "generated_summary": summary,
        })
        print(f"  Chunk {idx+1}: {input_words} words -> {output_words} words ({compression*100:.1f}% compression) in {gen_time:.3f} s")
        print(f"    Summary: {summary}")

    res = {
        "model_name": "t5-small",
        "sample_evaluations": summaries,
        "mean_generation_time_sec": round(float(np.mean(generation_times)), 3),
        "mean_compression_ratio": round(float(np.mean([s["compression_ratio"] for s in summaries])), 3),
    }

    out_file = os.path.join(OUTPUT_DIR, "exp10_summarization_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Results saved to: {out_file}")
    return res


if __name__ == "__main__":
    exp9_end_to_end_profiling()
    exp10_summarization_evaluation()
