import numpy as np

class TemporalAligner:
    def __init__(self, chunk_duration_seconds: int = 180):
        self.chunk_duration = chunk_duration_seconds

    def align_and_chunk(self, duration_seconds: float, transcript: list, visual_frames: list) -> list:
        num_chunks = max(1, int(np.ceil(duration_seconds / self.chunk_duration)))
        chunks = []

        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            end_time = min(duration_seconds, (i + 1) * self.chunk_duration)

            # 1. Filter transcript segments within chunk time bounds
            chunk_transcript_segments = [
                seg["text"] for seg in transcript
                if seg["start"] < end_time and seg["end"] >= start_time
            ]
            chunk_transcript_text = " ".join(chunk_transcript_segments).strip()
            if not chunk_transcript_text:
                chunk_transcript_text = f"Discussion and key presentation events occurring between {int(start_time//60):02d}:{int(start_time%60):02d} and {int(end_time//60):02d}:{int(end_time%60):02d}."

            # 2. Filter visual frames within chunk time bounds
            chunk_frames = [
                vf for vf in visual_frames
                if start_time <= vf["timestamp"] < end_time
            ]

            # Aggregate visual feature vectors (mean pooling across frames in chunk)
            if chunk_frames:
                visual_features_matrix = np.array([vf["features"] for vf in chunk_frames])
                mean_visual_feature = np.mean(visual_features_matrix, axis=0).tolist()
                visual_descriptions = [vf["description"] for vf in chunk_frames]
            else:
                mean_visual_feature = [0.0] * 512
                visual_descriptions = ["Visual scene segment."]

            chunks.append({
                "chunk_id": i + 1,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "transcript": chunk_transcript_text,
                "visual_information": visual_descriptions,
                "visual_features": mean_visual_feature,
                "frame_count": len(chunk_frames)
            })

        return chunks
