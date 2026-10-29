import os
import math
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from training.model import MultimodalImportanceModel


class VideoImportanceService:
    def __init__(self, model_path: str = "models/video_importance_model.pt", device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.text_embedder = None
        self.model = None

    def _init_model(self):
        if self.text_embedder is None:
            self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        if self.model is None:
            if not os.path.exists(self.model_path):
                print(f"[MODEL] Warning: checkpoint '{self.model_path}' not found. Using untrained weights.")
                self.model = MultimodalImportanceModel(
                    visual_dim=512, text_dim=384, fused_dim=256, hidden_dim=128
                ).to(self.device)
            else:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = MultimodalImportanceModel(
                    visual_dim=512, text_dim=384, fused_dim=256, hidden_dim=128
                ).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[MODEL] Loaded checkpoint from {self.model_path}")
            self.model.eval()

    def _dynamic_target_count(self, num_chunks: int, top_k_cap: int) -> int:
        """
        Scale the target number of selected sections with total chunks.
        For long videos (e.g., 40-45 mins with 15 chunks), select 10 to 15 sections
        to ensure full coverage from 00:00 to end.
        """
        if num_chunks <= 15:
            return max(min(10, num_chunks), num_chunks)
        target = max(10, math.ceil(num_chunks * 0.85))
        return min(target, top_k_cap)

    def _coverage_aware_select(self, scored_chunks: list, target_n: int) -> list:
        """
        Select `target_n` chunks while guaranteeing coverage of the full timeline.
        Divides the full video duration into `target_n` equal temporal windows,
        ensuring every part of the timeline (00:00 -> end) is represented.
        """
        if not scored_chunks:
            return []

        n = len(scored_chunks)
        if target_n >= n:
            return sorted(scored_chunks, key=lambda x: x["chunk_id"])

        total_start = scored_chunks[0]["start_time"]
        total_end = scored_chunks[-1]["end_time"]
        total_dur = max(1.0, total_end - total_start)
        window_size = total_dur / target_n

        selected = []
        used_ids = set()

        for w in range(target_n):
            win_start = total_start + w * window_size
            win_end = total_start + (w + 1) * window_size
            win_mid = (win_start + win_end) / 2.0

            # Chunks whose midpoint is within this window
            candidates = [
                c for c in scored_chunks
                if c["chunk_id"] not in used_ids
                and win_start <= (c["start_time"] + c["end_time"]) / 2.0 <= win_end
            ]

            if candidates:
                # Pick highest-scoring chunk in this window
                best = max(candidates, key=lambda c: c["importance_score"])
            else:
                # Pick nearest unused chunk to this window's centre
                all_unused = [c for c in scored_chunks if c["chunk_id"] not in used_ids]
                if not all_unused:
                    continue
                best = min(all_unused, key=lambda c: abs((c["start_time"] + c["end_time"]) / 2.0 - win_mid))

            selected.append(best)
            used_ids.add(best["chunk_id"])

        # Sort chronologically
        return sorted(selected, key=lambda x: x["chunk_id"])

    def score_chunks(self, chunks: list, top_k: int = 15) -> tuple:
        """
        Score every chunk then select sections using coverage-aware logic.
        """
        self._init_model()

        if not chunks:
            return [], []

        # ── Build feature tensors ─────────────────────────────────────────────
        v_feats_list, t_feats_list = [], []
        for chunk in chunks:
            v_feat = chunk.get("visual_features", [0.0] * 512)
            v_feats_list.append(v_feat)
            t_feat = self.text_embedder.encode(chunk.get("transcript", "")).tolist()
            t_feats_list.append(t_feat)

        v_tensor = torch.tensor([v_feats_list], dtype=torch.float32).to(self.device)  # (1, seq, 512)
        t_tensor = torch.tensor([t_feats_list], dtype=torch.float32).to(self.device)  # (1, seq, 384)

        with torch.no_grad():
            scores_tensor = self.model(v_tensor, t_tensor).squeeze(0).squeeze(-1)  # (seq,)
            scores = scores_tensor.cpu().numpy().tolist()

        if isinstance(scores, float):
            scores = [scores]

        # Ensure dynamic contrast across segment importance scores
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 0.15:
            # Scale scores across 0.45 to 0.95 using raw model output + visual/transcript novelty
            scaled_scores = []
            for i, chunk in enumerate(chunks):
                t_len = len(chunk.get("transcript", "").split())
                v_std = float(np.std(v_feats_list[i])) if v_feats_list[i] else 0.05
                base = 0.50 + 0.35 * (i / max(1, len(chunks)-1)) * 0.2 + (scores[i] - min_s)
                richness = min(0.35, t_len / 200.0) + min(0.15, v_std * 2.0)
                final_score = round(min(0.96, max(0.42, base + richness)), 2)
                scaled_scores.append(final_score)
            scores = scaled_scores

        # ── Attach scores ──────────────────────────────────────────────────────
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            c = dict(chunk)
            c["importance_score"] = round(float(scores[i]), 2)
            scored_chunks.append(c)

        print(f"[MODEL] Scores for {len(scored_chunks)} chunks: "
              f"min={min(scores):.2f} max={max(scores):.2f} mean={sum(scores)/len(scores):.2f}")

        # ── Dynamic target count then coverage-aware selection ─────────────────
        target_n = self._dynamic_target_count(len(scored_chunks), top_k)
        print(f"[MODEL] Coverage-aware selection: {len(scored_chunks)} candidates -> target {target_n} sections")

        selected_chunks = self._coverage_aware_select(scored_chunks, target_n)
        print(f"[MODEL] Final selected sections: {len(selected_chunks)}")
        for c in selected_chunks:
            s, e = c["start_time"], c["end_time"]
            print(f"        [{int(s//60):02d}:{int(s%60):02d} - {int(e//60):02d}:{int(e%60):02d}] "
                  f"score={c['importance_score']:.2f}")

        return scored_chunks, selected_chunks
