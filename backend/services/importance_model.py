import os
import math
import hashlib
import torch
import numpy as np
from training.model import MultimodalImportanceModel
from backend.utils.memory_utils import get_available_memory_mb

class VideoImportanceService:
    def __init__(self, model_path: str = "models/video_importance_model.pt", device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.text_embedder = None
        self.model = None

    def _fallback_text_embed(self, text: str) -> list:
        """
        Memory-safe 384-dimensional text feature vector.
        Maps word n-grams into a 384-D representation with term frequency weighting.
        Uses 0 MB of model RAM and runs instantaneously.
        """
        vec = [0.0] * 384
        words = text.lower().split()
        if not words:
            return vec
        for i, w in enumerate(words):
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % 384
            vec[idx] += 1.0
            # Bi-gram
            if i + 1 < len(words):
                bg = f"{w}_{words[i+1]}"
                h_bg = int(hashlib.md5(bg.encode('utf-8')).hexdigest(), 16)
                vec[h_bg % 384] += 0.5

        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _init_model(self):
        avail_mb = get_available_memory_mb()

        # 1. Text Embedder
        if self.text_embedder is None:
            if avail_mb < 220:
                print(f"[MODEL] Available RAM is {avail_mb:.1f} MB (< 220MB). Using memory-safe text embedding.")
                self.text_embedder = "lightweight"
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    print(f"[MODEL] Loading SentenceTransformer [RAM: {avail_mb:.1f} MB]...")
                    self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                except Exception as e:
                    print(f"[MODEL] SentenceTransformer load error: {e}. Using lightweight fallback.")
                    self.text_embedder = "lightweight"

        # 2. PyTorch Multimodal Importance Model (4MB - lightweight)
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
                print(f"[MODEL] Loaded trained checkpoint from {self.model_path}")
            self.model.eval()

    def _dynamic_target_count(self, num_chunks: int, top_k_cap: int) -> int:
        if num_chunks <= 15:
            return max(min(10, num_chunks), num_chunks)
        target = max(10, math.ceil(num_chunks * 0.85))
        return min(target, top_k_cap)

    def _coverage_aware_select(self, scored_chunks: list, target_n: int) -> list:
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

            candidates = [
                c for c in scored_chunks
                if c["chunk_id"] not in used_ids
                and win_start <= (c["start_time"] + c["end_time"]) / 2.0 <= win_end
            ]

            if candidates:
                best = max(candidates, key=lambda c: c["importance_score"])
            else:
                all_unused = [c for c in scored_chunks if c["chunk_id"] not in used_ids]
                if not all_unused:
                    continue
                best = min(all_unused, key=lambda c: abs((c["start_time"] + c["end_time"]) / 2.0 - win_mid))

            selected.append(best)
            used_ids.add(best["chunk_id"])

        return sorted(selected, key=lambda x: x["chunk_id"])

    def score_chunks(self, chunks: list, top_k: int = 15) -> tuple:
        self._init_model()

        if not chunks:
            return [], []

        # ── Build feature tensors ─────────────────────────────────────────────
        v_feats_list, t_feats_list = [], []
        for chunk in chunks:
            v_feat = chunk.get("visual_features", [0.0] * 512)
            v_feats_list.append(v_feat)

            t_text = chunk.get("transcript", "")
            if self.text_embedder == "lightweight" or self.text_embedder is None:
                t_feat = self._fallback_text_embed(t_text)
            else:
                try:
                    t_feat = self.text_embedder.encode(t_text).tolist()
                except Exception:
                    t_feat = self._fallback_text_embed(t_text)
            t_feats_list.append(t_feat)

        v_tensor = torch.tensor([v_feats_list], dtype=torch.float32).to(self.device)
        t_tensor = torch.tensor([t_feats_list], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores_tensor = self.model(v_tensor, t_tensor).squeeze(0).squeeze(-1)
            scores = scores_tensor.cpu().numpy().tolist()

        if isinstance(scores, float):
            scores = [scores]

        # Dynamic contrast scaling
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 0.15:
            scaled_scores = []
            for i, chunk in enumerate(chunks):
                t_len = len(chunk.get("transcript", "").split())
                v_std = float(np.std(v_feats_list[i])) if v_feats_list[i] else 0.05
                base = 0.50 + 0.35 * (i / max(1, len(chunks)-1)) * 0.2 + (scores[i] - min_s)
                richness = min(0.35, t_len / 200.0) + min(0.15, v_std * 2.0)
                final_score = round(min(0.96, max(0.42, base + richness)), 2)
                scaled_scores.append(final_score)
            scores = scaled_scores

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            c = dict(chunk)
            c["importance_score"] = round(float(scores[i]), 2)
            scored_chunks.append(c)

        print(f"[MODEL] Scores for {len(scored_chunks)} chunks: min={min(scores):.2f} max={max(scores):.2f}")

        target_n = self._dynamic_target_count(len(scored_chunks), top_k)
        selected_chunks = self._coverage_aware_select(scored_chunks, target_n)

        return scored_chunks, selected_chunks

    def unload_model(self):
        if self.text_embedder is not None and self.text_embedder != "lightweight":
            try:
                del self.text_embedder
            except Exception:
                pass
            self.text_embedder = None
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
        import gc
        gc.collect()
