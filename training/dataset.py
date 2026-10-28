import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalVideoDataset(Dataset):
    """
    Dataset for Video Importance Scoring.
    Each item represents a single VIDEO sequence containing N temporal segments/chunks,
    their visual features, text features, and ground-truth human importance scores.
    """
    def __init__(self, data_list: list, max_seq_len: int = 50):
        self.data_list = data_list
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        visual_feats = np.array(item["visual_features"], dtype=np.float32)
        text_feats = np.array(item["text_features"], dtype=np.float32)
        scores = np.array(item["importance_scores"], dtype=np.float32)

        seq_len = len(visual_feats)
        if seq_len > self.max_seq_len:
            visual_feats = visual_feats[:self.max_seq_len]
            text_feats = text_feats[:self.max_seq_len]
            scores = scores[:self.max_seq_len]
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            v_pad = np.zeros((pad_len, visual_feats.shape[1]), dtype=np.float32)
            t_pad = np.zeros((pad_len, text_feats.shape[1]), dtype=np.float32)
            s_pad = np.zeros((pad_len,), dtype=np.float32)

            visual_feats = np.vstack([visual_feats, v_pad])
            text_feats = np.vstack([text_feats, t_pad])
            scores = np.concatenate([scores, s_pad])

        return {
            "video_id": item.get("video_id", f"video_{idx}"),
            "visual_features": torch.tensor(visual_feats, dtype=torch.float32),
            "text_features": torch.tensor(text_feats, dtype=torch.float32),
            "importance_scores": torch.tensor(scores, dtype=torch.float32).unsqueeze(-1),
            "seq_len": min(seq_len, self.max_seq_len)
        }
