import torch
import torch.nn as nn

class MultimodalImportanceModel(nn.Module):
    """
    Multimodal Video Importance Scoring Model.
    Fuses visual features (e.g. 512-dim ResNet/CLIP) and textual features (e.g. 384-dim SentenceTransformer)
    through a Multimodal Fusion MLP, passes sequence context through a BiLSTM layer,
    and outputs a segment importance score between 0 and 1.
    """
    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 384,
        fused_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(MultimodalImportanceModel, self).__init__()
        
        # 1. Text Projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Visual Projection
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Fusion Layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Temporal Sequence Encoder (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 5. Importance Score Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: Tensor of shape (batch_size, seq_len, visual_dim)
            text_feats: Tensor of shape (batch_size, seq_len, text_dim)
        Returns:
            importance_scores: Tensor of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = visual_feats.size()
        
        # Reshape for 2D batch norm & projection
        v_flat = visual_feats.contiguous().view(batch_size * seq_len, -1)
        t_flat = text_feats.contiguous().view(batch_size * seq_len, -1)
        
        v_proj = self.visual_proj(v_flat)
        t_proj = self.text_proj(t_flat)
        
        fused_flat = torch.cat([v_proj, t_proj], dim=-1)
        fused_flat = self.fusion_fc(fused_flat)
        
        fused_seq = fused_flat.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.bilstm(fused_seq)
        
        scores = self.classifier(lstm_out)  # (batch_size, seq_len, 1)
        return scores
