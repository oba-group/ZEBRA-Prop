"""Neural network modules used by the ZEBRA-Prop training pipeline."""

import torch
import torch.nn as nn


class EmbeddingExtractor(nn.Module):
    """Wrap a Hugging Face encoder and return mean pooled sentence embeddings."""

    def __init__(self, base_model):
        """Initialize the extractor with a pretrained encoder model."""
        super().__init__()
        self.model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Compute mean pooled token embeddings for a batch."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Apply masked mean pooling to ignore padded token positions.
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        summed = (hidden_states * mask).sum(1)
        counts = mask.sum(1).clamp_min(1)
        return summed / counts


class ZEBRAProp(nn.Module):
    """Apply learnable description weights and predict a scalar with an MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_descriptions: int,
    ):
        """Build the ZEBRA-Prop regression head."""
        super().__init__()

        self.num_descriptions = num_descriptions
        self.hidden_size = hidden_size

        self.W_desc = nn.Parameter(torch.ones(num_descriptions, 1) / num_descriptions)
        in_dim = hidden_size

        lim_dim = max(1, in_dim // 6)
        self.hidden_dims = [lim_dim * 4, lim_dim * 2, lim_dim]

        # Use a compact MLP head after weighted aggregation over descriptions.
        layers = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor):
        """Predict target values from embedding tensor with shape (B, D, H)."""
        emb = emb.to(torch.float32)

        # Learn per-description weights, then collapse (B, D, H) -> (B, H).
        w = self.W_desc.view(1, -1, 1).to(emb.device)
        x = (w * emb).sum(dim=1)
        return self.regressor(x).squeeze(-1)
