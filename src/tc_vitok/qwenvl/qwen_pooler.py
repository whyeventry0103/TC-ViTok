"""Trainable pooler bridge used for Qwen2-VL FM1 reproduction."""

from __future__ import annotations

import torch
from torch import nn

from tc_vitok.models.masking import gumbel_sigmoid_ste


class QwenTrainablePooler(nn.Module):
    """Cross-attention selector plus trainable summary token.

    The trainable summary token is deliberately a learnable bridge between the
    selector and the LoRA-adapted LLM. FM1 predicts this closed gradient circuit
    should oscillate under joint optimization.
    """

    def __init__(
        self,
        visual_dim: int,
        question_dim: int,
        d_model: int = 384,
        n_heads: int = 4,
        hidden_dim: int = 512,
        k: int = 144,
    ) -> None:
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(question_dim, d_model)
        self.v_proj = nn.Linear(visual_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, visual_dim),
        )

    @staticmethod
    def _gather(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, _, D = tokens.shape
        idx_expanded = idx.unsqueeze(-1).expand(B, idx.shape[1], D)
        return torch.gather(tokens, dim=1, index=idx_expanded)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_embedding: torch.Tensor,
        tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Return selected visual tokens, trainable pooled token, and scores."""
        B, N, _ = visual_tokens.shape
        k = min(self.k, N)
        q = self.q_proj(question_embedding).unsqueeze(1)
        v = self.v_proj(visual_tokens)
        ctx, _ = self.attn(q, v, v)
        conditioned = v + ctx.expand(-1, N, -1)
        scores = self.score(conditioned).squeeze(-1)
        mask_ste, idx = gumbel_sigmoid_ste(scores, k=k, tau=tau, training=self.training)
        selected = self._gather(visual_tokens, idx)
        selected_scores = self._gather(scores.unsqueeze(-1), idx)
        # Forward value is unchanged, but the selected-score multiplier routes
        # VQA gradients into the score head even when the Gumbel mask saturates.
        selected = selected * (1.0 + selected_scores - selected_scores.detach())
        soft_summary = torch.bmm(mask_ste.unsqueeze(1), visual_tokens) / float(k)
        selected = selected + (soft_summary - soft_summary.detach()).expand(-1, k, -1)
        pooled = self.summary(ctx.squeeze(1)).unsqueeze(1)
        return {
            "selected_tokens": selected,
            "pooled_token": pooled,
            "scores": scores,
            "mask_ste": mask_ste,
            "idx": idx,
            "llm_visual_tokens": torch.cat([selected, pooled], dim=1),
        }
