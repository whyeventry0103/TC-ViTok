"""Tokenization Policy Network (TPN) for TC-ViTok.

This module implements the core TPN that learns a query-conditioned token selection
policy π: (Q, I_coarse) → S. The network uses cross-attention to fuse coarse visual
features with text embeddings, then produces per-token importance scores that are
used with Gumbel-Sigmoid STE to select the top-k most relevant tokens.

Architecture (paper Section 5.3, plan Section 5):
1. Cross-attention: query embedding attends to coarse visual features
2. Head-averaged attention weights are preserved as spatial signal
3. Concat [context; α_avg] → Score MLP → (B, 49) coarse scores
4. Bilinear upsampling: 7×7 → 24×24 to match full encoder resolution
5. Gumbel-Sigmoid STE: differentiable top-k selection

Parameter count: ~887K (see paper Appendix A.1)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TokenizationPolicyNetwork(nn.Module):
    """Lightweight policy network for learned visual token selection.

    The TPN takes coarse visual features (DINOv2-S, 49 tokens) and text embeddings
    (CLIP, 512-dim) as input, and outputs a binary selection mask over the full
    visual token grid (576 tokens from CLIP ViT-L/14).

    Architecture matches paper exactly:
    - Cross-attention with 4 heads, d_model=384
    - Score MLP input = concat(context_vector, head_avg_attention_weights)
    - Score MLP: (d_model + M) → d_h → M
    - Bilinear upsample: 7×7 → 24×24

    Args:
        d_model: Shared attention dimension (384)
        d_text: Dimension of text embeddings (512 for CLIP ViT-B/32)
        d_coarse: Dimension of coarse visual features (384 for DINOv2-S)
        n_heads: Number of attention heads (4, per paper)
        d_h: Hidden dimension for score MLP (512)
        M: Number of coarse spatial tokens (49 = 7×7)
        N: Number of full-resolution tokens (576 = 24×24)
        dropout: Dropout probability

    Total parameters: ~887K (see paper Appendix A.1)
    """

    def __init__(
        self,
        d_model: int = 384,
        d_text: int = 512,
        d_coarse: int = 384,
        n_heads: int = 4,
        d_h: int = 512,
        M: int = 49,
        N: int = 576,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_text = d_text
        self.d_coarse = d_coarse
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_h = d_h
        self.M = M
        self.N = N
        self.sqrt_M = int(M ** 0.5)  # 7
        self.sqrt_N = int(N ** 0.5)  # 24

        # --- Cross-attention projections (Eq. 3-4 in paper) ---
        # W^Q: maps CLIP text space → shared attention space
        self.q_proj = nn.Linear(d_text, d_model, bias=False)
        # W^K: maps DINOv2 space → shared attention space
        self.k_proj = nn.Linear(d_coarse, d_model, bias=False)
        # W^V: maps DINOv2 space → shared attention space
        self.v_proj = nn.Linear(d_coarse, d_model, bias=False)
        # W^O: output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Layer normalization after cross-attention
        self.norm1 = nn.LayerNorm(d_model)

        # --- Score MLP (Eq. 5-6 in paper) ---
        # Input: concat(context_vector, head_avg_attn_weights) → (d_model + M)
        self.norm2 = nn.LayerNorm(d_model + M)
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model + M, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, M),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        v_coarse: torch.Tensor,
        q_embed: torch.Tensor,
        k: int,
        tau: float = 1.0,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Tokenization Policy Network.

        Implements the paper's architecture exactly:
        1. Project q_embed and v_coarse into shared space
        2. Multi-head cross-attention (text queries visual)
        3. Average attention weights across heads → α_avg
        4. Concat [context; α_avg] → Score MLP → coarse scores
        5. Bilinear upsample 7×7 → 24×24
        6. Gumbel-Sigmoid STE for differentiable top-k

        Args:
            v_coarse: Coarse visual features from DINOv2-S
                      Shape: [B, 49, d_coarse]
            q_embed: Text embeddings from CLIP [EOS]
                     Shape: [B, d_text]
            k: Number of tokens to select (token budget)
            tau: Gumbel-Sigmoid temperature parameter
            training: Whether in training mode (affects STE behavior)

        Returns:
            idx: Top-k gather indices, Shape: [B, k]
            s_N: Raw importance scores, Shape: [B, N]
                 (for loss computation — gradients flow through STE)
        """
        B = v_coarse.size(0)

        # Step 1: Linear projections (Eq. 3-4)
        Q = self.q_proj(q_embed).unsqueeze(1)  # (B, 1, d_model)
        K = self.k_proj(v_coarse)               # (B, M, d_model)
        V = self.v_proj(v_coarse)               # (B, M, d_model)

        # Step 2: Multi-head reshape
        Q = Q.view(B, 1, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, 1, d_k)
        K = K.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, M, d_k)
        V = V.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, M, d_k)

        # Step 3: Scaled dot-product attention (Eq. 5)
        attn_logits = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, 1, M)
        alpha = attn_logits.softmax(dim=-1)                            # (B, H, 1, M)

        # Step 4: Context computation (Eq. 6)
        ctx = alpha @ V                                    # (B, H, 1, d_k)
        ctx = ctx.transpose(1, 2).reshape(B, 1, -1)       # (B, 1, d_model)
        ctx = self.o_proj(ctx).squeeze(1)                  # (B, d_model)
        ctx = self.norm1(ctx)

        # Step 5: Head-averaged attention weights (Eq. 7)
        alpha_avg = alpha.mean(dim=1).squeeze(-2)          # (B, M)

        # Step 6: Score MLP (Eq. 8-9)
        # Concat context vector with spatial attention weights
        combined = torch.cat([ctx, alpha_avg], dim=-1)     # (B, d_model + M)
        s_M = self.score_mlp(self.norm2(combined))         # (B, M)

        # Step 7: Bilinear upsample 7×7 → 24×24 (Eq. 10)
        s_spatial = s_M.view(B, 1, self.sqrt_M, self.sqrt_M)
        s_upsampled = F.interpolate(
            s_spatial,
            size=(self.sqrt_N, self.sqrt_N),
            mode='bilinear',
            align_corners=False,
        )
        s_N = s_upsampled.view(B, -1)                     # (B, N)

        # Step 8: Gumbel-Sigmoid STE for differentiable top-k
        from .masking import gumbel_sigmoid_ste

        if training:
            mask_ste, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=True)
            # Verify gradient flow (critical diagnostic)
            assert mask_ste.grad_fn is not None, (
                "Mask has no gradient! STE is broken. "
                "Check gumbel_sigmoid_ste implementation."
            )
            # Return mask_ste so tc_vitok_model can route VQA gradient back to TPN.
            # Without this, gather_tokens(v_full, idx) is NOT differentiable w.r.t. idx,
            # so TPN would only receive gradient from sparsity loss (→ scores ≈ 0 forever).
            return idx, s_N, mask_ste
        else:
            _, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=False)
            return idx, s_N, None

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
