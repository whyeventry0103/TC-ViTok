"""Tokenization Policy Network (TPN) for TC-ViTok.

This module implements the core TPN that learns a query-conditioned token selection
policy π: (Q, I_coarse) → S. The network uses cross-attention to fuse coarse visual
features with text embeddings, then produces per-token importance scores that are
used with Gumbel-Sigmoid STE to select the top-k most relevant tokens.

Architecture (Section 5.3):
1. Cross-attention: query features attend to coarse visual features
2. MLP score head: produces scalar importance score per spatial location
3. Bilinear upsampling: 7×7 → 24×24 to match full encoder resolution
4. Gumbel-Sigmoid STE: differentiable top-k selection
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

    Args:
        d_model: Dimension of coarse visual features (384 for DINOv2-S)
        d_text: Dimension of text embeddings (512 for CLIP ViT-B/32)
        d_h: Hidden dimension for cross-attention layers
        n_heads: Number of attention heads
        dropout: Dropout probability

    Total parameters: ~3M (lightweight compared to 7B LLaVA backbone)
    """

    def __init__(
        self,
        d_model: int = 384,
        d_text: int = 512,
        d_h: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_text = d_text
        self.d_h = d_h
        self.n_heads = n_heads

        # TODO (Phase C): Implement cross-attention block — see Section 5.3
        # Query: text embedding [B, d_text]
        # Key/Value: coarse visual features [B, 49, d_model]
        # Output: fused representation [B, 49, d_h]

        # Text embedding projection
        self.text_proj = nn.Linear(d_text, d_h)

        # Cross-attention: text queries attend to visual keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Visual feature projection for cross-attention
        self.visual_proj = nn.Linear(d_model, d_h)

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_h)
        self.ln2 = nn.LayerNorm(d_h)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h * 4, d_h),
            nn.Dropout(dropout),
        )

        # TODO (Phase C): Implement score head — see Eq. 5-6
        # Maps fused features to scalar importance scores
        # Input: [B, 49, d_h] → Output: [B, 49, 1]
        self.score_head = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
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

        Args:
            v_coarse: Coarse visual features from DINOv2-S
                      Shape: [batch_size, 49, d_model]
            q_embed: Text embeddings from CLIP
                     Shape: [batch_size, d_text]
            k: Number of tokens to select (token budget)
            tau: Gumbel-Sigmoid temperature parameter
            training: Whether in training mode (affects STE behavior)

        Returns:
            mask: Binary selection mask, Shape: [batch_size, 576]
                  1 = selected, 0 = masked
            scores: Raw importance scores before selection
                    Shape: [batch_size, 576]

        Notes:
            - Coarse features are 7×7 spatial grid (49 tokens)
            - Output mask is 24×24 spatial grid (576 tokens)
            - Bilinear upsampling bridges the resolution gap

        Paper references:
            - Cross-attention fusion: Section 5.3
            - Score computation: Eq. 5-6
            - Gumbel-Sigmoid STE: Eq. 7-12, Section 6
        """
        batch_size = v_coarse.size(0)

        # TODO (Phase C): Implement forward pass

        # Step 1: Project text embedding
        # q_proj: [B, d_h]
        q_proj = self.text_proj(q_embed)

        # Step 2: Project visual features
        # v_proj: [B, 49, d_h]
        v_proj = self.visual_proj(v_coarse)

        # Step 3: Cross-attention (Eq. 3-4)
        # Query: text features (broadcast to match attention interface)
        # Key/Value: visual features
        q_expanded = q_proj.unsqueeze(1)  # [B, 1, d_h]

        # Cross-attention: text attends to visual
        attn_out, attn_weights = self.cross_attn(
            query=q_expanded,
            key=v_proj,
            value=v_proj,
        )

        # Residual connection + layer norm
        attn_out = self.ln1(attn_out + q_expanded)

        # Step 4: Feed-forward network
        ffn_out = self.ffn(attn_out)
        fused = self.ln2(ffn_out + attn_out)  # [B, 1, d_h]

        # Broadcast fused representation to all spatial locations
        # This allows question-conditioned scoring of each visual location
        fused_broadcast = fused.expand(-1, 49, -1)  # [B, 49, d_h]
        combined = fused_broadcast + v_proj  # Combine with visual features

        # Step 5: Compute importance scores (Eq. 5-6)
        # scores_coarse: [B, 49, 1]
        scores_coarse = self.score_head(combined)
        scores_coarse = scores_coarse.squeeze(-1)  # [B, 49]

        # Step 6: Reshape to spatial grid for upsampling
        # 49 = 7×7 spatial locations
        scores_spatial = rearrange(scores_coarse, 'b (h w) -> b 1 h w', h=7, w=7)

        # Step 7: Bilinear upsample to full resolution (Eq. 6)
        # 7×7 → 24×24 (576 tokens)
        scores_upsampled = F.interpolate(
            scores_spatial,
            size=(24, 24),
            mode='bilinear',
            align_corners=False,
        )
        scores_flat = rearrange(scores_upsampled, 'b 1 h w -> b (h w)')  # [B, 576]

        # Step 8: Apply Gumbel-Sigmoid STE for differentiable top-k selection
        # This is delegated to masking.py — see gumbel_sigmoid_ste()
        from .masking import gumbel_sigmoid_ste

        mask = gumbel_sigmoid_ste(scores_flat, k=k, tau=tau, training=training)

        # CRITICAL DEBUGGING CHECK (Section 9.2):
        # Verify gradient flow through mask
        if training:
            assert mask.grad_fn is not None, (
                "Mask has no gradient! STE is broken. "
                "Check gumbel_sigmoid_ste implementation."
            )

        return mask, scores_flat

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
