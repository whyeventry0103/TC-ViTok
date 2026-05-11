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


class BudgetPredictor(nn.Module):
    """Predicts per-sample token budget fraction for adaptive selection.

    Takes the TPN's cross-attention context vector (which encodes how the
    query relates to the image) and predicts what fraction of tokens should
    be selected. The output is a continuous fraction in [k_min/N, k_max/N],
    which is fully differentiable — gradient flows from VQA loss through
    the selected tokens back to this predictor, and the budget loss
    λ * mean(k/N) provides direct gradient pressure to reduce k.

    Architecture: 2-layer MLP with score-entropy side input.
    The entropy of TPN scores serves as a query-free complexity proxy:
    high entropy → uniform scores → complex image → need more tokens.
    This gives BudgetPredictor a meaningful input signal beyond the
    cross-attention context, which otherwise only provides gradient
    from L_budget (pushing k toward k_min).

    Args:
        d_model: Input dimension from TPN context (384)
        k_min: Minimum tokens to select (36 = 6.25%)
        k_max: Maximum tokens to select (288 = 50%)
        N: Total number of visual tokens (576)
    """

    def __init__(
        self,
        d_model: int = 384,
        k_min: int = 36,
        k_max: int = 288,
        N: int = 576,
    ):
        super().__init__()
        assert k_min < k_max, (
            f"k_min ({k_min}) must be strictly less than k_max ({k_max}). "
            f"Equal values make BudgetPredictor a dead weight (zero output range)."
        )
        self.k_min = k_min
        self.k_max = k_max
        self.N = N

        # Score entropy projection: scalar entropy → 64-dim, added to hidden layer
        self.entropy_proj = nn.Linear(1, 64)

        self.head_pre = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
        )
        self.head_post = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        """Initialize to predict mid-range budget (~144 tokens) at step 0."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor, s_N: torch.Tensor | None = None) -> torch.Tensor:
        """Predict per-sample token fraction.

        Args:
            context: TPN cross-attention context vector [B, d_model]
            s_N: Instance-normalized TPN scores [B, N] (stop-gradient).
                 Used to compute score entropy as complexity proxy.

        Returns:
            k_frac: Target fraction of tokens to keep [B], in [k_min/N, k_max/N]
        """
        h = self.head_pre(context)            # [B, 64]

        if s_N is not None:
            # Score entropy: convert scores to probability distribution, compute H.
            # High entropy → uniform scores → complex/cluttered → need more tokens.
            # Stop-gradient: entropy is an INPUT signal, not a training target.
            with torch.no_grad():
                p = torch.softmax(s_N, dim=-1)        # [B, N]
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)  # [B, 1]
                # Normalize to ~[0, 1] range (max entropy = log(N) ≈ 6.36)
                ent = ent / 6.36
            # Project and add (stop-grad on entropy, but entropy_proj IS trained)
            h = h + self.entropy_proj(ent)    # [B, 64]

        raw = self.head_post(h)               # [B, 1]
        fraction = torch.sigmoid(raw)         # [B, 1] in (0, 1)
        # Map (0,1) → [k_min/N, k_max/N]
        k_frac = self.k_min / self.N + fraction * (self.k_max - self.k_min) / self.N
        return k_frac.squeeze(-1)             # [B]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        adaptive_budget: bool = False,
        budget_predictor: 'BudgetPredictor | None' = None,
        k_min: int = 36,
        k_max: int = 288,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Tokenization Policy Network.

        Implements the paper's architecture exactly:
        1. Project q_embed and v_coarse into shared space
        2. Multi-head cross-attention (text queries visual)
        3. Average attention weights across heads → α_avg
        4. Concat [context; α_avg] → Score MLP → coarse scores
        5. Bilinear upsample 7×7 → 24×24
        6. Gumbel-Sigmoid STE for differentiable top-k

        When adaptive_budget=True, uses BudgetPredictor to determine per-sample
        variable k instead of fixed k. The budget predictor sees the same
        cross-attention context used for scoring, and predicts a threshold that
        controls how many tokens pass selection.

        Args:
            v_coarse: Coarse visual features from DINOv2-S
                      Shape: [B, 49, d_coarse]
            q_embed: Text embeddings from CLIP [EOS]
                     Shape: [B, d_text]
            k: Number of tokens to select (token budget, used when adaptive_budget=False)
            tau: Gumbel-Sigmoid temperature parameter
            training: Whether in training mode (affects STE behavior)
            adaptive_budget: If True, use BudgetPredictor for variable k
            budget_predictor: BudgetPredictor module (required when adaptive_budget=True)
            k_min: Minimum tokens for adaptive mode (default 36)
            k_max: Maximum tokens for adaptive mode (default 288)

        Returns:
            idx: Top-k gather indices, Shape: [B, k] (fixed) or [B, k_max_batch] (adaptive)
            s_N: Raw importance scores, Shape: [B, N]
                 (for loss computation — gradients flow through STE)
            mask_ste: STE mask [B, N] (training) or None (eval)
            k_effective: Differentiable expected token count [B] (adaptive) or None (fixed)
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

        # Per-sample instance normalization: zero-mean, unit-std across the N token scores.
        #
        # Why: Without bounding, Adam pushes scores to O(100) in ~1000 steps, overwhelming
        # Gumbel noise (O(τ=1.0)) → same 144 tokens always selected (static crop).
        # tanh was tried but saturates at ±1 within ~1000 steps → tanh'(x)≈0.001 at
        # saturation → gradient vanishes through tanh, same problem as before.
        #
        # Instance norm properties:
        #   - Scores always at zero-mean, unit-std per sample → Gumbel noise O(1) always
        #     comparable to score differences throughout training (no saturation possible)
        #   - Gradient flows: d(s_norm)/d(s_raw) = (1/σ)(I - 1/N - outer(s_norm, s_norm)/N)
        #     non-vanishing as long as scores have any variance (degenerate only if all equal)
        #   - Only relative ordering matters for topk selection — absolute scale is irrelevant
        #   - No learnable parameters: cannot be tuned away by the optimizer
        s_N_mean = s_N.mean(dim=-1, keepdim=True)
        s_N_std = s_N.std(dim=-1, keepdim=True).clamp(min=1e-8)
        s_N = (s_N - s_N_mean) / s_N_std

        # Step 8: Token selection — fixed-k or adaptive budget
        if adaptive_budget and budget_predictor is not None:
            from .masking import adaptive_gumbel_sigmoid_ste

            # BudgetPredictor uses cross-attention context + score entropy
            k_frac = budget_predictor(ctx, s_N=s_N)  # [B] — fraction in [k_min/N, k_max/N]

            if training:
                mask_ste, idx, k_effective = adaptive_gumbel_sigmoid_ste(
                    s_N, k_frac=k_frac, tau=tau, training=True,
                    k_min=k_min, k_max=k_max, N=self.N,
                )
                assert mask_ste.grad_fn is not None, (
                    "Adaptive mask has no gradient! STE is broken."
                )
                return idx, s_N, mask_ste, k_effective
            else:
                _, idx, k_effective = adaptive_gumbel_sigmoid_ste(
                    s_N, k_frac=k_frac, tau=tau, training=False,
                    k_min=k_min, k_max=k_max, N=self.N,
                )
                return idx, s_N, None, k_effective
        else:
            # Original fixed-k path
            from .masking import gumbel_sigmoid_ste

            if training:
                mask_ste, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=True)
                assert mask_ste.grad_fn is not None, (
                    "Mask has no gradient! STE is broken. "
                    "Check gumbel_sigmoid_ste implementation."
                )
                return idx, s_N, mask_ste, None
            else:
                _, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=False)
                return idx, s_N, None, None

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TokenScorerV3(nn.Module):
    """v3 image-only token scorer operating directly on CLIP ViT-L/14 features.

    Differences from v1 TokenizationPolicyNetwork:
    - No DINOv2 coarse encoder (saves memory + forward time)
    - No CLIP text encoder (E3 empirically: query conditioning hurts; consistent
      with concurrent FasterVLM/VisPruner/VisionZip — all text-agnostic)
    - No 7×7 → 24×24 bilinear upsampling (eliminates resolution smearing)
    - Operates at native 24×24 = 576 resolution per token

    Architecture:
        proj_in  : Linear 1024 → d_h
        attn     : 1× pre-LN MultiheadAttention (spatial reasoning across tokens)
        ffn      : pre-LN 2-layer MLP (d_h → 2·d_h → d_h)
        score    : Linear d_h → 1
        + per-sample instance norm on scores (same as v1 — prevents Adam runaway)

    Parameters at d_h=256: ~790K. Smaller and faster than v1 TPN (887K).
    """

    def __init__(
        self,
        d_visual: int = 1024,
        d_h: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        N: int = 576,
    ):
        super().__init__()
        self.d_visual = d_visual
        self.d_h = d_h
        self.N = N

        self.proj_in = nn.Linear(d_visual, d_h)

        # Pre-LN self-attention block
        self.norm_attn = nn.LayerNorm(d_h)
        self.attn = nn.MultiheadAttention(
            d_h, n_heads, dropout=dropout, batch_first=True,
        )

        # Pre-LN FFN block
        self.norm_ffn = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_h * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h * 2, d_h),
        )

        # Score head
        self.norm_out = nn.LayerNorm(d_h)
        self.score_head = nn.Linear(d_h, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        v_full: torch.Tensor,
        k: int,
        tau: float = 1.0,
        training: bool = True,
        adaptive_budget: bool = False,
        budget_predictor: 'BudgetPredictor | None' = None,
        k_min: int = 36,
        k_max: int = 288,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score CLIP-L visual tokens and select top-k via Gumbel-Sigmoid STE.

        Args:
            v_full: CLIP-L features [B, N=576, 1024]
            k: Token budget (fixed-k mode)
            tau: Gumbel-Sigmoid temperature
            training: Training mode flag
            adaptive_budget: If True, use BudgetPredictor for per-sample k
            budget_predictor: BudgetPredictor module (required when adaptive)
            k_min, k_max: Adaptive budget bounds

        Returns:
            idx          : [B, k] (fixed) or [B, k_max_batch] (adaptive, -1 padding)
            s_N          : [B, N] instance-normed scores (for losses)
            mask_ste     : [B, N] STE mask (training) or None (eval)
            k_effective  : [B] expected token count (adaptive) or None (fixed)
        """
        B, N, _ = v_full.shape

        # Project per-token to internal dim
        h = self.proj_in(v_full)  # [B, N, d_h]

        # Pre-LN self-attention with residual
        h_n = self.norm_attn(h)
        attn_out, _ = self.attn(h_n, h_n, h_n, need_weights=False)
        h = h + attn_out

        # Pre-LN FFN with residual
        h = h + self.ffn(self.norm_ffn(h))

        # Score head
        h_out = self.norm_out(h)
        scores = self.score_head(h_out).squeeze(-1)  # [B, N]

        # Per-sample instance normalization (same rationale as v1 TPN)
        s_mean = scores.mean(dim=-1, keepdim=True)
        s_std = scores.std(dim=-1, keepdim=True).clamp(min=1e-8)
        s_N = (scores - s_mean) / s_std

        # Global context vector for BudgetPredictor — mean of post-self-attn
        # features. Carries scene-level info (complexity, layout) that lets the
        # BudgetPredictor decide per-sample k.
        ctx = h.mean(dim=1)  # [B, d_h]

        # Token selection
        if adaptive_budget and budget_predictor is not None:
            from .masking import adaptive_gumbel_sigmoid_ste

            k_frac = budget_predictor(ctx, s_N=s_N)

            if training:
                mask_ste, idx, k_eff = adaptive_gumbel_sigmoid_ste(
                    s_N, k_frac=k_frac, tau=tau, training=True,
                    k_min=k_min, k_max=k_max, N=N,
                )
                assert mask_ste.grad_fn is not None, (
                    "v3 adaptive mask has no gradient — STE is broken."
                )
                return idx, s_N, mask_ste, k_eff
            else:
                _, idx, k_eff = adaptive_gumbel_sigmoid_ste(
                    s_N, k_frac=k_frac, tau=tau, training=False,
                    k_min=k_min, k_max=k_max, N=N,
                )
                return idx, s_N, None, k_eff
        else:
            from .masking import gumbel_sigmoid_ste

            if training:
                mask_ste, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=True)
                assert mask_ste.grad_fn is not None, (
                    "v3 mask has no gradient — STE is broken."
                )
                return idx, s_N, mask_ste, None
            else:
                _, idx = gumbel_sigmoid_ste(s_N, k=k, tau=tau, training=False)
                return idx, s_N, None, None

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
