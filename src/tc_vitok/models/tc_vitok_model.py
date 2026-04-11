"""Full TC-ViTok model integrating TPN with VLM backbone.

This module ties together all components into the complete TC-ViTok system:
1. Coarse visual encoding (DINOv2-S)
2. Text encoding (CLIP)
3. Token selection policy (TPN)
4. Full visual encoding (CLIP ViT-L/14)
5. Gather-based token extraction (real compression)
6. VLM backbone (LLaVA-1.5 with LoRA, or stub for development)

The model is trained in 3 phases:
- Phase 0: TPN warmup with frozen LLaVA
- Phase 1: Joint finetuning of TPN + LoRA + projector
- Phase 2: Domain transfer to TextVQA (optional)

Paper reference: Section 5 (Full Architecture)
"""

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpn import TokenizationPolicyNetwork
from .encoders import CoarseVisualEncoder, TextEncoder, FullVisualEncoder


def gather_tokens(
    V: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Extract k visual tokens from V using top-k indices (Option B: gather).

    This is the core gather operation that provides real sequence compression.
    Output has shape [B, k, d_v] — the LLM sees k tokens, not N.

    Args:
        V: Full visual features from ViT-L, Shape: [B, N, d_v]
        idx: Top-k token indices from TPN, Shape: [B, k]

    Returns:
        V_selected: Extracted tokens, Shape: [B, k, d_v]
    """
    d_v = V.shape[-1]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, d_v)  # [B, k, d_v]
    return torch.gather(V, dim=1, index=idx_exp)       # [B, k, d_v]


def complement_gather(
    V: torch.Tensor,
    idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract unselected tokens (complement of idx) from V.

    Args:
        V: Full visual features, Shape: [B, N, D]
        idx: Top-k selected indices, Shape: [B, k]

    Returns:
        v_unsel: Unselected token features, Shape: [B, N-k, D]
        unsel_idx: Flat indices of unselected positions, Shape: [B, N-k]
    """
    B, N, D = V.shape
    k = idx.shape[1]

    # Build boolean mask: True = selected
    sel_mask = torch.zeros(B, N, device=V.device, dtype=torch.bool)
    sel_mask.scatter_(1, idx, True)

    # Unselected positions — exactly N-k per row (constant)
    all_pos = torch.arange(N, device=V.device).unsqueeze(0).expand(B, N)
    unsel_idx = all_pos[~sel_mask].reshape(B, N - k)          # [B, N-k]

    idx_exp = unsel_idx.unsqueeze(-1).expand(-1, -1, D)
    v_unsel = torch.gather(V, dim=1, index=idx_exp)           # [B, N-k, D]
    return v_unsel, unsel_idx


def sinusoidal_2d_pe(
    flat_indices: torch.Tensor,
    D: int,
    grid: int = 24,
) -> torch.Tensor:
    """2D sinusoidal positional encoding for a set of flat grid positions.

    Re-injects spatial structure into orthogonally-deflated unselected tokens
    so the UnselectedPooler cross-attention can attend based on WHERE each
    background token is located in the image.

    Args:
        flat_indices: [B, L] — flat indices in [0, grid²)
        D: Feature dimension of the visual tokens (must be divisible by 4)
        grid: Grid side length (24 for 24×24 = 576 tokens)

    Returns:
        pe: [B, L, D] positional encoding, same dtype as float32
    """
    row = (flat_indices // grid).float()   # [B, L]
    col = (flat_indices % grid).float()    # [B, L]
    half = D // 4
    div = torch.exp(
        torch.arange(half, device=flat_indices.device).float()
        * (-math.log(10000.0) / half)
    )                                      # [half]
    B, L = flat_indices.shape
    pe = torch.zeros(B, L, D, device=flat_indices.device, dtype=torch.float32)
    pe[..., 0:half]        = torch.sin(row.unsqueeze(-1) * div)
    pe[..., half:2*half]   = torch.cos(row.unsqueeze(-1) * div)
    pe[..., 2*half:3*half] = torch.sin(col.unsqueeze(-1) * div)
    pe[..., 3*half:]       = torch.cos(col.unsqueeze(-1) * div)
    return pe


class UnselectedPooler(nn.Module):
    """4-slot cross-attention pooler for orthogonally-deflated unselected tokens.

    Compresses the N-k unselected background tokens into 4 compact summary
    vectors. Each slot independently learns to aggregate a different aspect
    (global layout, background texture, peripheral objects, scene context).
    2D sinusoidal PE is injected into the keys/values so the slots know
    WHERE each background token is in the image.

    New trainable params: ~512K (4 × d_model for slots, MHA weights + norms)

    Args:
        d_model: Visual feature dimension (1024 for CLIP ViT-L/14)
        n_slots: Number of summary tokens (4 per innovations.md spec)
        n_heads: Attention heads for cross-attention (4)
    """

    def __init__(self, d_model: int = 1024, n_slots: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.slots = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        v_unsel_orth: torch.Tensor,
        unsel_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Pool orthogonal background tokens into n_slots summary vectors.

        Args:
            v_unsel_orth: Orthogonally-deflated unselected tokens [B, N-k, D]
            unsel_indices: Flat grid indices of unselected positions [B, N-k]

        Returns:
            summary: [B, n_slots, D]
        """
        B = v_unsel_orth.shape[0]
        D = v_unsel_orth.shape[-1]
        # Reinject 2D position info (lost during orthogonal deflation)
        pe = sinusoidal_2d_pe(unsel_indices, D).to(dtype=v_unsel_orth.dtype)
        kv = v_unsel_orth + pe                              # [B, N-k, D]
        q = self.slots.expand(B, -1, -1)                   # [B, n_slots, D]
        out, _ = self.cross_attn(q, kv, kv)
        return self.norm(out)                               # [B, n_slots, D]


class StubLLM(nn.Module):
    """Minimal LLM stub for development/testing without full LLaVA.

    Accepts visual tokens + text and produces dummy logits.
    This allows the full pipeline to run end-to-end for testing
    TPN training, mask quality, and gradient flow without requiring
    a 7B parameter LLaVA model.

    The stub uses a small transformer to process the visual tokens,
    producing logits over a small vocabulary. This is sufficient to:
    - Verify gradient flow through the STE
    - Test training loop mechanics
    - Debug TPN behavior

    Replace with real LLaVA integration for actual experiments.
    """

    def __init__(self, d_visual: int = 1024, d_model: int = 512, vocab_size: int = 1000):
        super().__init__()
        self.d_visual = d_visual
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Project visual tokens to model dim
        self.visual_proj = nn.Linear(d_visual, d_model)

        # Simple transformer layer to process tokens
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through stub LLM.

        Args:
            visual_tokens: Selected visual tokens, Shape: [B, k, d_visual]
            labels: Optional target token IDs, Shape: [B, seq_len]

        Returns:
            outputs: Dict with 'logits' and optionally 'loss'
        """
        # Project visual tokens
        x = self.visual_proj(visual_tokens)  # [B, k, d_model]

        # Process through transformer
        x = self.transformer(x)  # [B, k, d_model]

        # Produce logits
        logits = self.lm_head(x)  # [B, k, vocab_size]

        outputs = {"logits": logits}

        if labels is not None:
            # Compute loss against labels (truncated/padded to match k)
            B, k_len, V = logits.shape
            seq_len = labels.shape[1]
            # Align sequence lengths
            min_len = min(k_len, seq_len)
            loss = nn.functional.cross_entropy(
                logits[:, :min_len].reshape(-1, V),
                labels[:, :min_len].reshape(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss

        return outputs


class TCViTokModel(nn.Module):
    """Full TC-ViTok model with learned token selection policy.

    This model wraps either a real LLaVA backbone or a stub LLM, adding
    a lightweight TPN that dynamically selects k out of N visual tokens
    based on the input question. Selected tokens are extracted via
    torch.gather (real compression, sequence length = k not N).

    Components:
        - Coarse encoder: DINOv2-S (frozen, 22M params)
        - Text encoder: CLIP ViT-B/32 (frozen, 63M params)
        - TPN: Cross-attention + Score MLP (trainable, ~887K params)
        - Full encoder: CLIP ViT-L/14 (frozen, 304M params)
        - VLM: LLaVA-1.5 or StubLLM (partial/full training)

    Args:
        k: Token budget (number of tokens to select)
        tpn_config: Configuration dict for TPN
        use_stub_llm: If True, use StubLLM instead of LLaVA (for dev/testing)
        llava_model_name: HuggingFace model ID for LLaVA (if not using stub)
        lora_config: Configuration dict for LoRA (if not using stub)
    """

    def __init__(
        self,
        k: int = 144,
        tpn_config: Optional[Dict[str, Any]] = None,
        use_stub_llm: bool = True,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
        lora_config: Optional[Dict[str, Any]] = None,
        load_encoders: bool = True,
    ):
        super().__init__()

        self.k = k
        self.use_stub_llm = use_stub_llm

        # 1. Coarse visual encoder (DINOv2-S, frozen)
        # When load_encoders=False (e.g., testing with stub), use None and
        # rely on forward_cached or external feature pipelines.
        if load_encoders:
            self.coarse_encoder = CoarseVisualEncoder()
            self.text_encoder = TextEncoder()
            self.full_encoder = FullVisualEncoder()
        else:
            self.coarse_encoder = None
            self.text_encoder = None
            self.full_encoder = None

        # 3. Tokenization Policy Network (TPN, trainable)
        tpn_config = tpn_config or {}
        self.tpn = TokenizationPolicyNetwork(**tpn_config)

        # 4. Unselected token pooler (Phase 1.5: orthogonal soft recycling)
        # Compresses N-k background tokens into 4 summary vectors injected as
        # a prefix before the k selected tokens. Zero params in stub mode.
        if not use_stub_llm:
            self.unsel_pooler = UnselectedPooler(d_model=1024, n_slots=4, n_heads=4)
        else:
            self.unsel_pooler = None

        # 5. VLM backbone
        if use_stub_llm:
            self.llm = StubLLM(d_visual=1024)
        else:
            self._init_llava(llava_model_name, lora_config or {})

    def _init_llava(self, model_name: str, lora_config: dict):
        """Initialize LLaVA-1.5 with LoRA adapters and VQA classification head.

        TC-ViTok replaces LLaVA's internal ViT with its own CLIP ViT-L/14 + TPN
        pipeline. We therefore:
          1. Load LlavaForConditionalGeneration (keeps mm_projector + LM)
          2. Freeze LLaVA's vision_tower (never used — bypassed by our pipeline)
          3. Apply LoRA to the language model (q/k/v/o_proj)
          4. Add a VQA classification head: LLM hidden dim → answer vocabulary
             This maps pooled LM hidden states to the VQAv2 answer vocabulary
             (top-3129 answers by default, matching VQAv2Dataset)

        Handles two transformers architecture versions:
          OLD (transformers 4.36–4.44):
            llava.language_model        = LlamaForCausalLM  (has generate())
            llava.multi_modal_projector = LlavaMultiModalProjector
            llava.vision_tower          = CLIPVisionModel
          NEW (transformers 4.45+):
            llava.model.language_model        = LlamaModel  (no generate())
            llava.model.multi_modal_projector = LlavaMultiModalProjector
            llava.model.vision_tower          = CLIPVisionModel
        """
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        from peft import get_peft_model, LoraConfig, TaskType

        # low_cpu_mem_usage=True loads weights lazily on CPU (meta tensors → real).
        # Do NOT use device_map="cpu" — it installs accelerate dispatch hooks that
        # intercept .to() calls, preventing model.to(cuda:N) from actually moving
        # tensors to GPU. This caused GPU 1 to have only 727MiB while GPU 0 had 16GB.
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # match training BF16 autocast (float16 caused NaN via attention overflow)
            low_cpu_mem_usage=True,
        )
        self.processor = LlavaProcessor.from_pretrained(model_name)

        # Detect architecture and locate language model + mm_projector
        if hasattr(self.llava, "language_model"):
            # OLD architecture: language_model is LlamaForCausalLM (has generate() + lm_head)
            lm_target = self.llava.language_model
            self._mm_proj = self.llava.multi_modal_projector
            vision_tower = getattr(self.llava, "vision_tower", None)
            lora_task = TaskType.CAUSAL_LM
            # OLD arch: LlamaForCausalLM.forward() returns logits directly
            self._use_llava_lm_head = False
        elif hasattr(self.llava, "model") and hasattr(self.llava.model, "language_model"):
            # NEW architecture: language_model is LlamaModel (no generate(); use FEATURE_EXTRACTION)
            lm_target = self.llava.model.language_model
            self._mm_proj = self.llava.model.multi_modal_projector
            vision_tower = getattr(self.llava.model, "vision_tower", None)
            lora_task = TaskType.FEATURE_EXTRACTION
            # NEW arch: LlamaModel returns last_hidden_state; use llava.lm_head for logits
            self._use_llava_lm_head = True
        else:
            top = {k: type(v).__name__ for k, v in vars(self.llava).items() if not k.startswith("_")}
            raise AttributeError(
                f"Cannot locate language_model in {type(self.llava).__name__}. "
                f"Top-level attrs: {top}"
            )

        # Freeze LLaVA's internal ViT — TC-ViTok uses its own CLIP ViT-L/14
        if vision_tower is not None:
            for param in vision_tower.parameters():
                param.requires_grad = False

        # Apply LoRA (r=32, alpha=64) to language model q/k/v/o_proj
        peft_config = LoraConfig(
            task_type=lora_task,
            inference_mode=False,
            r=lora_config.get("rank", 32),
            lora_alpha=lora_config.get("alpha", 64),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get(
                "targets",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self._llm = get_peft_model(lm_target, peft_config)

        # Patch the LoRA-wrapped LM back into its original location so that
        # self.llava.parameters() / state_dict() still covers LoRA weights.
        if hasattr(self.llava, "language_model"):
            self.llava.language_model = self._llm
        else:
            self.llava.model.language_model = self._llm

    def forward(
        self,
        images_coarse: Optional[torch.Tensor] = None,
        images_full: Optional[torch.Tensor] = None,
        questions: Optional[list] = None,
        answer_strings: Optional[list] = None,
        tau: float = 1.0,
        return_idx: bool = False,
        v_coarse: Optional[torch.Tensor] = None,
        q_embed: Optional[torch.Tensor] = None,
        v_full: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward pass — dispatches to cached or full pipeline.

        DDP only hooks into forward(), so ALL training calls must go through
        this method (never call forward_cached() directly during training).
        When v_coarse and q_embed are provided, the frozen DINOv2-S and CLIP
        text encoders are bypassed (cached path). Otherwise the full pipeline
        runs from raw images and question strings.

        Args:
            images_coarse: Low-res images for TPN, Shape: [B, 3, 98, 98]
            images_full: High-res images for CLIP ViT-L, Shape: [B, 3, 336, 336]
            questions: List of question strings, length B
            answer_strings: List of raw answer strings for next-token prediction, length B
            tau: Gumbel-Sigmoid temperature
            return_idx: If True, include idx and scores in output
            v_coarse: Pre-computed coarse features [B, 49, 384] (cached path)
            q_embed: Pre-computed text embeddings [B, 512] (cached path)
            v_full: Pre-computed full visual features [B, 576, 1024] (optional)

        Returns:
            outputs: Dictionary containing:
                - logits: Language model logits [B, seq_len, vocab_size]
                - seq_labels: Token-level labels with -100 masking [B, seq_len]
                - idx: Top-k gather indices [B, k] (if return_idx)
                - scores: TPN raw scores [B, N]
                - selected_tokens: Gathered visual tokens [B, k, D]
        """
        # Dispatch: cached path (pre-computed features) vs full pipeline
        use_cached = v_coarse is not None and q_embed is not None

        if not use_cached:
            # Full pipeline: run frozen encoders
            v_coarse = self.coarse_encoder(images_coarse)  # [B, 49, 384]
            q_embed = self.text_encoder(questions)  # [B, 512]

        # Step 1: TPN produces top-k gather indices
        # During training, TPN also returns mask_ste for STE gradient routing.
        idx, s_N, mask_ste = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=tau,
            training=self.training,
        )  # idx: [B, k], s_N: [B, N], mask_ste: [B, N] or None

        # Step 2: Extract full visual features (frozen CLIP ViT-L/14)
        if v_full is None:
            if self.full_encoder is None:
                raise RuntimeError(
                    "forward() requires full_encoder but load_encoders=False. "
                    "Either initialize with load_encoders=True or pre-compute v_full."
                )
            v_full = self.full_encoder(images_full)  # [B, 576, 1024]

        # Step 3: GATHER selected tokens (real compression!)
        # Output: [B, k, 1024] — LLM sees k tokens, not N
        v_selected = gather_tokens(v_full, idx)

        # Step 3b: STE gradient routing — connect VQA loss to TPN parameters.
        # torch.gather is NOT differentiable w.r.t. idx, so VQA loss gradient
        # cannot reach TPN through the gather operation alone.
        #
        # PREVIOUS APPROACH (broken): multiply by mask_ste = m_hard + (m_soft - m_soft.detach())
        #   gradient ∝ d(m_soft)/d(score) = m_soft*(1-m_soft)/tau
        #   → as tau anneals 1.0→0.1, m_soft saturates to 0/1, gradient → 0 (VANISHES!)
        #   This killed TPN learning by step ~15k.
        #
        # CURRENT APPROACH (tau-independent additive STE):
        #   correction = s_N_at_selected - s_N_at_selected.detach()  [= 0 in forward]
        #   v_selected_out = v_selected * (1 + correction)           [= v_selected in forward]
        #   gradient: d(loss)/d(s_N_i) = Σ_d g_v[b,j,d] * v_selected[b,j,d]  [dot product]
        #   This is tau-independent and remains active throughout training.
        if self.training and not self.use_stub_llm:
            scores_at_selected = torch.gather(s_N, dim=1, index=idx)      # [B, k]
            correction = scores_at_selected - scores_at_selected.detach() # [B, k], = 0 fwd
            v_selected = v_selected * (1.0 + correction.unsqueeze(-1))    # [B, k, D], identity fwd

        # Step 3c: Orthogonal soft recycling (Phase 1.5)
        # Compress N-k unselected background tokens into 4 summary vectors and
        # prepend them to the LLM prefix. This recovers information hard-dropped
        # in Phase 1, expected to close the 3.5% gap to 76% VQAv2.
        #
        # Orthogonal deflation: project out mean(v_selected) from unselected tokens
        # before pooling, so summary slots carry COMPLEMENTARY information (not
        # redundant near-threshold tokens). See Efficient3D / innovations.md Concept 3.
        if not self.use_stub_llm and self.unsel_pooler is not None:
            v_unsel, unsel_idx = complement_gather(v_full, idx)   # [B, N-k, D], [B, N-k]

            # Orthogonal deflation: remove the selected-token subspace
            v_sel_mean = v_selected.mean(dim=1)                   # [B, D]
            v_sel_norm_sq = (
                (v_sel_mean * v_sel_mean).sum(-1, keepdim=True) + 1e-8
            )                                                      # [B, 1]
            dot = (v_unsel * v_sel_mean.unsqueeze(1)).sum(
                -1, keepdim=True
            )                                                      # [B, N-k, 1]
            proj_coeff = dot / v_sel_norm_sq.unsqueeze(1)         # [B, N-k, 1]
            v_unsel_orth = (
                v_unsel - proj_coeff * v_sel_mean.unsqueeze(1)
            )                                                      # [B, N-k, D]

            # Pool orthogonal background into 4 summary tokens
            v_summary = self.unsel_pooler(v_unsel_orth, unsel_idx)  # [B, 4, D]

            # LLM prefix: [4 summary | k selected] — total k+4 tokens
            v_input = torch.cat([v_summary, v_selected], dim=1)   # [B, k+4, D]
        else:
            v_input = v_selected                                   # [B, k, D]

        # Step 4: Forward through LLM backbone
        # _llava_forward uses B, seq_len, _ = v_input.shape so k here = k+4 (or k in stub mode)
        if self.use_stub_llm:
            llm_outputs = self.llm(visual_tokens=v_selected)
        else:
            llm_outputs = self._llava_forward(v_input, questions, answer_strings)

        # Build output dict
        outputs = {
            "logits": llm_outputs.get("logits"),
            "scores": s_N,
            "selected_tokens": v_selected,
        }

        if "seq_labels" in llm_outputs:
            outputs["seq_labels"] = llm_outputs["seq_labels"]

        if return_idx:
            outputs["idx"] = idx

        # Return mask_ste and q_embed for entropy + diversity loss gradient routing.
        # mask_ste (binary forward, m_soft backward) provides gradient to ALL N tokens,
        # not just the k selected ones — this is the escape route from static crop.
        # q_embed enables within-batch diversity loss to penalize query-agnostic selection.
        if self.training and mask_ste is not None:
            outputs["mask_ste"] = mask_ste
        if q_embed is not None:
            outputs["q_embed"] = q_embed.detach()  # frozen CLIP encoder, no gradient needed

        return outputs

    def forward_cached(
        self,
        v_coarse: torch.Tensor,
        q_embed: torch.Tensor,
        images_full: torch.Tensor,
        v_full: Optional[torch.Tensor] = None,
        questions: Optional[list] = None,
        answer_strings: Optional[list] = None,
        tau: float = 1.0,
        return_idx: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using pre-computed cached features.

        Delegates to forward() with cached kwargs. Kept for backward
        compatibility with evaluation scripts. During DDP training, always
        call forward() through the DDP wrapper instead of this method.

        Args:
            v_coarse: Cached coarse features, Shape: [B, 49, 384]
            q_embed: Cached text embeddings, Shape: [B, 512]
            images_full: High-res images for CLIP ViT-L, Shape: [B, 3, 336, 336]
            v_full: Pre-computed full visual features (optional)
            questions: List of question strings (needed for LLaVA text conditioning)
            answer_strings: List of raw answer strings for next-token prediction
            tau: Gumbel-Sigmoid temperature
            return_idx: If True, include idx and scores in output

        Returns:
            outputs: Same format as forward()
        """
        return self.forward(
            v_coarse=v_coarse,
            q_embed=q_embed,
            images_full=images_full,
            v_full=v_full,
            questions=questions,
            answer_strings=answer_strings,
            tau=tau,
            return_idx=return_idx,
        )

    def _llava_forward(
        self,
        v_selected: torch.Tensor,
        questions: Optional[list] = None,
        answer_strings: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through real LLaVA with next-token prediction training.

        Training pipeline (when answer_strings provided):
          1. Project k gathered tokens through LLaVA's mm_projector → [B, k, H]
          2. Per sample: tokenize "USER: q\\nASSISTANT:" + " answer</s>"
          3. Concatenate [visual_prefix | question_tokens | answer_tokens]
          4. Label mask: -100 for visual+question positions; answer token IDs for answer
          5. Pad all sequences to batch max length
          6. Forward through LoRA-adapted LM
          7. Get logits via lm_head (OLD arch: from output.logits; NEW: from llava.lm_head)

        Inference pipeline (when answer_strings is None):
          - Builds [visual_prefix | question_tokens] only (no labels)
          - Returns logits for greedy decoding (use _llava_generate for full generation)

        Args:
            v_selected: Selected visual tokens, Shape: [B, k, 1024]
            questions: List of question strings, length B (optional)
            answer_strings: List of raw answer strings for next-token prediction, length B

        Returns:
            outputs: Dict with:
                'logits': [B, seq_len, vocab_size] (float32)
                'seq_labels': [B, seq_len] with -100 masking (only when answer_strings given)
        """
        from contextlib import nullcontext

        B, k, _ = v_selected.shape
        device = v_selected.device
        lm = self._llm

        # Autocast: rely on caller's context (trainer uses BF16 autocast).
        # Using a nested FP16 autocast here would override the caller's BF16.
        _on_cuda = device.type == "cuda" if isinstance(device, torch.device) else "cuda" in str(device)
        _already_in_autocast = torch.is_autocast_enabled() if _on_cuda else False
        amp_ctx = (
            nullcontext() if _already_in_autocast
            else (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if _on_cuda else nullcontext())
        )

        # Use no_grad when LLM is frozen AND we don't need gradient through it.
        # During training: nullcontext — the tau-independent STE routes VQA gradient
        # to TPN via d(loss)/d(v_selected); this requires LLM activations to be
        # stored so backprop can flow from logits back to v_selected → correction → s_N.
        # During eval/inference: no_grad — no gradient needed, saves ~8GB activation memory.
        _lm_frozen = not any(p.requires_grad for p in lm.parameters())
        _lm_ctx = torch.no_grad() if (_lm_frozen and not self.training) else nullcontext()

        with amp_ctx:
            # Step 1: Project visual tokens through mm_projector (1024 → 4096)
            v_projected = self._mm_proj(v_selected)  # [B, k, H]
            embed_fn = lm.get_input_embeddings()
            tokenizer = self.processor.tokenizer

            if questions is not None and answer_strings is not None:
                # ── Training mode: build [visual | question | answer] per sample ──
                all_embeds = []
                all_masks = []
                all_labels = []

                for i in range(B):
                    # Question prompt (no answer yet)
                    q_text = f"USER: {questions[i]}\nASSISTANT:"
                    q_ids = tokenizer(
                        q_text,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=128,
                        add_special_tokens=True,
                    ).input_ids[0].to(device)           # [T_q]
                    q_embeds = embed_fn(q_ids)            # [T_q, H]

                    # Answer tokens (space prefix for proper tokenization)
                    a_text = f" {answer_strings[i]}"
                    a_ids = tokenizer(
                        a_text,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=32,
                        add_special_tokens=False,
                    ).input_ids[0].to(device)             # [T_a]
                    # Append EOS to mark end of answer
                    eos = torch.tensor([tokenizer.eos_token_id], device=device, dtype=torch.long)
                    a_ids = torch.cat([a_ids, eos])       # [T_a + 1]
                    a_embeds = embed_fn(a_ids)             # [T_a+1, H]

                    # Full sequence: [visual_prefix | question | answer]
                    seq_embeds = torch.cat([v_projected[i], q_embeds, a_embeds], dim=0)

                    # Labels: -100 (ignore) for visual + all-but-last-question positions.
                    # logits[i] predicts the token AFTER position i (causal LM), so:
                    #   logits[k+T_q-1] = f([visual, full_question]) → should predict a_ids[0]
                    # We shift answer labels one position LEFT vs the naive placement:
                    #   ignore length = k + T_q - 1  (NOT k + T_q)
                    #   trailing -100 pads to match seq_embeds length
                    ignore = torch.full(
                        (k + len(q_ids) - 1,), -100, device=device, dtype=torch.long
                    )
                    trailing = torch.tensor([-100], device=device, dtype=torch.long)
                    seq_labels = torch.cat([ignore, a_ids, trailing])  # same length as seq_embeds

                    attn_mask = torch.ones(len(seq_embeds), device=device, dtype=torch.long)

                    all_embeds.append(seq_embeds)
                    all_masks.append(attn_mask)
                    all_labels.append(seq_labels)

                # Pad to batch max length (right-padding with zeros / -100)
                max_len = max(e.shape[0] for e in all_embeds)
                H = all_embeds[0].shape[-1]

                padded_embeds = torch.zeros(
                    B, max_len, H, device=device, dtype=all_embeds[0].dtype
                )
                padded_masks  = torch.zeros(B, max_len, device=device, dtype=torch.long)
                padded_labels = torch.full(
                    (B, max_len), -100, device=device, dtype=torch.long
                )

                for i in range(B):
                    L = all_embeds[i].shape[0]
                    padded_embeds[i, :L] = all_embeds[i]
                    padded_masks[i, :L]  = all_masks[i]
                    padded_labels[i, :L] = all_labels[i]

                with _lm_ctx:
                    lm_outputs = lm(
                        inputs_embeds=padded_embeds,
                        attention_mask=padded_masks,
                        output_hidden_states=True,
                    )

                # Get logits (arch-dependent)
                if self._use_llava_lm_head:
                    logits = self.llava.lm_head(lm_outputs.last_hidden_state).float()
                else:
                    logits = lm_outputs.logits.float()

                return {"logits": logits, "seq_labels": padded_labels}

            else:
                # ── Inference mode: [visual | question] only ──
                if questions is not None:
                    q_texts = [f"USER: {q}\nASSISTANT:" for q in questions]
                    text_inputs = tokenizer(
                        q_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                        add_special_tokens=True,
                    )
                    input_ids = text_inputs.input_ids.to(device)
                    text_attn = text_inputs.attention_mask.to(device)
                    text_embeds = embed_fn(input_ids)

                    combined_embeds = torch.cat([v_projected, text_embeds], dim=1)
                    vis_attn = torch.ones(B, k, device=device, dtype=torch.long)
                    combined_mask = torch.cat([vis_attn, text_attn], dim=1)
                else:
                    combined_embeds = v_projected
                    combined_mask = torch.ones(B, k, device=device, dtype=torch.long)

                with _lm_ctx:
                    lm_outputs = lm(
                        inputs_embeds=combined_embeds,
                        attention_mask=combined_mask,
                        output_hidden_states=True,
                    )

                if self._use_llava_lm_head:
                    logits = self.llava.lm_head(lm_outputs.last_hidden_state).float()
                else:
                    logits = lm_outputs.logits.float()

                return {"logits": logits}

    def _llava_generate(
        self,
        v_selected: torch.Tensor,
        questions: list,
        max_new_tokens: int = 20,
    ) -> list:
        """Generative answer decoding using LLaVA backbone.

        Constructs the LLaVA-1.5 conversation template, concatenates visual
        prefix with question embeddings, and calls lm.generate. Used during
        evaluation to produce natural language answers for VQA scoring.

        Args:
            v_selected: Selected visual tokens, Shape: [B, k, 1024]
            questions: List of question strings, length B
            max_new_tokens: Maximum tokens to generate per sample

        Returns:
            answers: List of decoded answer strings, length B
        """
        from contextlib import nullcontext

        B, k, _ = v_selected.shape
        device = v_selected.device
        tokenizer = self.processor.tokenizer
        lm = self._llm

        # Autocast: model loaded as bfloat16 — use bfloat16 autocast to match.
        _on_cuda = device.type == "cuda" if isinstance(device, torch.device) else "cuda" in str(device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if _on_cuda
            else nullcontext()
        )

        with amp_ctx:
            # Project visual tokens
            v_projected = self._mm_proj(v_selected)  # [B, k, H]

            # Build LLaVA-1.5 conversation template (without <image> token —
            # we inject visual tokens directly as embeddings)
            prompts = [f"USER: {q}\nASSISTANT:" for q in questions]
            text_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True,
            )
            input_ids = text_inputs.input_ids.to(device)
            text_attn = text_inputs.attention_mask.to(device)

            # Embed prompt tokens
            embed_fn = lm.get_input_embeddings()
            text_embeds = embed_fn(input_ids)  # [B, T, H]

            # Concatenate: visual prefix + conversation prompt
            combined_embeds = torch.cat([v_projected, text_embeds], dim=1)  # [B, k+T, H]
            vis_attn = torch.ones(B, k, device=device, dtype=torch.long)
            combined_mask = torch.cat([vis_attn, text_attn], dim=1)        # [B, k+T]

            # Determine which object can truly call generate().
            # PEFT PeftModel inherits generate() from PreTrainedModel but at runtime
            # delegates to self.get_base_model().generate() — which fails when the
            # base model is LlamaModel (new arch, no lm_head, no generate()).
            # LlamaForCausalLM (old arch) has lm_head → can generate.
            # LlavaForConditionalGeneration always has generate() via GenerationMixin.
            base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
            gen_fn = lm if hasattr(base, "lm_head") else self.llava
            generated_ids = gen_fn.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and strip conversation template artifacts
        raw_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers = []
        for ans in raw_answers:
            # Strip "ASSISTANT:" prefix and any leading/trailing whitespace
            if "ASSISTANT:" in ans:
                ans = ans.split("ASSISTANT:")[-1]
            answers.append(ans.strip())

        return answers

    def _select_tokens(
        self,
        images_coarse: torch.Tensor,
        images_full: torch.Tensor,
        questions: list,
    ) -> torch.Tensor:
        """Select visual tokens via TPN. Override in subclasses for ablations.

        Returns:
            v_selected: Gathered visual tokens, Shape: [B, k, 1024]
        """
        v_coarse = self.coarse_encoder(images_coarse)
        q_embed = self.text_encoder(questions)
        idx, _, __ = self.tpn(
            v_coarse=v_coarse, q_embed=q_embed, k=self.k, tau=0.1, training=False,
        )
        v_full = self.full_encoder(images_full)
        return gather_tokens(v_full, idx)

    @torch.no_grad()
    def generate(
        self,
        images_coarse: torch.Tensor,
        images_full: torch.Tensor,
        questions: list[str],
        max_new_tokens: int = 20,
    ) -> list[str]:
        """Generative answer decoding. Use this for all VQA benchmark evaluation."""
        self.eval()
        v_selected = self._select_tokens(images_coarse, images_full, questions)
        if self.use_stub_llm:
            llm_outputs = self.llm(visual_tokens=v_selected)
            logits = llm_outputs["logits"]
            preds = logits[:, 0, :].argmax(dim=-1)
            return [str(p.item()) for p in preds]
        else:
            return self._llava_generate(v_selected, questions, max_new_tokens)

    def get_selection_info(
        self,
        images_coarse: torch.Tensor,
        questions: list[str],
    ) -> Dict[str, torch.Tensor]:
        """Get token selection indices and scores for visualization.

        Args:
            images_coarse: Low-res images, Shape: [B, 3, 98, 98]
            questions: List of question strings

        Returns:
            info: Dict with idx [B, k], scores [B, N], mask [B, N]
        """
        self.eval()
        with torch.no_grad():
            v_coarse = self.coarse_encoder(images_coarse)
            q_embed = self.text_encoder(questions)
            idx, s_N, _ = self.tpn(
                v_coarse=v_coarse,
                q_embed=q_embed,
                k=self.k,
                tau=0.1,
                training=False,
            )

        # Reconstruct binary mask from indices for visualization
        mask = torch.zeros(s_N.shape, device=s_N.device)
        mask.scatter_(-1, idx, 1.0)

        return {"idx": idx, "scores": s_N, "mask": mask}

    @torch.no_grad()
    def _get_teacher_importance(
        self,
        v_full: torch.Tensor,
        questions: list,
        answer_strings: list,
    ) -> torch.Tensor:
        """Run frozen base LLaVA with all 576 tokens to get per-token importance.

        Implements hidden-state cosine similarity distillation (innovations.md P1b).
        The base model (LoRA disabled) sees all N visual tokens + question + answer.
        Per-token importance = cosine_sim(mean(answer_hidden_states), token_hidden_state).

        Runs with torch.no_grad() — no gradient, ~19GB peak GPU memory.
        Sequential: teacher first, then free activations, then student forward.

        Args:
            v_full: Full CLIP visual features [B, N, D] (already computed)
            questions: List of question strings, length B
            answer_strings: List of answer strings, length B

        Returns:
            importance: Per-token importance scores [B, N], detached
        """
        from contextlib import nullcontext
        B, N, _ = v_full.shape
        device = v_full.device
        lm = self._llm
        tokenizer = self.processor.tokenizer

        _on_cuda = "cuda" in str(device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if _on_cuda else nullcontext()
        )

        with amp_ctx:
            # Project all 576 tokens through mm_projector
            v_proj_full = self._mm_proj(v_full)              # [B, N, H]
            embed_fn = lm.get_input_embeddings()

            # ── Bug fix #2: correct LLaVA-1.5 prompt structure ───────────────
            # LLaVA-1.5 was trained with:
            #   <s> USER: [image_tokens]\n{question}\nASSISTANT: {answer}</s>
            # Visual tokens are sandwiched between the "USER: " prefix and the
            # "\n{question}\nASSISTANT:" suffix — NOT prepended before <bos>.
            # Using the original structure gives the teacher (base LLaVA) the
            # most meaningful hidden-state patterns for the distillation target.
            #
            # Per-sample tracking:
            #   visual_starts[i]: position in sequence where visual tokens begin
            #   answer_starts[i]: position where answer tokens begin
            #   seq_lens[i]:      true sequence length (excludes zero-padding)
            all_embeds  = []
            all_masks   = []
            visual_starts  = []
            answer_starts  = []

            for i in range(B):
                # Prefix: "<s> USER: " — includes BOS via add_special_tokens=True
                prefix_ids = tokenizer(
                    "USER: ",
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,   # adds <s> (BOS)
                ).input_ids[0].to(device)      # [T_pre]
                prefix_embeds = embed_fn(prefix_ids)

                # Suffix: "\n{question}\nASSISTANT:" — no BOS (already in prefix)
                suffix_text = f"\n{questions[i]}\nASSISTANT:"
                suffix_ids = tokenizer(
                    suffix_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=128,
                    add_special_tokens=False,
                ).input_ids[0].to(device)      # [T_suf]
                suffix_embeds = embed_fn(suffix_ids)

                # Answer: " {answer}</s>"
                a_text = f" {answer_strings[i]}"
                a_ids = tokenizer(
                    a_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=32,
                    add_special_tokens=False,
                ).input_ids[0].to(device)
                eos = torch.tensor(
                    [tokenizer.eos_token_id], device=device, dtype=torch.long
                )
                a_ids = torch.cat([a_ids, eos])
                a_embeds = embed_fn(a_ids)

                # Full sequence: [<s> USER: | visual_tokens | \n{q}\nASSISTANT: | answer</s>]
                T_pre = len(prefix_ids)
                T_suf = len(suffix_ids)
                seq = torch.cat([prefix_embeds, v_proj_full[i], suffix_embeds, a_embeds], dim=0)

                visual_starts.append(T_pre)
                answer_starts.append(T_pre + N + T_suf)
                all_embeds.append(seq)
                all_masks.append(torch.ones(len(seq), device=device, dtype=torch.long))

            # ── Bug fix #1: record true lengths BEFORE padding ───────────────
            # Needed to exclude zero-padding hidden states from the h_ans mean.
            seq_lens = [e.shape[0] for e in all_embeds]  # [B]

            max_len = max(seq_lens)
            H = all_embeds[0].shape[-1]
            padded_embeds = torch.zeros(
                B, max_len, H, device=device, dtype=all_embeds[0].dtype
            )
            padded_masks = torch.zeros(B, max_len, device=device, dtype=torch.long)
            for i in range(B):
                L = seq_lens[i]
                padded_embeds[i, :L] = all_embeds[i]
                padded_masks[i, :L]  = all_masks[i]

            # Free per-sample lists and projected features — no longer needed
            del all_embeds, all_masks, v_proj_full

            # Run base LLM with LoRA disabled (teacher = base pretrained model)
            lm.disable_adapter_layers()
            try:
                lm_out = lm(
                    inputs_embeds=padded_embeds,
                    attention_mask=padded_masks,
                    output_hidden_states=True,
                    use_cache=False,
                )
            finally:
                lm.enable_adapter_layers()

            # ── Bug fix #3: free padded tensors immediately after LM forward ──
            # lm_out holds ~19GB of activations. padded_embeds / padded_masks
            # are no longer needed. Explicit del ensures CUDA frees them before
            # the student forward — without this, the student OOMs on the 2nd pass.
            del padded_embeds, padded_masks

            # Extract last-layer hidden states [B, max_len, D_lm]
            if self._use_llava_lm_head:
                h = lm_out.last_hidden_state          # NEW arch: LlamaModel
            else:
                h = lm_out.hidden_states[-1]          # OLD arch: LlamaForCausalLM
            del lm_out                                # free ~19GB activation memory

            # Per-sample importance: cosine_sim(mean(h_answer), h_visual).
            # h_visual: visual positions are at [visual_start : visual_start + N]
            #           (NOT 0:N — BOS + "USER: " prefix comes first).
            # h_answer: slice [ans_start : seq_lens[i]] to exclude padding tokens.
            importances = []
            for i in range(B):
                vis_start = visual_starts[i]
                ans_start = answer_starts[i]
                ans_end   = seq_lens[i]               # true end, excludes padding

                h_vis_i = h[i, vis_start : vis_start + N, :]   # [N, D_lm]

                if ans_start < ans_end:
                    h_ans = h[i, ans_start:ans_end, :].mean(0, keepdim=True)   # [1, D_lm]
                else:
                    # Degenerate: empty answer_string — use last question token
                    h_ans = h[i, ans_start - 1 : ans_start, :]

                imp = F.cosine_similarity(
                    h_ans.expand(N, -1), h_vis_i, dim=-1
                )                                     # [N]
                importances.append(imp)

            del h  # free hidden states before student forward

        return torch.stack(importances, dim=0).detach()          # [B, N]

    def set_training_phase(self, phase: int):
        """Configure which components are trainable based on training phase.

        Phase 0: Train TPN only
        Phase 1: Train TPN + LoRA + projector
        Phase 2: Train TPN + LoRA + projector (continued)

        Args:
            phase: Training phase (0, 1, or 2)
        """
        # Freeze all encoders (always frozen)
        if self.coarse_encoder is not None:
            for param in self.coarse_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if self.full_encoder is not None:
            for param in self.full_encoder.parameters():
                param.requires_grad = False

        if phase == 0:
            # Train TPN only — freeze all LLM components and unsel_pooler
            for param in self.tpn.parameters():
                param.requires_grad = True
            if self.unsel_pooler is not None:
                for param in self.unsel_pooler.parameters():
                    param.requires_grad = False
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False
            else:
                # Freeze entire LLaVA (LM + mm_projector + vision_tower)
                for param in self.llava.parameters():
                    param.requires_grad = False

        elif phase in [1, 2]:
            # Train TPN + LoRA + mm_projector + unsel_pooler (Phase 1.5)
            for param in self.tpn.parameters():
                param.requires_grad = True
            # UnselectedPooler: new module, always trainable in Phase 1+
            if self.unsel_pooler is not None:
                for param in self.unsel_pooler.parameters():
                    param.requires_grad = True
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = True
            else:
                # LoRA adapter params are set trainable by PEFT automatically
                # Explicitly unfreeze mm_projector (vision projector, ~9M params)
                for param in self._mm_proj.parameters():
                    param.requires_grad = True
                # Keep vision_tower frozen (never used) — check both arch locations
                for vt_path in (
                    getattr(self.llava, "vision_tower", None),
                    getattr(getattr(self.llava, "model", None), "vision_tower", None),
                ):
                    if vt_path is not None:
                        for param in vt_path.parameters():
                            param.requires_grad = False

        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 0, 1, or 2.")

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
