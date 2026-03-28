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

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .tpn import TokenizationPolicyNetwork
from .encoders import CoarseVisualEncoder, TextEncoder, FullVisualEncoder


def gather_selected_tokens(
    visual_tokens: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Extract selected tokens using gather (real compression).

    This replaces zero-masking with actual token extraction:
    output has shape [B, k, D] instead of [B, N, D] with zeros.
    This provides REAL compute savings in the downstream LLM.

    Args:
        visual_tokens: Full visual features, Shape: [B, N, D]
        mask: Binary selection mask, Shape: [B, N]
        k: Number of tokens to select

    Returns:
        selected: Extracted tokens, Shape: [B, k, D]
    """
    B, N, D = visual_tokens.shape

    # Get indices of selected tokens (top-k positions where mask == 1)
    # mask is binary with exactly k ones per sample from hard top-k
    indices = mask.topk(k, dim=-1).indices  # [B, k]
    indices = indices.sort(dim=-1).values   # [B, k], sorted for stable ordering

    # Gather selected tokens
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
    selected = torch.gather(visual_tokens, dim=1, index=indices_expanded)  # [B, k, D]

    return selected


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
    ):
        super().__init__()

        self.k = k
        self.use_stub_llm = use_stub_llm

        # 1. Coarse visual encoder (DINOv2-S, frozen)
        self.coarse_encoder = CoarseVisualEncoder()

        # 2. Text encoder (CLIP, frozen)
        self.text_encoder = TextEncoder()

        # 3. Tokenization Policy Network (TPN, trainable)
        tpn_config = tpn_config or {}
        self.tpn = TokenizationPolicyNetwork(**tpn_config)

        # 4. Full visual encoder (CLIP ViT-L/14, frozen)
        self.full_encoder = FullVisualEncoder()

        # 5. VLM backbone
        if use_stub_llm:
            self.llm = StubLLM(d_visual=1024)
        else:
            self._init_llava(llava_model_name, lora_config or {})

    def _init_llava(self, model_name: str, lora_config: dict):
        """Initialize LLaVA-1.5 with LoRA adapters."""
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        from peft import get_peft_model, LoraConfig, TaskType

        self.llava = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.processor = LlavaProcessor.from_pretrained(model_name)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get("rank", 32),
            lora_alpha=lora_config.get("alpha", 64),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get(
                "targets",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self.llava.language_model = get_peft_model(
            self.llava.language_model,
            peft_config,
        )

    def forward(
        self,
        images_coarse: torch.Tensor,
        images_full: torch.Tensor,
        questions: list[str],
        labels: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        return_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through TC-ViTok model.

        Args:
            images_coarse: Low-res images for TPN, Shape: [B, 3, 98, 98]
            images_full: High-res images for CLIP ViT-L, Shape: [B, 3, 336, 336]
            questions: List of question strings, length B
            labels: Optional answer token IDs, Shape: [B, seq_len]
            tau: Gumbel-Sigmoid temperature
            return_mask: If True, include mask and scores in output

        Returns:
            outputs: Dictionary containing:
                - loss: VQA loss (if labels provided)
                - logits: Language model logits
                - mask: Token selection mask (if return_mask)
                - scores: TPN raw scores (if return_mask)
                - selected_tokens: Gathered visual tokens [B, k, D]
        """
        # Step 1: Extract coarse visual features (frozen DINOv2-S)
        v_coarse = self.coarse_encoder(images_coarse)  # [B, 49, 384]

        # Step 2: Extract text embeddings (frozen CLIP)
        q_embed = self.text_encoder(questions)  # [B, 512]

        # Step 3: Run TPN to get token selection mask
        mask, scores = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=tau,
            training=self.training,
        )  # mask: [B, N], scores: [B, N]

        # Step 4: Extract full visual features (frozen CLIP ViT-L/14)
        v_full = self.full_encoder(images_full)  # [B, 576, 1024]

        # Step 5: GATHER selected tokens (real compression!)
        # Output: [B, k, 1024] — sequence length is k, NOT 576
        v_selected = gather_selected_tokens(v_full, mask, self.k)

        # Step 6: Forward through LLM backbone
        if self.use_stub_llm:
            llm_outputs = self.llm(visual_tokens=v_selected, labels=labels)
        else:
            # Real LLaVA integration path
            # Inject v_selected into LLaVA's projector, bypassing its ViT
            v_projected = self.llava.multi_modal_projector(v_selected)
            # TODO: Full LLaVA integration requires custom forward pass
            # to inject projected tokens and handle text/answer generation.
            # For now, use projector output shape for consistency check.
            llm_outputs = {"logits": v_projected}  # placeholder

        # Build output dict
        outputs = {
            "logits": llm_outputs.get("logits"),
            "scores": scores,
            "selected_tokens": v_selected,
        }

        if "loss" in llm_outputs:
            outputs["loss"] = llm_outputs["loss"]

        if return_mask:
            outputs["mask"] = mask

        return outputs

    def set_training_phase(self, phase: int):
        """Configure which components are trainable based on training phase.

        Phase 0: Train TPN only
        Phase 1: Train TPN + LoRA + projector
        Phase 2: Train TPN + LoRA + projector (continued)

        Args:
            phase: Training phase (0, 1, or 2)
        """
        # Freeze all encoders (always frozen)
        for param in self.coarse_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.full_encoder.parameters():
            param.requires_grad = False

        if phase == 0:
            # Train TPN only
            for param in self.tpn.parameters():
                param.requires_grad = True
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False
            else:
                for param in self.llava.parameters():
                    param.requires_grad = False

        elif phase in [1, 2]:
            # Train TPN + LLM adapters
            for param in self.tpn.parameters():
                param.requires_grad = True
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = True
            # LoRA params are already trainable via PEFT when not using stub

        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 0, 1, or 2.")

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
