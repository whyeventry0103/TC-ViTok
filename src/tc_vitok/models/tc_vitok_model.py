"""Full TC-ViTok model integrating TPN with LLaVA-1.5 backbone.

This module ties together all components into the complete TC-ViTok system:
1. Coarse visual encoding (DINOv2-S)
2. Text encoding (CLIP)
3. Token selection policy (TPN)
4. Full visual encoding (CLIP ViT-L/14)
5. Masked token projection
6. LLaVA-1.5 with LoRA

The model is trained in 3 phases:
- Phase 0: TPN warmup with frozen LLaVA
- Phase 1: Joint finetuning of TPN + LoRA + projector
- Phase 2: Domain transfer to TextVQA (optional)

Paper reference: Section 5 (Full Architecture)
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import get_peft_model, LoraConfig, TaskType

from .tpn import TokenizationPolicyNetwork
from .encoders import CoarseVisualEncoder, TextEncoder, FullVisualEncoder


class TCViTokModel(nn.Module):
    """Full TC-ViTok model with learned token selection policy.

    This model wraps LLaVA-1.5-7B and adds a lightweight TPN that dynamically
    selects k out of 576 visual tokens based on the input question. Only the
    selected tokens are projected and fed to the language model.

    Components:
        - Coarse encoder: DINOv2-S (frozen, 22M params)
        - Text encoder: CLIP ViT-B/32 (frozen, 63M params)
        - TPN: Cross-attention + MLP (trainable, 3M params)
        - Full encoder: CLIP ViT-L/14 (frozen, 304M params)
        - LLaVA: Vicuna-7B + vision projector (LoRA, 7B + 32M params)

    Total trainable: ~35M params (TPN + LoRA + projector)
    Total frozen: ~7.4B params

    Args:
        k: Token budget (number of tokens to select)
        tpn_config: Configuration dict for TPN
        lora_config: Configuration dict for LoRA
        llava_model_name: HuggingFace model ID for LLaVA base

    Paper reference: Section 5
    """

    def __init__(
        self,
        k: int = 144,
        tpn_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    ):
        super().__init__()

        self.k = k

        # TODO (Phase B): Initialize all encoder components

        # 1. Coarse visual encoder (DINOv2-S)
        self.coarse_encoder = CoarseVisualEncoder()

        # 2. Text encoder (CLIP)
        self.text_encoder = TextEncoder()

        # 3. Tokenization Policy Network (TPN)
        tpn_config = tpn_config or {}
        self.tpn = TokenizationPolicyNetwork(**tpn_config)

        # 4. Full visual encoder (CLIP ViT-L/14)
        self.full_encoder = FullVisualEncoder()

        # TODO (Phase B): Initialize LLaVA-1.5 with LoRA

        # 5. Load LLaVA-1.5 base model
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.processor = LlavaProcessor.from_pretrained(llava_model_name)

        # 6. Apply LoRA to language model
        # Target modules: q_proj, k_proj, v_proj, o_proj in attention layers
        lora_config = lora_config or {}
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

        # Vision projector is trainable (maps 1024-dim CLIP to LM embedding dim)
        # Already part of LlavaForConditionalGeneration

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
            images_full: High-res images for LLaVA, Shape: [B, 3, 336, 336]
            questions: List of question strings, length B
            labels: Optional answer token IDs for training, Shape: [B, seq_len]
            tau: Gumbel-Sigmoid temperature
            return_mask: If True, return token selection mask in output dict

        Returns:
            outputs: Dictionary containing:
                - loss: VQA loss (if labels provided)
                - logits: Language model logits
                - mask: Token selection mask (if return_mask=True)
                - scores: TPN raw scores (if return_mask=True)

        Notes:
            - In Phase 0, LLaVA is frozen
            - In Phase 1+, TPN + LoRA + projector are trained jointly

        Paper reference: Section 7 (Training Procedure)
        """
        batch_size = len(questions)

        # TODO (Phase C): Implement full forward pass

        # Step 1: Extract coarse visual features
        v_coarse = self.coarse_encoder(images_coarse)  # [B, 49, 384]

        # Step 2: Extract text embeddings
        q_embed = self.text_encoder(questions)  # [B, 512]

        # Step 3: Run TPN to get token selection mask
        mask, scores = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=tau,
            training=self.training,
        )  # mask: [B, 576], scores: [B, 576]

        # Step 4: Extract full visual features
        v_full = self.full_encoder(images_full)  # [B, 576, 1024]

        # Step 5: Apply mask to select tokens
        # Expand mask: [B, 576] -> [B, 576, 1]
        mask_expanded = mask.unsqueeze(-1)
        v_masked = v_full * mask_expanded  # Zero out non-selected tokens

        # TODO (Phase C): Handle variable-length token sequences
        # Currently this creates a [B, 576, 1024] tensor with many zeros.
        # For efficiency, we should pack only the selected k tokens.
        # This requires dynamic sequence length handling in LLaVA.
        # For now, we keep the full 576-token format.

        # Step 6: Forward through LLaVA
        # TODO (Phase C): Integrate v_masked with LLaVA's vision tower
        # This requires careful handling of LLaVA's internal vision encoder.
        # We bypass the CLIP encoder and directly inject our masked features
        # into the multi-modal projector.

        # Placeholder: return dict with required outputs
        outputs = {
            "mask": mask if return_mask else None,
            "scores": scores if return_mask else None,
        }

        # TODO (Phase C): Complete LLaVA integration
        # outputs["loss"] = ...
        # outputs["logits"] = ...

        return outputs

    def set_training_phase(self, phase: int):
        """Configure which components are trainable based on training phase.

        Phase 0: Train TPN only
        Phase 1: Train TPN + LoRA + projector
        Phase 2: Train TPN + LoRA + projector (continued)

        Args:
            phase: Training phase (0, 1, or 2)
        """
        # TODO (Phase C): Implement phase-specific parameter freezing

        if phase == 0:
            # Freeze everything except TPN
            for param in self.parameters():
                param.requires_grad = False
            for param in self.tpn.parameters():
                param.requires_grad = True

        elif phase in [1, 2]:
            # Freeze encoders, train TPN + LoRA + projector
            for param in self.coarse_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.full_encoder.parameters():
                param.requires_grad = False

            for param in self.tpn.parameters():
                param.requires_grad = True
            # LoRA params are already trainable via PEFT
            # Vision projector params are trainable by default

        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 0, 1, or 2.")

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return number of parameters.

        Args:
            trainable_only: If True, count only trainable params

        Returns:
            num_params: Total parameter count
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
