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

        self.llava = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",  # BUG-16 fix: load on CPU first; model.to(device) moves to GPU later
        )
        self.processor = LlavaProcessor.from_pretrained(model_name)

        # Detect architecture and locate language model + mm_projector
        if hasattr(self.llava, "language_model"):
            # OLD architecture: language_model is LlamaForCausalLM (has generate())
            lm_target = self.llava.language_model
            self._mm_proj = self.llava.multi_modal_projector
            vision_tower = getattr(self.llava, "vision_tower", None)
            lora_task = TaskType.CAUSAL_LM
        elif hasattr(self.llava, "model") and hasattr(self.llava.model, "language_model"):
            # NEW architecture: language_model is LlamaModel (no generate(); use FEATURE_EXTRACTION)
            lm_target = self.llava.model.language_model
            self._mm_proj = self.llava.model.multi_modal_projector
            vision_tower = getattr(self.llava.model, "vision_tower", None)
            lora_task = TaskType.FEATURE_EXTRACTION
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

        # VQA classification head: LLM hidden size → answer vocabulary
        # answer_vocab_size must match the dataset's answer_to_idx vocab
        # (VQAv2Dataset defaults to top-3129; GQADataset to top-1878)
        llm_hidden_size = self.llava.config.text_config.hidden_size  # 4096 for LLaVA-1.5-7B
        answer_vocab_size = lora_config.get("answer_vocab_size", 3129)
        self.vqa_head = nn.Linear(llm_hidden_size, answer_vocab_size)

    def forward(
        self,
        images_coarse: torch.Tensor,
        images_full: torch.Tensor,
        questions: list[str],
        labels: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        return_idx: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through TC-ViTok model.

        Args:
            images_coarse: Low-res images for TPN, Shape: [B, 3, 98, 98]
            images_full: High-res images for CLIP ViT-L, Shape: [B, 3, 336, 336]
            questions: List of question strings, length B
            labels: Optional answer token IDs, Shape: [B, seq_len]
            tau: Gumbel-Sigmoid temperature
            return_idx: If True, include idx and scores in output

        Returns:
            outputs: Dictionary containing:
                - loss: VQA loss (if labels provided)
                - logits: Language model logits
                - idx: Top-k gather indices [B, k] (if return_idx)
                - scores: TPN raw scores [B, N]
                - selected_tokens: Gathered visual tokens [B, k, D]
        """
        # Step 1: Extract coarse visual features (frozen DINOv2-S)
        v_coarse = self.coarse_encoder(images_coarse)  # [B, 49, 384]

        # Step 2: Extract text embeddings (frozen CLIP)
        q_embed = self.text_encoder(questions)  # [B, 512]

        # Step 3: TPN produces top-k gather indices
        idx, s_N = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=tau,
            training=self.training,
        )  # idx: [B, k], s_N: [B, N]

        # Step 4: Extract full visual features (frozen CLIP ViT-L/14)
        v_full = self.full_encoder(images_full)  # [B, 576, 1024]

        # Step 5: GATHER selected tokens (real compression!)
        # Output: [B, k, 1024] — LLM sees k tokens, not N
        v_selected = gather_tokens(v_full, idx)

        # Step 6: Forward through LLM backbone
        if self.use_stub_llm:
            llm_outputs = self.llm(visual_tokens=v_selected, labels=labels)
        else:
            llm_outputs = self._llava_forward(v_selected, questions, labels)

        # Build output dict
        outputs = {
            "logits": llm_outputs.get("logits"),
            "scores": s_N,
            "selected_tokens": v_selected,
        }

        if "loss" in llm_outputs:
            outputs["loss"] = llm_outputs["loss"]

        if return_idx:
            outputs["idx"] = idx

        return outputs

    def forward_cached(
        self,
        v_coarse: torch.Tensor,
        q_embed: torch.Tensor,
        images_full: torch.Tensor,
        v_full: Optional[torch.Tensor] = None,
        questions: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        return_idx: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using pre-computed cached features.

        Bypasses the frozen DINOv2-S and CLIP text encoders by accepting
        pre-computed v_coarse and q_embed directly from HDF5 cache.
        Still runs the full visual encoder (CLIP ViT-L/14) live.

        Args:
            v_coarse: Cached coarse features, Shape: [B, 49, 384]
            q_embed: Cached text embeddings, Shape: [B, 512]
            images_full: High-res images for CLIP ViT-L, Shape: [B, 3, 336, 336]
            questions: List of question strings (needed for LLaVA text conditioning)
            labels: Optional answer class indices, Shape: [B, 1]
            tau: Gumbel-Sigmoid temperature
            return_idx: If True, include idx and scores in output

        Returns:
            outputs: Same format as forward()
        """
        # Step 1: TPN produces top-k gather indices (using cached features)
        idx, s_N = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=tau,
            training=self.training,
        )

        # Step 2: Extract full visual features (frozen CLIP ViT-L/14)
        if v_full is None:
            if self.full_encoder is None:
                raise RuntimeError(
                    "forward_cached() requires full_encoder but load_encoders=False. "
                    "Either initialize with load_encoders=True or pre-compute v_full externally."
                )
            v_full = self.full_encoder(images_full)  # [B, 576, 1024]

        # Step 3: GATHER selected tokens
        v_selected = gather_tokens(v_full, idx)  # [B, k, 1024]

        # Step 4: Forward through LLM backbone
        if self.use_stub_llm:
            llm_outputs = self.llm(visual_tokens=v_selected, labels=labels)
        else:
            llm_outputs = self._llava_forward(v_selected, questions, labels)

        # Build output dict
        outputs = {
            "logits": llm_outputs.get("logits"),
            "scores": s_N,
            "selected_tokens": v_selected,
        }

        if "loss" in llm_outputs:
            outputs["loss"] = llm_outputs["loss"]

        if return_idx:
            outputs["idx"] = idx

        return outputs

    def _llava_forward(
        self,
        v_selected: torch.Tensor,
        questions: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through real LLaVA with injected visual tokens + question text.

        Pipeline:
          1. Project k gathered tokens through LLaVA's mm_projector → [B, k, H]
          2. Tokenize question strings, embed via LM embedding layer → [B, T, H]
          3. Concatenate [visual_prefix | question_text] → [B, k+T, H]
          4. Forward through LoRA-adapted LM, collect last hidden states
          5. Pool over visual positions → [B, H], apply VQA head → [B, answer_vocab]
          6. Cross-entropy against classification labels [B, 1]

        This keeps the training interface compatible with datasets.py (which stores
        answer class indices, not tokenized sequences) while properly conditioning
        the LM on both visual tokens and the question text.

        Args:
            v_selected: Selected visual tokens, Shape: [B, k, 1024]
            questions: List of question strings, length B (optional; visual-only if None)
            labels: Answer class indices, Shape: [B, 1], values in [0, answer_vocab_size)

        Returns:
            outputs: Dict with 'logits' [B, 1, answer_vocab] and optionally 'loss'
        """
        import torch.nn.functional as F_local

        B, k, _ = v_selected.shape
        device = v_selected.device
        lm = self._llm

        # Step 1: Project visual tokens through mm_projector
        # mm_projector is a 2-layer MLP: 1024 → 4096 (LLaVA-1.5-7B)
        v_projected = self._mm_proj(v_selected)  # [B, k, H]

        # Step 2: Tokenize questions and embed (if provided)
        if questions is not None:
            tokenizer = self.processor.tokenizer
            text_inputs = tokenizer(
                list(questions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True,
            )
            input_ids = text_inputs.input_ids.to(device)           # [B, T]
            text_attn = text_inputs.attention_mask.to(device)      # [B, T]

            # Embed text tokens using the LM embedding layer
            embed_fn = lm.get_input_embeddings()
            text_embeds = embed_fn(input_ids)                      # [B, T, H]

            # Concatenate: visual prefix first, then question text
            combined_embeds = torch.cat([v_projected, text_embeds], dim=1)  # [B, k+T, H]
            vis_attn = torch.ones(B, k, device=device, dtype=torch.long)
            combined_mask = torch.cat([vis_attn, text_attn], dim=1)        # [B, k+T]
        else:
            # No question text — visual tokens only (not recommended for production)
            combined_embeds = v_projected                                    # [B, k, H]
            combined_mask = torch.ones(B, k, device=device, dtype=torch.long)

        # Step 3: Forward through LoRA-adapted language model
        # Use no_grad when LLM is fully frozen (Phase 0) to save activation memory
        from contextlib import nullcontext
        _lm_frozen = not any(p.requires_grad for p in lm.parameters())
        _lm_ctx = torch.no_grad() if _lm_frozen else nullcontext()
        with _lm_ctx:
            lm_outputs = lm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                output_hidden_states=True,
            )

        # Step 4: Pool last hidden state over visual token positions
        # Using visual positions only (first k positions) focuses the pooling
        # on the visual representation rather than the question text
        last_hidden = lm_outputs.hidden_states[-1]   # [B, k+T, H] (or [B, k, H])
        visual_hidden = last_hidden[:, :k, :]        # [B, k, H]
        pooled = visual_hidden.mean(dim=1)           # [B, H]

        # Step 5: VQA classification head: H → answer_vocab
        logits = self.vqa_head(pooled).unsqueeze(1)  # [B, 1, answer_vocab]

        result: Dict[str, torch.Tensor] = {"logits": logits}

        # Step 6: Cross-entropy against answer class labels
        if labels is not None:
            loss = F_local.cross_entropy(
                logits.squeeze(1),   # [B, answer_vocab]
                labels.view(-1),     # [B]
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

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
        B, k, _ = v_selected.shape
        device = v_selected.device
        tokenizer = self.processor.tokenizer
        lm = self._llm

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

        # Generate answer tokens
        generated_ids = lm.generate(
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

    @torch.no_grad()
    def generate(
        self,
        images_coarse: torch.Tensor,
        images_full: torch.Tensor,
        questions: list[str],
        max_new_tokens: int = 20,
    ) -> list[str]:
        """Generate answers for VQA inference.

        Args:
            images_coarse: Low-res images, Shape: [B, 3, 98, 98]
            images_full: High-res images, Shape: [B, 3, 336, 336]
            questions: List of question strings
            max_new_tokens: Maximum tokens to generate

        Returns:
            answers: List of generated answer strings
        """
        self.eval()

        # Encode inputs
        v_coarse = self.coarse_encoder(images_coarse)
        q_embed = self.text_encoder(questions)

        # TPN token selection (deterministic in eval)
        idx, s_N = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=self.k,
            tau=0.1,
            training=False,
        )

        # Extract full features and gather
        v_full = self.full_encoder(images_full)
        v_selected = gather_tokens(v_full, idx)

        if self.use_stub_llm:
            # StubLLM: argmax of first visual position's logit → class index string
            llm_outputs = self.llm(visual_tokens=v_selected)
            logits = llm_outputs["logits"]
            preds = logits[:, 0, :].argmax(dim=-1)
            return [str(p.item()) for p in preds]
        else:
            # Real LLaVA: generative decode with conversation template
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
            idx, s_N = self.tpn(
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
            # Train TPN only — freeze all LLM components
            for param in self.tpn.parameters():
                param.requires_grad = True
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False
            else:
                # Freeze entire LLaVA (LM + mm_projector + vision_tower)
                for param in self.llava.parameters():
                    param.requires_grad = False
                # Freeze VQA classification head too (TPN learns independently)
                for param in self.vqa_head.parameters():
                    param.requires_grad = False

        elif phase in [1, 2]:
            # Train TPN + LoRA + mm_projector + VQA head
            for param in self.tpn.parameters():
                param.requires_grad = True
            if self.use_stub_llm:
                for param in self.llm.parameters():
                    param.requires_grad = True
            else:
                # LoRA adapter params are set trainable by PEFT automatically
                # Explicitly unfreeze mm_projector (vision projector, ~9M params)
                for param in self._mm_proj.parameters():
                    param.requires_grad = True
                # Unfreeze VQA classification head
                for param in self.vqa_head.parameters():
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
