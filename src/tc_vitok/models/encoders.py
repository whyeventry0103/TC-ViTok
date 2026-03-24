"""Visual and text encoders for TC-ViTok.

This module provides wrappers around frozen pretrained encoders:
- CoarseVisualEncoder: DINOv2-Small for low-resolution features
- TextEncoder: CLIP ViT-B/32 for text embeddings
- FullVisualEncoder: CLIP ViT-L/14 for high-resolution features (in tc_vitok_model.py)

All encoders are frozen during training to reduce computational cost and
maintain the benefits of strong pretrained representations.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import timm


class CoarseVisualEncoder(nn.Module):
    """Wrapper around DINOv2-Small for coarse visual feature extraction.

    Extracts low-resolution spatial features used by TPN for token selection.
    Uses frozen DINOv2-S (ViT-S/14) pretrained on ImageNet-22k.

    Architecture:
        - Input: 98×98 RGB image
        - Output: 49 spatial tokens (7×7 grid), 384-dim each
        - Total: [batch_size, 49, 384]

    Why DINOv2-S?
        - Self-supervised → rich spatial features without labels
        - Lightweight (22M params) → fast token scoring
        - Good locality → captures spatial semantics

    Paper reference: Section 5.2
    """

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m"):
        super().__init__()

        # TODO (Phase B): Load pretrained DINOv2-Small from timm
        # Model: vit_small_patch14_dinov2.lvd142m
        # Extract spatial features before global pooling

        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        # DINOv2-S feature dimension
        self.d_model = 384

        # Expected input size
        self.input_size = 98  # Chosen to give 7×7 spatial grid (98/14 = 7)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract coarse spatial features.

        Args:
            images: Input images, Shape: [batch_size, 3, 98, 98]
                    Normalized with ImageNet stats:
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

        Returns:
            features: Spatial features, Shape: [batch_size, 49, 384]
                      49 = 7×7 spatial grid

        Notes:
            - All computation in torch.no_grad() for memory efficiency
            - Encoder is frozen, so no gradients needed
        """
        # TODO (Phase B): Implement forward pass
        # Extract patch embeddings before global pooling

        batch_size = images.size(0)

        # Forward through ViT
        features = self.encoder.forward_features(images)

        # DINOv2 returns [batch_size, num_patches + 1, d_model]
        # where +1 is the [CLS] token
        # We only need the patch tokens
        if features.size(1) == 50:  # 49 patches + 1 CLS
            features = features[:, 1:, :]  # Remove CLS token

        return features


class TextEncoder(nn.Module):
    """Wrapper around CLIP ViT-B/32 text encoder.

    Extracts text embeddings from questions using frozen CLIP text encoder.
    The [EOS] token embedding is used as the question representation.

    Architecture:
        - Input: Text string (question)
        - Output: 512-dim embedding from [EOS] token
        - Total: [batch_size, 512]

    Why CLIP text encoder?
        - Trained jointly with CLIP vision encoder
        - Strong language understanding for visual grounding
        - Commonly used in VLM pipelines

    Paper reference: Section 5.2
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()

        # TODO (Phase B): Load pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze text encoder
        for param in self.clip.text_model.parameters():
            param.requires_grad = False

        self.clip.text_model.eval()

        # CLIP-B/32 text embedding dimension
        self.d_text = 512

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Extract text embeddings from questions.

        Args:
            texts: List of question strings, length = batch_size

        Returns:
            embeddings: Text embeddings, Shape: [batch_size, 512]
                        Extracted from [EOS] token of CLIP text encoder

        Notes:
            - Automatically handles tokenization via CLIPProcessor
            - All computation in torch.no_grad() for efficiency
        """
        # TODO (Phase B): Implement forward pass
        # Tokenize and extract [EOS] embeddings

        # Tokenize texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to same device as model
        inputs = {k: v.to(next(self.clip.parameters()).device) for k, v in inputs.items()}

        # Forward through CLIP text encoder
        text_outputs = self.clip.text_model(**inputs)

        # Extract [EOS] token embedding
        # pooler_output is the [EOS] token representation
        embeddings = text_outputs.pooler_output

        return embeddings


class FullVisualEncoder(nn.Module):
    """Wrapper around CLIP ViT-L/14 for full visual feature extraction.

    Extracts high-resolution spatial features that are fed to LLaVA.
    This encoder is frozen and only used after TPN selects relevant tokens.

    Architecture:
        - Input: 336×336 RGB image
        - Output: 576 spatial tokens (24×24 grid), 1024-dim each
        - Total: [batch_size, 576, 1024]

    Why CLIP ViT-L/14?
        - Used by LLaVA-1.5 baseline for fair comparison
        - Strong vision-language alignment
        - High-resolution features (336px vs 224px)

    Paper reference: Section 4 (Baseline: LLaVA-1.5)
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()

        # TODO (Phase B): Load pretrained CLIP vision encoder
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze vision encoder
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False

        self.clip.vision_model.eval()

        # CLIP ViT-L/14 feature dimension
        self.d_model = 1024

        # Expected input size
        self.input_size = 336

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract full-resolution spatial features.

        Args:
            images: Input images, Shape: [batch_size, 3, 336, 336]
                    Normalized with CLIP stats:
                    mean=[0.48145466, 0.4578275, 0.40821073]
                    std=[0.26862954, 0.26130258, 0.27577711]

        Returns:
            features: Spatial features, Shape: [batch_size, 576, 1024]
                      576 = 24×24 spatial grid

        Notes:
            - All computation in torch.no_grad() for memory efficiency
            - These features are masked by TPN output before LLaVA
        """
        # TODO (Phase B): Implement forward pass
        # Extract patch embeddings before pooling

        # Forward through CLIP vision encoder
        vision_outputs = self.clip.vision_model(pixel_values=images)

        # Extract patch embeddings (exclude [CLS] token)
        # vision_outputs.last_hidden_state: [batch_size, num_patches + 1, 1024]
        features = vision_outputs.last_hidden_state[:, 1:, :]  # Remove CLS

        return features
