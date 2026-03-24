"""Model components for TC-ViTok.

This module contains the core neural network components:
- TokenizationPolicyNetwork (TPN): cross-attention-based policy network
- CoarseVisualEncoder: DINOv2-S wrapper for low-resolution features
- TextEncoder: CLIP text encoder wrapper
- Gumbel-Sigmoid STE: differentiable top-k selection
- TCViTokModel: full model integrating TPN with LLaVA-1.5
"""
