"""TC-ViTok: Task-Conditioned Visual Tokenizer.

This package implements a learned visual tokenization policy for efficient
vision-language reasoning. The system uses a lightweight Tokenization Policy
Network (TPN) to dynamically select k relevant visual tokens from a set of N
candidates based on the input question, reducing computational cost while
maintaining VQA accuracy.

Main components:
- models: TPN, encoders, masking utilities, full TC-ViTok model
- training: loss functions, curriculum scheduling, training loops
- data: dataset loaders, feature caching, spatial annotations
- evaluation: metrics, spatial evaluation, collapse detection, Pareto analysis
"""

__version__ = "0.1.0"
__author__ = "Muhammad Ali Shah, Talha Hassan Ul Haq"
__institution__ = "Lahore University of Management Sciences"
