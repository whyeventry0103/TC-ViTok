# Codex Task Bundle — Phase 2 Track C (Fundamental-Limit Paper)

## Mission

Phase 1 diagnostics decisively confirmed Track C: causal token importance cannot be predicted from encoder features (D4 = -0.268). This bundle turns that finding into a **NeurIPS / TMLR / ICML-grade publication** by extending the result along three axes:

- **C1 — Scale verification:** Reproduce D4 at 5× sample size on the same encoder. Settle whether the unpredictability is a real signal or a small-sample artifact.
- **C2 — Mid-LLM predictability:** Show that the same probe trained on mid-LLM hidden states *does* recover causal importance. This is the mechanistic core of the paper: information needed for selection exists in the LLM but not in encoder features.
- **C3 — Cross-encoder verification:** Reproduce D4 on SigLIP and DINOv2 features. If unpredictability holds, the limit is encoder-agnostic.

Total compute budget: ~120-150 GPU-h across 3 weeks. Outcome: one strong paper.

## Critical instructions

1. **Do not start until backup checklist is complete.** Phase 1 artifacts must be safe before any new compute.
2. **Reuse Phase 1 infrastructure.** `compute_causal_importance.py`, `train_causal_probe.py`, the data loaders. Do not rewrite.
3. **Save everything.** All probe trainings, all feature dumps, all intermediate JSONs.
4. **One-variable-per-experiment.** When testing alternative encoders, change only the encoder. Same images, same questions, same probe architecture.
5. **Pre-register every probe's R² interpretation.** Before running, decide what threshold settles the question. Document in the bundle.

## Output that matters

By end of bundle, produce **one figure and one table** that anchor the paper:

- **Figure 1 — Predictability curve.** X-axis: source of features (CLIP enc, SigLIP enc, DINOv2 enc, LLM layer 0, layer 2, layer 4, layer 8, layer 16, layer 28). Y-axis: probe R² on held-out images. Expected shape: low (< 0.2) for all encoder rows, rises monotonically with LLM depth.
- **Table 1 — Per-encoder D4.** Three rows (CLIP, SigLIP, DINOv2) × two columns (linear R², MLP R²) on N=1000 images each.

That figure and table are the visual claims of the paper. Everything else supports them.

## Execution order

```
Week 1: C1 — Scale verification + paper outline
   ├─ C1.1 Causal importance on 1000 GQA samples           [~12 GPU-h]
   ├─ C1.2 Probe training and bootstrapped CI               [~1 GPU-h]
   ├─ C1.3 Per-question-type breakdown                      [~1 GPU-h]
   └─ Paper outline draft                                   [no GPU]

Week 2: C2 — Mid-LLM predictability
   ├─ C2.1 Mid-LLM activation extraction (6 layers)         [~20 GPU-h]
   ├─ C2.2 Probe per-layer R² sweep                         [~2 GPU-h]
   └─ C2.3 Layer-onset analysis                             [no GPU]

Week 3: C3 — Cross-encoder verification + Figure 1
   ├─ C3.1 SigLIP feature extraction + ablation             [~15 GPU-h]
   ├─ C3.2 DINOv2 feature extraction + ablation             [~15 GPU-h]
   ├─ C3.3 Combined Figure 1 generation                     [no GPU]
   └─ Draft sections 4-5                                    [no GPU]

Week 4: Writeup
   ├─ Robustness checks (probe capacity, regularization)    [~5 GPU-h]
   └─ Full paper draft                                      [no GPU]
```

Total: ~70-75 GPU-h, plus ~5 GPU-h slack. Comfortable within 120 GPU-h budget.

---

# C1 — Scale Verification (Week 1)

## C1.1 — Causal importance on 1000 GQA samples

### Goal
Reproduce D4 at 5× sample size to rule out small-sample artifacts and tighten error bars.

### Run

```bash
mkdir -p runs/c1_causal_1000

# Reuse existing script with N=1000
DATA_ROOT=~/tc-vitok-data python3 scripts/compute_causal_importance.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --n-images 1000 \
  --ablation-batch-size 4 \
  --output runs/c1_causal_1000/causal_data.pt \
  2>&1 | tee runs/c1_causal_1000/compute.log
```

Expected wall-clock: ~10-12 GPU-h with `ablation-batch-size 4`. Increase batch size to 8 or 16 if GPU memory allows (test in smoke first).

### Sanity check first

Before launching 1000 samples, smoke test with `--n-images 50` on a different seed. Verify:
- Output has shape `[50, 576]` for importances and `[50, 576, 1024]` for features.
- Importance values are in the same magnitude range as the original D4 run (mean ~0.002, std ~0.006).
- The smoke probe R² (linear) is roughly consistent with the 200-sample result (within ± 0.15 — small samples are noisy but should not flip sign).

```bash
DATA_ROOT=~/tc-vitok-data python3 scripts/compute_causal_importance.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --n-images 50 \
  --ablation-batch-size 4 \
  --output runs/c1_causal_1000/smoke50.pt

python3 scripts/train_causal_probe.py \
  --data runs/c1_causal_1000/smoke50.pt \
  --probe linear --epochs 50 --batch-size 4096 \
  --output runs/c1_causal_1000/smoke50_linear.json
```

If smoke linear R² is < -0.15 (consistent with original) → proceed.
If smoke R² is significantly positive (> 0.3) → flag immediately, something changed.

## C1.2 — Probe training with bootstrapped confidence interval

### Goal
Compute D4 at N=1000 with explicit 95% confidence interval via bootstrap resampling over images.

### New script `scripts/probe_with_bootstrap.py`

```python
"""Train causal-importance probe with bootstrap CI over held-out images."""
import argparse, json, random
import numpy as np
import torch
from torch import nn

class LinearProbe(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.linear = nn.Linear(d, 1)
    def forward(self, x): return self.linear(x).squeeze(-1)

class MLPProbe(nn.Module):
    def __init__(self, d=1024, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_and_score(probe, X_train, Y_train, X_val, Y_val, epochs=50, batch=4096, lr=1e-3):
    probe.train()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = X_train.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            pred = probe(X_train[idx])
            loss = ((pred - Y_train[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred_val = probe(X_val)
        ss_res = ((Y_val - pred_val) ** 2).sum().item()
        ss_tot = ((Y_val - Y_val.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = torch.load(args.data, map_location="cpu")
    importances = data["importances"]                # [N, 576]
    features = data["features"]                      # [N, 576, 1024]
    N, P, D = features.shape

    random.seed(args.seed); torch.manual_seed(args.seed)

    r2s = []
    for b in range(args.n_bootstrap):
        # Resample images (not tokens)
        idx_train = torch.tensor(random.choices(range(N), k=int(0.8 * N)))
        idx_val_pool = list(set(range(N)) - set(idx_train.tolist()))
        if len(idx_val_pool) < 5:
            continue
        idx_val = torch.tensor(idx_val_pool)
        X_train = features[idx_train].reshape(-1, D)
        Y_train = importances[idx_train].reshape(-1)
        X_val = features[idx_val].reshape(-1, D)
        Y_val = importances[idx_val].reshape(-1)
        cuda = X_train.cuda() if torch.cuda.is_available() else X_train
        Y_train_d = Y_train.cuda() if torch.cuda.is_available() else Y_train
        X_val_d = X_val.cuda() if torch.cuda.is_available() else X_val
        Y_val_d = Y_val.cuda() if torch.cuda.is_available() else Y_val
        probe = (LinearProbe(D) if args.probe == "linear" else MLPProbe(D))
        if torch.cuda.is_available(): probe = probe.cuda()
        r2 = train_and_score(probe, cuda, Y_train_d, X_val_d, Y_val_d, epochs=args.epochs)
        r2s.append(r2)
        print(f"bootstrap {b+1}/{args.n_bootstrap}: r2 = {r2:.4f}")
        del probe; torch.cuda.empty_cache()

    r2s = np.array(r2s)
    result = {
        "probe": args.probe,
        "n_bootstrap": len(r2s),
        "mean_r2": float(r2s.mean()),
        "std_r2": float(r2s.std()),
        "ci_95_lower": float(np.percentile(r2s, 2.5)),
        "ci_95_upper": float(np.percentile(r2s, 97.5)),
        "all_r2": r2s.tolist(),
    }
    json.dump(result, open(args.output, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### Run

```bash
python3 scripts/probe_with_bootstrap.py \
  --data runs/c1_causal_1000/causal_data.pt \
  --probe linear --n-bootstrap 20 \
  --output runs/c1_causal_1000/probe_linear_boot.json

python3 scripts/probe_with_bootstrap.py \
  --data runs/c1_causal_1000/causal_data.pt \
  --probe mlp --n-bootstrap 20 \
  --output runs/c1_causal_1000/probe_mlp_boot.json
```

### Pre-registered interpretation
- **Mean R² < 0 with 95% CI upper bound < 0.1:** unpredictability robustly confirmed at scale. Paper headline result.
- **Mean R² in [0, 0.2]:** weak signal exists at scale; paper claim weakens to "very weakly predictable."
- **Mean R² > 0.2:** unpredictability was a small-sample artifact. Reconsider track decision.

### Deliverable
`runs/c1_causal_1000/probe_linear_boot.json` and `probe_mlp_boot.json`. Plus a 1-paragraph interpretation.

## C1.3 — Per-question-type breakdown

### Goal
Test whether unpredictability is uniform across question types or concentrated in specific categories. If "what color" questions have low R² but "spatial location" questions have high R² (or vice versa), the limit is question-dependent — useful nuance for the paper.

### Approach

Classify the 1000 GQA questions into types using a simple heuristic:

```python
def classify_question(q):
    q = q.lower()
    if any(k in q for k in ["what color", "what colour"]): return "color"
    if any(k in q for k in ["how many", "count", "number of"]): return "count"
    if any(k in q for k in ["is there", "are there"]): return "existence"
    if any(k in q for k in ["where", "left", "right", "top", "bottom", "behind", "in front"]): return "spatial"
    if any(k in q for k in ["what is", "what's", "what kind"]): return "identity"
    return "other"
```

Then train a probe on the *subset* of tokens belonging to each question type and compare R²s.

### New script `scripts/probe_by_question_type.py`

```python
"""Train probes on subsets of the causal data split by question type."""
# Reuse the train_and_score function from probe_with_bootstrap.py
# Load data and metadata
# For each question type, restrict to images with that question type
# Train probe; record R²
# Compare
```

### Deliverable
`runs/c1_causal_1000/per_question_type_r2.json`. A bar chart in `figures/c1_per_qtype.pdf`.

---

# C2 — Mid-LLM Predictability (Week 2)

This is the most important block. C2 shows that the unpredictability is *encoder-specific*, not fundamental to the question. Mid-LLM hidden states contain the information; encoder features do not. This is the explanatory mechanism for the paper.

## C2.1 — Mid-LLM activation extraction

### Goal
For the same 1000 images and questions used in C1, extract LLaVA's intermediate hidden states at the visual-token positions, at multiple LLM layer depths. Save per-layer per-token feature tensors.

### Architecture

LLaVA-1.5-7B has 32 transformer layers. Probe layers: **0, 2, 4, 8, 16, 24, 28** (7 depths). Layer 0 is the embedding output; layer 32 would be the LM head input.

Per layer, save:
- Hidden states at the 576 visual-token positions: shape `[1000, 576, 4096]` (LLaVA hidden dim is 4096).

That's 1000 × 576 × 4096 × 2 bytes (bf16) × 7 layers = ~16 GB. Significant but manageable. Store on a fast disk.

### New script `scripts/extract_mid_llm_features.py`

```python
"""Extract LLaVA hidden states at multiple layers for visual tokens.

For each (image, question) pair:
1. Forward through LoRA-576 (full visual context).
2. Hook hidden states at layers 0, 2, 4, 8, 16, 24, 28.
3. Restrict to the visual-token positions (576 of them).
4. Save as a dict: {layer: tensor[1000, 576, 4096]}.

Compute: ~20 GPU-h for 1000 images at LLaVA-576 inference rates.
"""
import argparse
import torch
from pathlib import Path
from scripts.evaluate import load_model

LAYERS_TO_PROBE = [0, 2, 4, 8, 16, 24, 28]

captured = {}

def make_hook(layer_idx):
    def hook(module, inp, out):
        # out is typically (hidden_states, ...) or tensor
        h = out[0] if isinstance(out, tuple) else out
        captured[f"L{layer_idx}"] = h.detach().to("cpu", dtype=torch.bfloat16)
    return hook


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="~/tc-vitok-data")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = load_model(checkpoint=args.checkpoint, baseline="full_576", k=576).eval().cuda()

    # Locate the LLM layer modules. LLaVA wraps the model in different ways
    # depending on version; the layers list is typically at:
    layers = None
    candidates = [
        ("model.llava.language_model.model.layers", None),
        ("model.llava.model.layers", None),
        ("model.language_model.model.layers", None),
    ]
    # walk the model to find the layer list
    for path, _ in candidates:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            layers = obj
            print(f"Found layers at {path}, count={len(layers)}")
            break
        except AttributeError:
            continue
    assert layers is not None, "Could not find LLaVA layers; inspect model structure"

    # Register hooks
    handles = []
    for li in LAYERS_TO_PROBE:
        handles.append(layers[li].register_forward_hook(make_hook(li)))

    # Locate visual-token positions in the input (same logic as P1.3)
    # LLaVA tokenization: typically [BOS, <576 visual tokens>, <text tokens>, ...]
    VIS_START, VIS_END = 1, 577

    # Loop over GQA val 1000 samples
    from tc_vitok.data.datasets import GQADataset, collate_fn
    from torch.utils.data import DataLoader
    ds = GQADataset(args.data_root, split="val", max_samples=args.n_images)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    accum = {f"L{li}": [] for li in LAYERS_TO_PROBE}
    image_ids = []
    for batch in dl:
        captured.clear()
        _ = model.forward(
            images_coarse=batch["images_coarse"].cuda(),
            images_full=batch["images_full"].cuda(),
            questions=batch["questions"],
            max_new_tokens=1,
        )
        for li in LAYERS_TO_PROBE:
            key = f"L{li}"
            h = captured[key]   # [B, seq, 4096] bf16 on CPU
            visual_h = h[:, VIS_START:VIS_END, :]   # [B, 576, 4096]
            accum[key].append(visual_h)
        image_ids.extend(batch["image_ids"])

    # Save per-layer tensors
    for handle in handles: handle.remove()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {key: torch.cat(v, dim=0) for key, v in accum.items()}
    save_dict["image_ids"] = image_ids
    torch.save(save_dict, args.output)
    print(f"Saved to {args.output}")
    for key, t in save_dict.items():
        if isinstance(t, torch.Tensor):
            print(f"  {key}: shape={tuple(t.shape)}, dtype={t.dtype}")


if __name__ == "__main__":
    main()
```

### Sanity check
Run with `--n-images 5` first. Verify:
- The output dict has 7 keys L0, L2, L4, L8, L16, L24, L28 plus image_ids.
- Each tensor has shape `[5, 576, 4096]`.
- Hidden states are non-zero (the model is actually computing).
- The VIS_START / VIS_END indices isolate the 576 visual tokens (compare against P1.3 settings).

If VIS_START is wrong, the per-layer features won't be visual — they'll be a mix of text and visual. This invalidates all of C2. **Verify before scaling up.**

### Run

```bash
mkdir -p runs/c2_midllm
DATA_ROOT=~/tc-vitok-data python3 scripts/extract_mid_llm_features.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --n-images 1000 \
  --output runs/c2_midllm/midllm_features.pt
```

### Deliverable
`runs/c2_midllm/midllm_features.pt` containing 7 layer tensors.

## C2.2 — Probe per-layer R² sweep

### Goal
For each LLM layer in {0, 2, 4, 8, 16, 24, 28}, train a probe to predict the *same* causal-importance targets (from `runs/c1_causal_1000/causal_data.pt`) using the hidden states from that layer. Plot R² vs layer.

Expected outcome: R² is near zero or negative at L0 (LLM input embeddings, close to encoder features), then *rises* with depth, peaking somewhere around layer 8-16 where text-visual mixing has occurred.

### New script `scripts/probe_per_layer.py`

```python
"""Train probes at each LLM layer; predict causal importance from layer hidden states."""
import argparse, json
import torch
import numpy as np
# Reuse train_and_score, LinearProbe, MLPProbe from probe_with_bootstrap.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--causal-data", required=True)
    p.add_argument("--midllm-features", required=True)
    p.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    causal = torch.load(args.causal_data, map_location="cpu")
    midllm = torch.load(args.midllm_features, map_location="cpu")
    importances = causal["importances"]   # [1000, 576]
    N, P = importances.shape

    # Layers to probe
    layers = [k for k in midllm.keys() if k.startswith("L")]

    results = {}
    for layer_key in layers:
        feats = midllm[layer_key].float()   # [1000, 576, 4096]
        d = feats.shape[-1]

        # Train/val split per image
        n_train = int(0.8 * N)
        train_idx = torch.arange(0, n_train)
        val_idx = torch.arange(n_train, N)

        X_train = feats[train_idx].reshape(-1, d).cuda()
        Y_train = importances[train_idx].reshape(-1).cuda()
        X_val = feats[val_idx].reshape(-1, d).cuda()
        Y_val = importances[val_idx].reshape(-1).cuda()

        from scripts.probe_with_bootstrap import LinearProbe, MLPProbe, train_and_score
        probe = (LinearProbe(d) if args.probe == "linear" else MLPProbe(d)).cuda()
        r2 = train_and_score(probe, X_train, Y_train, X_val, Y_val, epochs=args.epochs)
        results[layer_key] = r2
        print(f"{layer_key}: R^2 = {r2:.4f}")
        del probe, X_train, Y_train, X_val, Y_val; torch.cuda.empty_cache()

    json.dump(results, open(args.output, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

### Run

```bash
python3 scripts/probe_per_layer.py \
  --causal-data runs/c1_causal_1000/causal_data.pt \
  --midllm-features runs/c2_midllm/midllm_features.pt \
  --probe mlp \
  --output runs/c2_midllm/probe_per_layer.json

# Also linear
python3 scripts/probe_per_layer.py \
  --causal-data runs/c1_causal_1000/causal_data.pt \
  --midllm-features runs/c2_midllm/midllm_features.pt \
  --probe linear \
  --output runs/c2_midllm/probe_per_layer_linear.json
```

### Pre-registered interpretation
- **R² at L0 ≈ R² at encoder features:** Confirms encoder features and L0 hidden states carry similar information (expected — L0 is just the embedding of the encoder features after MLP projector).
- **R² rises with layer depth:** Confirms the mid-LLM compression advantage of FastV is information-theoretic. The information needed for selection emerges as the LLM processes.
- **R² peaks at some intermediate layer L\*:** L\* is the "optimal point for compression" — neither too early (info not yet computed) nor too late (already used for prediction).
- **R² stays low at all layers:** Causal importance is *globally* unpredictable. Strongest possible Track C result.

### Deliverable
`runs/c2_midllm/probe_per_layer.json` (MLP) and `_linear.json`. The R²-vs-layer table is the headline finding.

## C2.3 — Layer-onset analysis

### Goal
Identify the specific layer where R² becomes "high" (say, ≥ 0.3). This is the "onset depth" of selectivity.

Pure analysis on C2.2 output. Find the smallest layer index where R² ≥ 0.3. Report onset depth and the slope of R² vs layer.

### Deliverable
A short markdown note `runs/c2_midllm/onset_analysis.md`.

---

# C3 — Cross-Encoder Verification (Week 3)

This shows the unpredictability finding is not CLIP-specific. If SigLIP and DINOv2 features also yield R² < 0.2, the finding is encoder-agnostic.

## C3.1 — SigLIP features + ablation

### Goal
Replace CLIP ViT-L/14 features with SigLIP features in the causal-importance computation. Same images, same questions, same ablation procedure.

### Approach

1. Load `google/siglip-large-patch16-384` via HuggingFace. Use its visual features as the "encoder features" input to the LLM (replacing CLIP).
2. Do **not** re-train LoRA — we are testing whether SigLIP features make causal importance predictable, not whether SigLIP improves the system. Use the existing Phase-1-Final LoRA on top of CLIP features in the forward pass; SigLIP features are used *only as the input to the probe*.

Wait — clarify this. There are two designs:
- **Design A:** Replace CLIP entirely with SigLIP throughout (encoder + LLM input). Then compute causal importance over SigLIP-derived tokens. Trains a fresh LoRA. **Heavy.**
- **Design B:** Keep CLIP as the LLM input. Compute causal importance over CLIP tokens (already done in C1). Then train a probe to predict that importance from *SigLIP features at the same patch locations*.

**Design B is correct.** We are asking: "is the causal importance pattern predictable from SigLIP features at corresponding patches?" Not: "does SigLIP work as a VLM encoder?"

### New script `scripts/extract_alt_encoder_features.py`

```python
"""Extract per-patch features from an alternative encoder for the same images.

For each GQA image:
1. Load image at the resolution the encoder expects.
2. Forward through alternative encoder (SigLIP or DINOv2).
3. Extract per-patch features.
4. Save as [N, num_patches, d_alt].

Note: num_patches may differ from 576 (CLIP). Resample / interpolate to 576 if needed.
"""
import argparse
import torch
from PIL import Image
from pathlib import Path

ENCODERS = {
    "siglip": ("google/siglip-large-patch16-384", 384, 576),   # 24x24 patches
    "dinov2": ("facebook/dinov2-large", 518, 1369),            # 37x37 patches
}

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=list(ENCODERS.keys()), required=True)
    p.add_argument("--causal-data", required=True, help="To get the same image set")
    p.add_argument("--data-root", default="~/tc-vitok-data")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    name, size, n_patches = ENCODERS[args.encoder]

    if args.encoder == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        model = SiglipModel.from_pretrained(name).cuda().eval()
        proc = SiglipProcessor.from_pretrained(name)
    elif args.encoder == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(name).cuda().eval()
        proc = AutoImageProcessor.from_pretrained(name)

    # Get image list from the causal-data run's metadata.
    # If image IDs aren't saved, we need to re-enumerate the GQA val set with the
    # same seed/order as compute_causal_importance.py used.
    causal = torch.load(args.causal_data, map_location="cpu")
    image_ids = causal.get("image_ids")
    if image_ids is None:
        # Recompute image list using the same loader settings
        from tc_vitok.data.datasets import GQADataset
        N = causal["importances"].shape[0]
        ds = GQADataset(args.data_root, split="val", max_samples=N)
        image_ids = [s["image_id"] for s in ds]

    # For each image, forward through encoder, get per-patch features
    accum = []
    for img_id in image_ids:
        # Load image (path varies by dataset; use same loader as GQADataset)
        img = Image.open(get_gqa_image_path(args.data_root, img_id)).convert("RGB")
        inp = proc(images=img, return_tensors="pt").pixel_values.cuda()
        if args.encoder == "siglip":
            out = model.vision_model(pixel_values=inp)
            feats = out.last_hidden_state.squeeze(0)   # [num_patches, d]
        else:
            out = model(pixel_values=inp)
            feats = out.last_hidden_state.squeeze(0)[1:, :]   # drop CLS

        # If num_patches != 576, downsample with bilinear interpolation
        if feats.shape[0] != 576:
            side = int(feats.shape[0] ** 0.5)
            feats_2d = feats.permute(1, 0).reshape(1, -1, side, side)
            feats_resized = torch.nn.functional.interpolate(
                feats_2d, size=(24, 24), mode="bilinear", align_corners=False
            )
            feats = feats_resized.reshape(-1, 576).permute(1, 0)

        accum.append(feats.cpu().bfloat16())

    out_t = torch.stack(accum)   # [N, 576, d_alt]
    save_dict = {"features": out_t, "encoder": args.encoder, "d": out_t.shape[-1]}
    torch.save(save_dict, args.output)
    print(f"Saved {args.output}, shape={tuple(out_t.shape)}")
```

### Run

```bash
mkdir -p runs/c3_cross_encoder

# SigLIP
DATA_ROOT=~/tc-vitok-data python3 scripts/extract_alt_encoder_features.py \
  --encoder siglip \
  --causal-data runs/c1_causal_1000/causal_data.pt \
  --output runs/c3_cross_encoder/siglip_features.pt

# DINOv2
DATA_ROOT=~/tc-vitok-data python3 scripts/extract_alt_encoder_features.py \
  --encoder dinov2 \
  --causal-data runs/c1_causal_1000/causal_data.pt \
  --output runs/c3_cross_encoder/dinov2_features.pt
```

### Now train probes on these features predicting the *CLIP-derived* causal importance

```python
# Adapt probe_with_bootstrap.py to load features from an arbitrary .pt
python3 scripts/probe_with_bootstrap.py \
  --features-from runs/c3_cross_encoder/siglip_features.pt \
  --targets-from runs/c1_causal_1000/causal_data.pt \
  --probe mlp --n-bootstrap 20 \
  --output runs/c3_cross_encoder/probe_siglip_mlp.json

# Same for DINOv2
```

(Add a `--features-from` and `--targets-from` flag to `probe_with_bootstrap.py` for this.)

### Pre-registered interpretation
- **All three encoders (CLIP, SigLIP, DINOv2) yield R² < 0.2:** unpredictability is encoder-agnostic. Strongest result. Paper title: "Causal Token Importance is Encoder-Unpredictable."
- **SigLIP or DINOv2 yields R² > 0.4:** the unpredictability is CLIP-specific. Paper scopes back to "CLIP-specific limit" — still publishable but weaker.
- **DINOv2 only yields high R² (DINOv2 is known to have richer spatial features):** interesting nuance — feature richness matters; explore further.

### Deliverable
`runs/c3_cross_encoder/probe_siglip_mlp.json`, `probe_dinov2_mlp.json`, plus comparable linear probes.

---

# C-Writeup — Paper Outline

## Title (candidates)

1. **"Causal Token Importance is Unpredictable from Encoder Features: An Information-Theoretic Limit for Pre-LLM Visual Token Compression"**
2. "Why Pre-LLM Visual Token Compression Hits a Floor"
3. "The Encoder Doesn't Know What the LLM Needs"

Recommend #1 (descriptive, claim-forward).

## Target venues, in priority order

1. **TMLR** — best fit. TMLR explicitly accepts mechanism / negative-results papers with rigorous methodology. Acceptance probability with this content: 70-85%.
2. **NeurIPS** — possible. Need to position as "mechanism + implication for the field." Acceptance probability: 30-45%.
3. **ICML** — similar to NeurIPS. Acceptance probability: 25-40%.
4. **ICLR** — possible. Workshop track is also strong fallback.

## Section structure

```
Abstract
  Single paragraph. Lead claim: causal token importance for VLM tasks is unpredictable
  from frozen encoder features but predictable from mid-LLM hidden states; this implies
  pre-LLM compression is fundamentally bounded relative to mid-LLM methods.

1. Introduction
   - The pre-LLM vs mid-LLM compression debate.
   - The implicit assumption: encoder features carry enough info to select useful tokens.
   - Our finding: this assumption is empirically false.
   - Contribution: (a) per-token causal importance probe, (b) cross-encoder verification,
     (c) per-layer LLM analysis showing where the info actually lives, (d) implications
     for designing compression methods.

2. Related Work
   - Pre-LLM compressors (PruMerge, ToMe, EViT, TC-ViTok).
   - Mid-LLM compressors (FastV, SparseVLM).
   - Token-importance / saliency in NLP and vision.

3. Method: Causal Token Importance
   - Definition: per-token causal importance via ablation on a teacher VLM.
   - Probe formulation: predicting per-token importance from per-token features.
   - Reproducibility setup.

4. Encoder-Feature Unpredictability (C1 results)
   - Probe R² < 0 at N=200 and N=1000 with bootstrap CI.
   - Robustness: linear vs MLP probe; per-question-type breakdown (C1.3).
   - Result: unpredictability is not a small-sample or capacity artifact.

5. Mid-LLM Predictability (C2 results)
   - R² rises with LLM layer depth.
   - Onset analysis: layer L* where R² becomes high.
   - Connection to FastV: explains why mid-LLM compression at layer ~2 works.

6. Cross-Encoder Generality (C3 results)
   - Same procedure with SigLIP, DINOv2.
   - Result: unpredictability holds across all three encoders.

7. Implications
   - Pre-LLM compression is bounded by encoder-feature information content.
   - Hybrid pre-LLM + mid-LLM is the natural next direction.
   - Implications for compute-efficient VLM design.

8. Limitations
   - Single VLM family (LLaVA-1.5-7B).
   - Single benchmark dataset (GQA val).
   - Ablation-based importance; counterfactual not deconfounded by token interactions.
   - Probe capacity is bounded; larger probes might find more signal.

9. Conclusion
```

## Headline figures and tables

| Artifact | Source | Purpose |
|---|---|---|
| **Fig. 1** — Predictability curve | C1 (encoders) + C2 (LLM layers) combined | Headline. Shows R² near-zero at encoders, rises with LLM depth. |
| **Fig. 2** — Per-question-type R² | C1.3 | Robustness; shows unpredictability is universal across question types. |
| **Table 1** — Per-encoder D4 | C1, C3 | The numerical headline. |
| **Table 2** — Per-layer R² | C2 | The mechanistic core. |
| **Fig. 3** — Coverage map (re-used) | Phase 1 P1.4 | Supporting evidence: degenerate position bias in trained selector. |
| **Fig. 4** — TPN-vs-trivial benchmark gap (re-used) | ICONIP paper | Supporting evidence: practical consequence of the limit. |

## How to draft

Recommended order:
1. **Sections 3, 4, 5, 6** first — these are methods + results. Pure description.
2. **Section 7 (Implications)** — the field-level argument.
3. **Section 8 (Limitations)** — be aggressive about scoping; reviewers will respect it.
4. **Sections 1, 2** last — informed by what the results actually showed.

Target length: 12-16 pages including refs. TMLR has no hard page limit; NeurIPS / ICML are 9 pages + refs.

---

# Final compute summary

| Phase | Compute | Cumulative |
|---|---|---|
| C1.1 + smoke | ~12 GPU-h | 12 |
| C1.2 + C1.3 | ~2 GPU-h | 14 |
| C2.1 | ~20 GPU-h | 34 |
| C2.2 | ~2 GPU-h | 36 |
| C3.1 (SigLIP) | ~15 GPU-h | 51 |
| C3.2 (DINOv2) | ~15 GPU-h | 66 |
| Robustness checks | ~5 GPU-h | 71 |
| Slack | ~10 GPU-h | 81 |

Total: ~81 GPU-h. Comfortably within 120 GPU-h budget. ~4 weeks of part-time work on 1-2 GPUs.

# Critical do-nots for Track C

- **Do not retrain LoRA** anywhere in this bundle. We are not building a method; we are measuring an information-theoretic property.
- **Do not invent new encoders.** SigLIP-Large and DINOv2-Large are sufficient. Adding more encoders weakens focus.
- **Do not skip the smoke tests.** Especially C2.1 (verifying VIS_START / VIS_END) — getting this wrong invalidates the whole paper.
- **Do not gate Track C on D2.** D2 is not needed.
- **Do not modify the submitted ICONIP paper.** That paper is done.
- **Do not commit large feature dumps to git.** Use cloud backup only.

# Reporting back

After each block (C1, C2, C3) completes, append to `runs/TRACKC_LOG.md`:

```
- [DATE] C1.1: causal_data_1000 saved (10.2 GB)
- [DATE] C1.2: D4 at N=1000 = -0.21 (95% CI [-0.28, -0.14]). Unpredictability robust at scale.
- [DATE] C2.1: midllm_features for L0,2,4,8,16,24,28 saved (16 GB)
- [DATE] C2.2: per-layer R² {L0: -0.21, L2: 0.18, L4: 0.42, L8: 0.61, L16: 0.65, L24: 0.58, L28: 0.51}. Mid-LLM info onset at L4.
- [DATE] C3.1: SigLIP D4 = -0.18 (consistent with CLIP).
- [DATE] C3.2: DINOv2 D4 = -0.12 (slightly higher signal but still below 0.2).
```

Send the contents of `TRACKC_LOG.md` to the human after each block completes, plus any blockers.

# What happens next (after C3 completes)

Human writes the paper. Codex assists with figures and final reproducibility scripts.
