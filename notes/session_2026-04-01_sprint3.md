# Session 2026-04-01 — Sprint 3: T006–T013
Tool: Claude Code Sonnet 4.6

## Goal
Implement T006–T013 per TODO.md.

## What Was Found
T007–T013 were already implemented (pre-existing stubs that got fleshed out before the "done till T005" commit). Only T006 had real gaps:
1. `--max-steps` CLI argument missing from `parse_args()`
2. No Phase 0 → Phase 1 sequential training logic

## Changes Made

### scripts/train.py (T006)
- Added `--max-steps` (int, default None) — overrides `config["training"]["max_steps"]`; confirmed working via dry-run print output
- Added `--run-both-phases` flag — when config is phase 0, auto-runs Phase 1 using `phase1.yaml` from same config directory
- Extracted `build_loaders(data_config, batch_size, num_workers, args)` helper
- Extracted `run_phase(model, config_dict, train_loader, val_loader, device, resume)` helper
- Phase 1 resumes from `phase0_dir / "best.ckpt"` with graceful fallback warning
- `--max-steps` override propagated to Phase 1 dict as well (for dry-run consistency)

### src/tc_vitok/evaluation/collapse_detection.py (T012 fix)
- Added `assert num_tokens == h * w` before `masks.view(batch_size, h, w)` to catch wrong-shaped inputs

### src/tc_vitok/data/caching.py (T007 fix)
- Added `UserWarning` when `CachedFeatureDataset(use_images=True, image_dataset=None)` — prevents silent random-noise training

## Verification
- All 7 core tests pass (`scripts/test_core.py`)
- Dry-run `python scripts/train.py --config configs/phase0.yaml --max-steps 1 --stub --dummy-data --no-wandb` confirmed:
  - Config loaded ✅
  - Phase 0 detected ✅
  - "max_steps overridden to 1" printed ✅
  - Failed on CLIP model download (PyTorch 2.5.1 vs CVE-2025-32434 requirement ≥2.6) — expected blocker

## Reviewer Result
PASS WITH WARNINGS → two issues fixed before close:
1. Shape assertion in `detect_spatial_collapse` (num_tokens check)
2. UserWarning in `CachedFeatureDataset` for silent data corruption case

## Commit
`f6844c9` — feat: T006 + T012 fixes — train CLI complete, collapse detection hardened

## Notes
- `src/tc_vitok/data/` is gitignored (`.gitignore` has `data/` — too broad); caching.py fix lives locally only
- `TODO.md` also gitignored; lives locally
- wandb + omegaconf installed in torch-m1 during this session

## Next Session Priorities
1. E001: dry-run Phase 0 (1 step) — blocked on PyTorch upgrade or workaround for CVE-2025-32434
2. Download LLaVA-1.5-7B (~13.5GB) and run full test_pipeline.py
3. Download VQAv2 train/val splits and begin real training
