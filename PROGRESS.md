# Progress Log: Compression vs Adversarial Patch Robustness

Scope and schedule come from `Context/project_work_plan.pdf`. This file tracks what is
built, what is run, and what comes next. Update it at the end of every part.

## Research questions (from the plan / lit review)
- **RQ1**: How do pruning, quantization, and distillation (matched ~4x size reduction, plus 2x/8x) change
  vulnerability to universal (Brown et al.) and per-image (LaVAN) patches on CIFAR-10?
- **RQ2**: Do patches transfer between the full model and its compressed versions (both directions)?
- **RQ3**: How much PatchCleanser certified accuracy is retained once the base classifier is compressed?
- **RQ4**: Are pruning effects due to sparsity or to capacity loss (pruned vs dense-small at equal nonzero params)?

## Design decisions (fixed)
- Images are in `[0, 1]`; normalization is a layer inside every model. Patch attacks and
  PatchCleanser masks act on raw pixels, so no model-specific preprocessing leaks into them.
- Fixed 1,000-image test subset (`src/data.py: eval_subset_indices`, seed 0) for LaVAN and certification.
- Every run writes `results/<run_name>.json` with the schema in `src/utils.py: save_result`.
- Experiments are driven by one YAML per run in `configs/`.

## Part 1: Foundation + baseline (DONE, run 2026-10-07)
Files:
| File | Purpose |
|---|---|
| `src/models/resnet.py` | CIFAR ResNet-18 (3x3 stem, no max-pool), normalization inside |
| `src/data.py` | CIFAR-10 loaders, fixed 1,000-image eval subset |
| `src/eval/metrics.py` | clean accuracy, param / nonzero count, FP32 size (MB), latency per image |
| `src/utils.py` | YAML config, seeding, results JSON writer |
| `src/train.py` | baseline training (SGD+Nesterov, cosine LR, 200 epochs, AMP) |
| `configs/baseline.yaml` | baseline run config |
| `tests/test_smoke.py` | CPU check: output shape, param count, subset determinism, latency |

### How to run (on the GPU machine)
```bash
pip install -r requirements.txt
python -m tests.test_smoke                          # ~seconds, CPU, no download
python -m src.train --config configs/baseline.yaml  # downloads CIFAR-10 to ./data
```
Outputs: `checkpoints/baseline_resnet18.pt`, `results/baseline_resnet18.json`.
Target: clean accuracy around 94% (this recipe usually lands at ~95%).

### Results
| Run | Clean acc | Params | Size (MB, FP32) | Latency (ms/img) | GPU |
|---|---|---|---|---|---|
| baseline_resnet18 | 95.35% | 11,173,962 | 42.66 | 1.48 | RTX A4000 |

## Part 2: Compression arms (DONE 2026-10-08)
Twelve runs, three per arm at nominal 2x / 4x / 8x, all from the Part 1 baseline (or, for the
students and controls, trained with the baseline recipe). All seeds 42.

Files:
| File | Purpose |
|---|---|
| `src/engine.py` | shared training loop (`fit`, best-on-test checkpoint) and final metrics; used by every arm |
| `src/compress/prune.py` | global unstructured L1 magnitude pruning of all conv/linear weights, one-shot, then 40-epoch fine-tune (lr 0.01, cosine) with masks held fixed |
| `src/compress/quantize.py` | BN folding + simulated (fake-quant) quantization; FP16 cast, INT8 PTQ, INT4 QAT (15 epochs, lr 0.005, FP32 math) |
| `src/compress/distill.py` | Hinton KD, T = 4, alpha = 0.9, teacher = baseline, student trained 200 epochs from scratch |
| `src/models/resnet.py` | `width` argument: 45 / 32 / 23 give 2.02x / 3.99x / 7.72x fewer params |
| `src/models/load.py` | `load_model(cfg, device)` rebuilds any trained arm from its config (for Parts 3-5) |
| `src/eval/metrics.py` | `compressed_size`: storage size + ratio vs the baseline (the matched-size metric, below) |
| `src/run_queue.py` | runs configs in order, one process each, logs to `logs/<phase>/<run>.log`, skips runs with existing results (restartable) |
| `src/report.py` | prints the results table below from `results/*.json` |
| `configs/phase2/01..12_*.yaml` | one config per run |

Runs (nominal ratio -> setting):
| Arm | 2x | 4x | 8x |
|---|---|---|---|
| Quantization | FP16 weights + activations | W8A8 PTQ (20 calibration batches) | W4A8 QAT |
| Pruning | 50% sparsity | 75% sparsity | 87.5% sparsity |
| Dense-small (RQ4 control) | width 45 | width 32 | width 23 |
| Distillation | width 45 student | width 32 student | width 23 student |

### Decisions made in Part 2
- **Matched size (open item, now fixed):** arms are matched on the *nominal* ratio (nonzero params for
  pruning, bit-width for quantization, param count for students). Every result also reports the
  *storage* size (`compressed_size`): conv/linear weights at their bit-width, all other parameters at
  32 bits, buffers not counted; pruned models store nonzero weights plus a 1-bit-per-weight mask.
  With the mask, pruning's storage ratio is below its nominal one (87.5% sparsity -> 6.37x, not 8x).
- **Quantization is simulated.** BN is folded into convs, weights are symmetric per-channel,
  activations (inputs of every conv/linear) asymmetric per-tensor with EMA min/max ranges; residual adds
  stay float. Rounding uses a straight-through estimator so attacks can backprop; Part 3 checks
  gradient masking (BPDA with the FP32 surrogate). INT4 uses 8-bit activations (size depends only on weights).
- **FP16 cast fixed 2026-10-08 (`HalfSTE`).** The plain `x.half().float()` also cast the *gradient* to FP16,
  where small values underflow: 12.75% of input-pixel gradients were exactly 0 with a batch-mean loss.
  Forward pass is unchanged (same outputs and accuracy, no retraining); backward is now FP32 identity.
- **Latency is only meaningful for the dense models.** Unstructured sparsity runs as dense in PyTorch,
  and fake-quant adds overhead, so pruned and quantized latencies are not deployment numbers.

### How to run
```bash
python -m src.run_queue configs/phase2/*.yaml   # ~7.5 h on the RTX A4000; restartable
python -m src.report                            # results table
```

### Sanity checks (2026-10-07, before launch)
BN folding leaves accuracy unchanged (95.35%). PTQ without training: W8A8 95.27%, W4A8 94.78%;
60 QAT steps lift W4A8 to 94.95%. One-shot 87.5% pruning drops to 22.5% before fine-tuning.
A quantized checkpoint reloads through `load_model` with identical accuracy.

### Results
<!-- results-table: regenerate with `python -m src.report` -->
| Run | Arm | Nominal | Storage ratio | Clean acc | Nonzero params | Size (MB) | Latency (ms/img) |
|---|---|---|---|---|---|---|---|
| baseline_resnet18 | none | 1x | 1.00x | 95.35% | 11,173,962 | 42.63 | 1.48 |
| quant_2x_fp16 | quant | 2x | 2.00x | 95.35% | 11,169,162 | 21.31 | 2.13 |
| quant_4x_int8 | quant | 4x | 4.00x | 95.30% | 11,169,162 | 10.67 | 15.28 |
| quant_8x_int4 | quant | 8x | 7.98x | 94.98% | 11,169,162 | 5.34 | 15.65 |
| prune_2x | prune | 2x | 1.88x | 95.42% | 5,591,786 | 22.66 | 3.60 |
| prune_4x | prune | 4x | 3.55x | 95.43% | 2,800,698 | 12.01 | 3.63 |
| prune_8x | prune | 8x | 6.37x | 95.49% | 1,405,154 | 6.69 | 3.92 |
| dense_small_2x | dense_small | 2x | 2.02x | 95.22% | 5,527,675 | 21.09 | 3.94 |
| dense_small_4x | dense_small | 4x | 3.99x | 94.87% | 2,797,610 | 10.67 | 3.52 |
| dense_small_8x | dense_small | 8x | 7.72x | 94.00% | 1,446,917 | 5.52 | 1.59 |
| kd_2x | distill | 2x | 2.02x | 95.63% | 5,527,675 | 21.09 | 1.46 |
| kd_4x | distill | 4x | 3.99x | 95.02% | 2,797,610 | 10.67 | 1.42 |
| kd_8x | distill | 8x | 7.72x | 94.24% | 1,446,917 | 5.52 | 1.43 |

_All 12 runs done 2026-10-08 (KD finished 05:18; first KD 2x attempt was killed at epoch 185 and rerun)._

## Review of external critique (2026-10-08)
Five suggestions were checked against the code and Part 2 results before Part 3.
| # | Suggestion | Verdict | Action |
|---|---|---|---|
| 1 | BPDA for quantized models | Partly valid. Rounding already uses STE (= BPDA with identity): INT8 input gradients have cosine 0.97 with FP32, no zero pixels, <0.01% clamp-saturated activations. But FP16 had gradient underflow (fixed above) and INT4 gradients diverge (cosine 0.70). | Part 3 attacks every quantized model twice (STE white-box + FP32-surrogate BPDA) and reports the stronger attack. |
| 2 | Replace unstructured with structured pruning | Not adopted. Channel pruning gives a narrower dense model, which is what dense-small already is, so RQ4 (sparsity vs capacity) would disappear. Unstructured also matches the closest prior work. | Keep unstructured. Structured L1-filter pruning is an optional extra arm (deployment relevance). |
| 3 | Clean accuracy under PatchCleanser masks | Valid. Also: PatchCleanser's authors fine-tune the base model with mask augmentation; ours are not. | Part 5: add masked clean accuracy; decide on mask fine-tuning before certification. |
| 4 | Rotation / brightness in EOT | Partly valid. Extra transforms make a patch more physically robust but *weaker* digitally, so location-only EOT is the stronger digital attack (not the weaker). | Location-only stays the main attack; physical EOT (rotation +/-20 deg, brightness and contrast 0.8-1.2) runs as an ablation. |
| 5 | Add WRN-28-10 | Weak. The width-helps-robustness result is for adversarially trained models; ours are standard trained. ~40+ GPU-h for 13 models. | Second setting stays as planned (GTSRB, or MobileNetV2 as an edge-relevant second architecture). |

## Part 3: Patch attacks (RQ1) (DONE 2026-10-08)
Files:
| File | Purpose |
|---|---|
| `src/attacks/patch.py` | patch placement, physical EOT transform, universal patch training / evaluation, LaVAN |
| `src/attack.py` | entry: one attack x one model x both patch sizes; writes `results/<run>.json` and `patches/<run>.pt` |
| `src/compress/quantize.py` | `HalfSTE` (FP16 gradient fix), `float_shadow` (FP32 surrogate of a quantized model) |
| `src/report.py` | adds the Part 3 table and the physical-EOT ablation table |
| `configs/phase3/*.yaml` | one config per attack x model |

Setup:
- Patch side 5 px (2.44% of the image) and 8 px (6.25%); pasted, not blended; random location.
- **Universal (Brown):** targeted, one patch per target class (all 10, so no target is cherry-picked),
  Adam on the patch, EOT over location on the training set. Evaluated on the full 10k test set at fixed
  seeded locations (same placements for every model). Reports accuracy with patch, target hit rate
  (images not of the target class), mean +/- std over targets, plus a random-noise patch control
  (occlusion alone). Convergence curve: hit rate on the 1,000-image subset every 250 steps.
- **LaVAN:** per-image patch on the fixed 1,000-image subset, sign-gradient steps, random location,
  restarts at a new location for images not yet fooled. Untargeted (robust accuracy) and targeted
  (random wrong class per image, same for every model).
- **Quantized models:** both attacks also run with FP32-surrogate gradients (forward on the quantized
  model, backward on its float copy). "Effective" = worse of the two, per target (universal) or per image (LaVAN).
- **Physical EOT ablation:** universal, side 8, baseline + the four 4x models.
- Patches, locations and targets are saved for the Part 4 transfer matrix.
- "Effective" is the single stronger attack (white-box or surrogate), chosen per patch size / mode, not a
  per-image union: a union would give quantized models twice the attack budget (in a dry run, the union
  raised INT4 LaVAN success from 68% to 77% by budget alone).

Settings (tuned on the baseline, side 5, 2026-10-08):
| Attack | Setting | Tuning evidence |
|---|---|---|
| Universal | Adam lr 0.02, 1,000 steps, batch 128, all 10 targets | lr 0.005 / 0.02 / 0.05 / 0.2 all plateau at a 17% hit rate by step 250 |
| LaVAN | sign-gradient step 0.05, 100 steps x 10 restarts (new location each) | 1 location: 200 or 500 steps both ~50%; 5 locations x 100 steps: 82%. Location matters, steps don't |
| Physical EOT | as universal, side 8, rotation +/-20 deg, brightness / contrast 0.8-1.2 | ablation only |

Dry-run notes: on INT4, STE white-box was slightly *stronger* than the FP32 surrogate (universal side 8:
60.0% vs 59.5% hit; LaVAN side 5: 69% vs 64%), so INT4's gradients are not masked.

Runs: 31 configs in `configs/phase3/` (13 universal, 13 LaVAN, 5 physical-EOT ablation).

### How to run
```bash
python -m src.run_queue configs/phase3/*.yaml   # ~9-10 h on the RTX A4000; restartable
python -m src.report                            # Part 2 table + Part 3 table
```

### Status
All 31 runs done 2026-10-08: queue started 13:13, finished 20:08 (~7 h), no failures
(`logs/phase3/queue_phase3.log`). Patches, locations and targets saved in `patches/` for Part 4.

### Results
Numbers are % on CIFAR-10. U = universal (mean over the 10 targets, full 10k test set), L = LaVAN
(1,000-image subset). "acc" = accuracy with the patch; "hit" = target hit rate on images not of the target class;
Noise8 = random-noise 8 px patch (occlusion only). Quantized models use the effective (stronger) attack.

<!-- attack-table: regenerate with `python -m src.report` -->
| Model | U5 acc | U5 hit | U8 acc | U8 hit | Noise8 acc | L5 robust acc | L8 robust acc | L5 tgt hit | L8 tgt hit |
|---|---|---|---|---|---|---|---|---|---|
| baseline_resnet18 | 81.1 | 6.7 | 61.2 | 35.3 | 87.8 | 8.6 | 0.0 | 55.0 | 96.2 |
| quant_2x_fp16 | 81.0 | 6.5 | 61.1 | 35.6 | 87.8 | 9.0 | 0.0 | 55.5 | 96.5 |
| quant_4x_int8 | 84.1 | 7.1 | 63.3 | 33.7 | 87.8 | 11.2 | 0.2 | 54.2 | 95.6 |
| quant_8x_int4 | 81.6 | 7.7 | 60.3 | 37.9 | 86.6 | 8.6 | 0.1 | 55.7 | 95.9 |
| prune_2x | 84.1 | 6.1 | 63.3 | 34.7 | 87.9 | 10.7 | 0.2 | 54.4 | 94.5 |
| prune_4x | 83.3 | 7.0 | 63.2 | 34.3 | 87.6 | 8.4 | 0.2 | 58.0 | 96.6 |
| prune_8x | 84.3 | 4.6 | 63.8 | 33.7 | 87.8 | 10.5 | 0.2 | 55.1 | 95.7 |
| dense_small_2x | 78.9 | 8.1 | 60.1 | 33.1 | 87.7 | 7.7 | 0.0 | 59.1 | 96.5 |
| dense_small_4x | 75.9 | 10.4 | 56.8 | 37.5 | 85.5 | 6.5 | 0.0 | 62.9 | 98.4 |
| dense_small_8x | 75.8 | 8.8 | 52.9 | 39.7 | 85.0 | 5.4 | 0.0 | 65.8 | 98.4 |
| kd_2x | 80.4 | 7.6 | 61.4 | 34.7 | 87.8 | 10.6 | 0.0 | 52.5 | 95.5 |
| kd_4x | 81.2 | 7.7 | 61.5 | 35.2 | 86.5 | 10.6 | 0.0 | 54.6 | 95.8 |
| kd_8x | 70.7 | 7.2 | 56.9 | 38.7 | 84.5 | 6.1 | 0.0 | 55.0 | 95.6 |

Physical-EOT ablation (universal, side 8):
| Model | Location-only U8 acc | Location-only U8 hit | Physical-EOT U8 acc | Physical-EOT U8 hit |
|---|---|---|---|---|
| baseline_resnet18 | 61.2 | 35.3 | 71.6 | 21.2 |
| quant_4x_int8 | 63.3 | 33.7 | 72.8 | 21.8 |
| prune_4x | 63.2 | 34.3 | 70.8 | 24.6 |
| dense_small_4x | 56.8 | 37.5 | 69.1 | 21.9 |
| kd_4x | 61.5 | 35.2 | 70.1 | 22.1 |

### First reading (single seed; differences of 2-3 points need the extra seeds before they are claims)
- **Pruning does not hurt, dense-small does (RQ4 signal).** Pruned models match or slightly beat the baseline
  at every ratio (U8 acc 63-64 vs 61.2); dense-small models at the same nonzero-param count get steadily worse
  (U8 acc 60.1 -> 56.8 -> 52.9). So it is the lost capacity, not sparsity, that costs robustness.
- **Quantization is roughly neutral.** FP16 = baseline, INT8 slightly better, INT4 about equal. The FP32-surrogate
  attack did not beat STE white-box enough to change the picture (no sign of gradient masking).
- **Distillation holds at 2x/4x, drops at 8x** (U5 acc 70.7 vs 81.1). Part of that is occlusion: KD 8x also has the
  lowest noise-patch accuracy (84.5).
- **LaVAN side 8 breaks every model** (robust acc ~0%), so only side 5 separates models for per-image patches.
- **Physical EOT is weaker digitally**, as expected (hit 21-25% vs 33-38% location-only); all 4x models within ~3 points.
- **U5 hit rate (6.7% on the baseline) is lower than the ~17% seen in tuning.** Not a bug: tuning used target 0,
  which reaches 17.1% here too; the table is the mean over all 10 targets (std 6.8), and some targets barely move.

## Paper plan: IEEE ICIC 2026 (UMT Lahore)
Target: 6-page IEEE paper. Check deadline and page limit at https://icic.umt.edu.pk/Call-For-Papers.aspx.

### Positioning (quick literature check, 2026-10-07; do a full Scholar search before writing)
- Closest prior work: Matachana et al. 2021 (AAAI-W, arXiv 2012.06024): universal *perturbations*
  (not patches) on pruned / quantized models, CIFAR-10 + SVHN, transfer pruned <-> full.
  Vora et al. 2023 (arXiv 2308.08160): Lp attacks on pruned models. Review: arXiv 2311.15782.
- PatchCleanser (Xiang et al., USENIX Sec 2022) was evaluated only on full-size classifiers.
- To search: "adversarial patch" + quantization / pruning / TinyML / edge; papers citing PatchCleanser 2023-2026.

### Contributions (in paper order)
1. **C1, certified robustness under compression (RQ3, lead result):** first measurement of how much
   PatchCleanser certified accuracy survives pruning, quantization and distillation at matched 2x/4x/8x.
2. **C2, patch vulnerability across compression families (RQ1):** universal (Brown) and per-image (LaVAN)
   patches, 2.4% and 6.25% of image area, three families at matched ratios.
3. **C3, patch transfer between full and compressed models, both directions (RQ2)**, with a
   gradient-masking check so quantized "robustness" is not an artifact of rounding.
4. **C4, sparsity vs capacity (RQ4):** pruned vs dense-small at equal nonzero params. Clean-accuracy gap
   alone is known (Zhu & Gupta 2017); the contribution is the robustness gap.
5. Practical guidance: which compression to pick for an edge device that needs patch robustness.

### Rigor items reviewers will check (added to the parts below)
- [ ] **Seeds:** 3 seeds (42, 43, 44) for baseline and every 4x run at minimum; report mean +/- std.
- [ ] **Adaptive attacks on quantized models:** STE white-box + FP32-surrogate BPDA, stronger attack reported (Part 3).
- [ ] **Second setting:** GTSRB (traffic signs, the standard physical-patch scenario) with the same pipeline,
      or a second architecture if time is short.
- [ ] **Attack strength sanity:** patch success vs optimization steps curve, to show attacks converged
      (logged for every universal patch in Part 3).
- [ ] Full literature search (above) before the related-work section is written.

### Paper layout (6 pages)
Intro + contributions (0.75) | Related work (0.5) | Setup: models, matched-size rule, attacks, defense (1)
| RQ1 (1) | RQ3 (1) | RQ2 + RQ4 (1) | Discussion, guidance, limitations (0.5) | refs (0.25).
Figures: pipeline overview, accuracy-under-patch vs ratio per family, certified-accuracy bars,
transfer heatmap.

## Later parts (per plan)
- Part 4: transfer matrix (patches saved in Part 3), both directions full <-> compressed.
- Part 5: PatchCleanser double masking + certification on the 1,000-image subset. Also report clean accuracy
  under the masks alone (no patch) to separate "the model can't handle occlusion" from "the defense fails".
  Decide first: fine-tune each model with mask augmentation (as PatchCleanser does) or report vanilla models.
- Part 6: RQ4 analysis, figures, then the interactive 3D explainer site (built only after Parts 3-5 are done, so it shows real results).

## Open items / notes
- Matched-size definition: fixed in Part 2 (nominal ratio for matching, storage ratio reported alongside).
- Baseline checkpoint is selected by best test accuracy (common CIFAR practice); flagged in `src/engine.py`; every arm uses the same rule.
