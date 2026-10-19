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

## Part 4: Patch transfer (RQ2) (DONE 2026-10-09)
Files:
| File | Purpose |
|---|---|
| `src/transfer.py` | every source model's Part 3 patches pasted on every target model (13 x 13), universal + LaVAN + physical-EOT (5 x 5) |
| `src/report.py` | transfer matrices (U8 hit, L5 untargeted) and a full <-> compressed summary table |
| `src/run_queue.py` | dispatches configs with a `transfer:` block to `src.transfer` |
| `configs/phase4/transfer.yaml` | the 13 models, physical-EOT subset |

Design (decided 2026-10-09):
- **Full matrix, not only full <-> compressed.** Same cost order, and it also gives compressed <-> compressed
  (e.g. does a pruned-model patch hit the dense-small model with the same nonzero params?).
- **Source patch = Part 3 "effective" patch** (for quantized sources the white-box or FP32-surrogate patch,
  whichever Part 3 reported). Same seeded placements as Part 3, so **the diagonal must reproduce Part 3**:
  this is the built-in correctness check.
- **Universal:** mean +/- std over the 10 targeted patches, full 10k test set, accuracy and target hit rate.
- **LaVAN:** per-image patch at its own saved location. Transfer rate = share fooled on the target, counted only
  over images that both models classify correctly when clean and whose patch fooled the source (standard
  definition). Also the unconditional rate, and an **occlusion control**: a random-noise patch at the same
  locations on the same images, so occlusion is not counted as transfer.

Finding while building it: **replaying LaVAN patches on quantized models loses some success.** The INT8 model's
own side-5 targeted patches fool it 87.6% of the time on replay, vs ~100% in Part 3; FP32 models replay at 99.8%.
LaVAN stops at the first success, so patches sit on the decision boundary, and Part 3 saved them as FP16. That
rounding is enough to push them back across a quantization bin. Fixed for future runs (`src/attack.py` now
saves FP32 patches). Seed-42 transfer numbers are "patch as stored"; the diagonal shows the size of the effect.
Minimal-margin patches also transfer worse than high-confidence ones, so LaVAN transfer is a lower bound
(see the journal plan: confidence-margin ablation).

### How to run
```bash
python -m src.run_queue configs/phase4/transfer.yaml
python -m src.report
```

### Results
<!-- transfer-table: regenerate with `python -m src.report` -->
Universal side 8, target hit rate (%)

| source \ target | base | fp16 | int8 | int4 | pr2 | pr4 | pr8 | ds2 | ds4 | ds8 | kd2 | kd4 | kd8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 35.3 | 35.3 | 31.4 | 30.5 | 27.5 | 26.6 | 28.6 | 19.4 | 23.6 | 19.7 | 21.9 | 20.6 | 18.9 |
| fp16 | 35.6 | 35.6 | 31.6 | 30.5 | 27.3 | 26.6 | 28.5 | 19.3 | 23.5 | 20.3 | 22.1 | 19.9 | 19.8 |
| int8 | 35.0 | 35.0 | 33.7 | 31.5 | 27.0 | 26.6 | 28.8 | 18.5 | 22.2 | 19.1 | 21.0 | 18.7 | 17.6 |
| int4 | 34.9 | 34.9 | 33.3 | 37.9 | 26.8 | 27.7 | 29.6 | 20.4 | 25.3 | 21.1 | 22.3 | 21.1 | 19.6 |
| pr2 | 27.1 | 27.1 | 24.2 | 24.5 | 34.7 | 25.6 | 28.6 | 17.7 | 21.0 | 17.5 | 20.2 | 16.6 | 18.1 |
| pr4 | 29.3 | 29.3 | 26.5 | 28.4 | 26.4 | 34.3 | 28.2 | 19.2 | 21.7 | 19.5 | 21.8 | 18.5 | 18.7 |
| pr8 | 28.5 | 28.5 | 24.9 | 26.6 | 24.5 | 25.0 | 33.7 | 16.4 | 19.1 | 18.7 | 18.8 | 16.1 | 16.8 |
| ds2 | 15.0 | 15.0 | 13.9 | 16.0 | 13.8 | 13.8 | 13.9 | 33.1 | 19.3 | 16.9 | 17.4 | 17.2 | 14.9 |
| ds4 | 11.0 | 11.0 | 10.3 | 12.6 | 10.6 | 11.6 | 11.1 | 14.9 | 37.5 | 16.6 | 13.6 | 14.3 | 13.0 |
| ds8 | 12.4 | 12.4 | 11.7 | 14.3 | 11.1 | 11.3 | 11.9 | 15.7 | 20.3 | 39.7 | 13.4 | 15.7 | 13.4 |
| kd2 | 20.8 | 20.8 | 19.3 | 21.0 | 18.2 | 18.3 | 19.6 | 21.6 | 24.5 | 19.5 | 34.7 | 22.4 | 18.5 |
| kd4 | 17.1 | 17.1 | 16.5 | 18.6 | 13.8 | 14.7 | 15.0 | 19.0 | 21.3 | 18.6 | 19.2 | 35.2 | 17.4 |
| kd8 | 13.9 | 13.9 | 12.8 | 15.0 | 12.7 | 11.6 | 12.2 | 15.7 | 17.4 | 16.7 | 15.0 | 15.3 | 38.7 |

LaVAN side 5 untargeted, transfer rate (%)

| source \ target | base | fp16 | int8 | int4 | pr2 | pr4 | pr8 | ds2 | ds4 | ds8 | kd2 | kd4 | kd8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 99.8 | 99.0 | 54.8 | 44.6 | 21.5 | 27.3 | 28.8 | 18.7 | 22.5 | 19.1 | 18.0 | 21.0 | 24.0 |
| fp16 | 99.2 | 99.5 | 52.7 | 44.4 | 19.5 | 26.7 | 26.1 | 20.0 | 20.5 | 18.0 | 17.0 | 18.6 | 25.1 |
| int8 | 90.2 | 90.3 | 89.9 | 52.6 | 24.6 | 26.0 | 29.1 | 18.9 | 23.2 | 19.0 | 16.9 | 20.0 | 26.3 |
| int4 | 33.7 | 33.7 | 29.6 | 91.3 | 17.4 | 25.1 | 22.6 | 17.6 | 19.7 | 15.6 | 15.8 | 19.6 | 24.6 |
| pr2 | 31.5 | 31.5 | 27.1 | 33.4 | 99.5 | 28.4 | 28.6 | 19.9 | 22.7 | 17.1 | 17.6 | 20.1 | 25.6 |
| pr4 | 27.0 | 27.0 | 23.4 | 29.9 | 21.5 | 99.8 | 26.6 | 17.7 | 21.3 | 18.1 | 15.2 | 17.8 | 26.1 |
| pr8 | 29.3 | 29.4 | 26.0 | 30.9 | 21.4 | 26.0 | 99.3 | 19.5 | 20.4 | 18.3 | 14.4 | 18.4 | 22.6 |
| ds2 | 11.3 | 11.3 | 10.2 | 12.9 | 10.1 | 11.5 | 11.5 | 100.0 | 17.5 | 12.6 | 12.9 | 14.3 | 19.3 |
| ds4 | 10.7 | 10.7 | 9.7 | 12.1 | 8.2 | 9.4 | 10.8 | 12.0 | 99.6 | 12.7 | 10.4 | 11.8 | 18.5 |
| ds8 | 9.9 | 9.9 | 9.7 | 11.5 | 9.0 | 10.5 | 10.8 | 12.6 | 14.6 | 99.8 | 9.1 | 11.4 | 17.9 |
| kd2 | 17.1 | 17.2 | 15.4 | 19.2 | 12.8 | 14.3 | 14.5 | 20.4 | 21.2 | 18.9 | 100.0 | 18.9 | 23.9 |
| kd4 | 12.9 | 13.0 | 12.1 | 16.7 | 11.5 | 13.4 | 14.4 | 16.8 | 20.1 | 16.1 | 11.6 | 99.8 | 22.3 |
| kd8 | 10.8 | 10.8 | 10.1 | 11.5 | 9.9 | 10.8 | 10.3 | 10.1 | 13.2 | 10.6 | 9.1 | 9.2 | 99.4 |

Physical-EOT universal side 8, target hit rate (%)

| source \ target | base | int8 | pr4 | ds4 | kd4 |
|---|---|---|---|---|---|
| base | 21.2 | 19.4 | 16.7 | 14.1 | 12.4 |
| int8 | 23.0 | 21.8 | 17.7 | 14.2 | 13.1 |
| pr4 | 23.4 | 21.9 | 24.6 | 17.1 | 14.5 |
| ds4 | 9.4 | 9.1 | 9.3 | 21.9 | 9.5 |
| kd4 | 16.3 | 15.1 | 13.2 | 16.5 | 22.1 |

Full <-> compressed (white-box = diagonal; LaVAN noise = random patch at the same locations)

| Model | U8 hit white-box | U8 hit full->comp | U8 hit comp->full | L5 white-box | L5 full->comp | L5 comp->full | L5 noise (comp) |
|---|---|---|---|---|---|---|---|
| quant_2x_fp16 | 35.6 | 35.3 | 35.6 | 99.5 | 99.0 | 99.2 | 7.1 |
| quant_4x_int8 | 33.7 | 31.4 | 35.0 | 89.9 | 54.8 | 90.2 | 6.4 |
| quant_8x_int4 | 37.9 | 30.5 | 34.9 | 91.3 | 44.6 | 33.7 | 8.3 |
| prune_2x | 34.7 | 27.5 | 27.1 | 99.5 | 21.5 | 31.5 | 6.5 |
| prune_4x | 34.3 | 26.6 | 29.3 | 99.8 | 27.3 | 27.0 | 6.8 |
| prune_8x | 33.7 | 28.6 | 28.5 | 99.3 | 28.8 | 29.3 | 6.7 |
| dense_small_2x | 33.1 | 19.4 | 15.0 | 100.0 | 18.7 | 11.3 | 7.9 |
| dense_small_4x | 37.5 | 23.6 | 11.0 | 99.6 | 22.5 | 10.7 | 9.0 |
| dense_small_8x | 39.7 | 19.7 | 12.4 | 99.8 | 19.1 | 9.9 | 7.4 |
| kd_2x | 34.7 | 21.9 | 20.8 | 100.0 | 18.0 | 17.1 | 7.7 |
| kd_4x | 35.2 | 20.6 | 17.1 | 99.8 | 21.0 | 12.9 | 7.3 |
| kd_8x | 38.7 | 18.9 | 13.9 | 99.4 | 24.0 | 10.8 | 10.4 |

Run 2026-10-09 17:46-18:56 (70 min), no failures. Diagonal = Part 3 numbers (checked: baseline U5 81.1 / 6.7,
U8 61.2 / 35.3). Rows = model the patch was made on, columns = model it is pasted on.

### First reading (single seed; needs seeds 43/44 before these are claims)
- **Quantization keeps the full model's weaknesses.** FP16 is the same model to an attacker (99% LaVAN transfer
  both ways). Baseline LaVAN patches fool INT8 55% and INT4 45% of the time, against a 6-8% noise control, and
  baseline universal patches keep ~85% of their white-box hit rate on INT8 / INT4. Someone holding the public FP32
  weights can attack a quantized deployment almost as well as the deployed model itself.
- **Asymmetry, quantization:** INT8 -> full transfers better than full -> INT8 (LaVAN 90% vs 55%). Patches made on the
  coarser model still work on the smooth one, not the reverse.
- **Pruning sits in the middle** and is roughly symmetric: universal hit 27-29% both ways (~80% of white-box),
  LaVAN 21-31%.
- **Capacity-reduced models (dense-small, KD) are the most isolated**, and asymmetric: full -> small transfers
  (universal 19-24%), small -> full much less (11-21%; LaVAN 10-17%, close to the 7-10% noise control for
  dense-small). Patches made on smaller models carry less of the shared, transferable features.
- **KD students inherit some of the teacher's weaknesses:** kd -> full (14-21%) is higher than dense-small -> full
  (11-15%) at equal size, and the gap shrinks as the student gets smaller. Consistent with distillation
  transferring the teacher's decision features.
- **Physical-EOT patches** show the same structure (dense-small 4x patches transfer worst, 9% hit).
- Takeaway for RQ2 / guidance: compression that preserves the original weights (quantization, then pruning) keeps
  the full model's attack surface; only re-training a smaller network (dense-small, KD) breaks transfer, and
  those models are more vulnerable white-box (Part 3). Analysis item J8 should test whether transfer tracks
  prediction agreement / representation similarity with the baseline.


## Part 5: PatchCleanser certification (RQ3) (DONE 2026-10-17, all 3 seeds)
Files:
| File | Purpose |
|---|---|
| `src/defense/patchcleanser.py` | mask set, two-mask prediction table, certification (Thm. 1), double-masking inference (Alg. 1), `MaskAugment` |
| `src/certify.py` | entry: one model, patch sides 5 and 8, writes `results/pc_<model>.json` (incl. per-image certified flags) |
| `src/engine.py` | `train.mask_aug` turns on mask augmentation in every training loop (baseline, prune fine-tune, QAT, KD) |
| `src/seed_configs.py` | `--defended` writes the defended zoo configs (`configs/seed<S>_def/`) |
| `configs/phase5/*.yaml` | certification of the 13 vanilla seed-42 models |

Setup: k = 6 mask positions per axis, stride ceil((32 - p + 1) / 6). Side 5: 36 masks of 9 px; side 8: 25 masks of
12 px. Mask fill = dataset mean (zero after normalization, as in the authors' code). Coverage of every patch
position is checked exhaustively in `tests/test_smoke.py`. 1,000-image subset; ~3-6 min per model.

**Decision: a defended model zoo (2026-10-09).** Vanilla baseline: certified 10.7% (side 5) / 2.1% (side 8) at
95% clean; the worst single mask already drops accuracy to 80% / 62%. Comparing compression arms on near-zero
certified numbers says nothing, and PatchCleanser's authors also train with masks. So the baseline is retrained
with mask augmentation (each image gets 0, 1 or 2 masks from the union of both mask sets, equally likely), and
every compression arm is rebuilt from it with augmentation kept on in all fine-tuning / QAT / KD.
**Order = defend, then compress**, which is how a deployed model is built (shrink an already-defended model).
Compress-then-defend (fine-tune each vanilla compressed model with masks) is a possible ablation at 4x.
Vanilla certification is still reported as the reference point.

### Queue (started after Part 4, ~66 GPU-h, `logs/master_queue.log`, restartable)
1. defended zoo seed 42 (`configs/seed42_def/phase2`) -> 2. certify vanilla + defended seed 42 ->
3. seed 43 vanilla (models, attacks, transfer, certify) -> 4. seed 43 defended (models, certify) -> 5-6. seed 44 likewise.

2026-10-10: machine shut down unexpectedly at ~01:15 during `dense_small_8x_def` (epoch 95/200; 9 of 13 defended
models done). Restarted 12:49 with `.venv\Scripts\python.exe` (the `python` on PATH has no numpy); the run restarts
from epoch 1 (no mid-run resume). Log continues in `logs/master_queue_2.log`. Three unexpected shutdowns since
2026-10-09, so `engine.fit` now writes a per-epoch resume file (`<checkpoint>.resume.pt`: model, optimizer, scheduler, scaler,
RNG states) and continues from it when the queue is restarted; deleted when training finishes. Covers baseline, dense-small, KD,
prune fine-tune and QAT. Checked in `tests/test_smoke.py` (resumed run = uninterrupted run, bit-exact). Active from the run after
`dense_small_8x_def` (that one started on the old code).

2026-10-10 ~21:16: fourth shutdown, during `dense_small_2x_s43` (epoch 8). Done by then: defended seed-42 zoo,
certification of vanilla + defended seed 42, seed-43 baseline, quantization and pruning. The resume file was unreadable:
`os.replace` without `fsync` let the power cut leave the renamed file with unflushed contents. Fixed in `engine.fit`:
the resume file is fsynced before the rename, and an unreadable one is deleted and the run starts from scratch.
Restarted 2026-10-11 15:53 (`logs/master_queue_4.log`); `dense_small_2x_s43` restarted from epoch 1.

2026-10-11 ~15:55: fifth stop, during `dense_small_2x_s43` (epoch 3). This time the resume file was readable (fsync fix works).
Restarted 2026-10-12 11:17 as a detached process (`logs/master_queue_5.log`); resumed after epoch 3, no work lost.
Remaining: rest of seed 43 (dense-small, KD, attacks, transfer, certify, defended zoo) + all of seed 44, ~60 GPU-h (~2.5 days) if uninterrupted.

2026-10-12 ~21:15: sixth stop (silent, no error in the log), during `universal_prune_8x_s43` (attack, no resume; rerun from
the start, ~13 min lost). Done by then: all seed-43 models, seed-43 universal attacks on baseline / quant / prune 2x, 4x.
Restarted 2026-10-12 23:01 (`logs/master_queue_6.log`, PID 5884). Remaining estimate from measured run times: seed 43 ~17 h
(universal 55 min, LaVAN 3.3 h, transfer 1.2 h, certify 0.8 h, defended zoo 10.8 h), seed 44 ~29 h -> ~46 h, done around
2026-10-14 21:00 if uninterrupted.

2026-10-13 ~08:17: seventh stop (machine rebooted, back up 14:53), during `dense_small_4x_s43_def` (epoch 127/200). Done by
then: all of seed 43 vanilla (models, attacks, transfer, certify) and the seed-43 defended baseline, quant, prune, dense-small 2x.
Restarted 2026-10-13 17:45 (`logs/master_queue_7.log`, PID 21000); resumed after epoch 127 from the resume file, no work lost.
Remaining: seed-43 defended dense-small 4x/8x, KD, certify (~6.5 h) + seed 44 (~29 h) -> ~35 h, done around 2026-10-15 05:00.

2026-10-13 ~19:12: eighth stop, during `dense_small_8x_s43_def` (epoch 151/200). Restarted 19:35 (`logs/master_queue_8.log`,
PID 13252; only the remaining stages seed43_def, seed44, seed44_def are passed); resumed after epoch 151, no work lost.
Remaining ~34 h -> done around 2026-10-15 05:30 if uninterrupted.

2026-10-13 ~20:44: ninth stop (silent), during `kd_2x_s43_def` (epoch 111/200). Restarted 2026-10-14 11:23
(`logs/master_queue_9.log`, PID 19656; stages seed43_def, seed44, seed44_def); resumed after epoch 111, no work lost.
Remaining from measured seed-43 run times: seed-43 defended KD + certify ~3.6 h, seed 44 vanilla ~17 h, seed 44 defended
~10.7 h -> ~31 h, done around 2026-10-15 18:30-19:00 if uninterrupted.

2026-10-14 ~11:27: tenth stop (silent), during `kd_2x_s43_def` (epoch 116/200). Restarted 11:31 as a detached process
(`logs/master_queue_10.log`, PID 9152; stages seed43_def, seed44, seed44_def, 105 configs, finished ones skipped); resumed
after epoch 116, no work lost. Done around 2026-10-15 19:00 if uninterrupted.

2026-10-14 ~11:38: eleventh stop (silent), during `kd_2x_s43_def` (epoch 129/200). Restarted 11:42 with `.venv\Scripts\python.exe`
(`logs/master_queue_11.log`, PID 24260; same 3 stages, 105 configs); resumed after epoch 129, no work lost. Done around
2026-10-15 19:00 if uninterrupted.

2026-10-14 ~17:46: twelfth stop (silent), during `prune_8x_s44` (epoch 24/40); seed-43 defended stage and seed-44 baseline,
quantization, prune 2x/4x had finished. Found and restarted 2026-10-15 01:16 (`logs/master_queue_12.log`, PID 18100; same 3
stages, 105 configs); resumed after epoch 24, no work lost. Remaining (73 runs, from seed-43 run times corrected for resumed
runs): seed 44 vanilla ~14.7 h, seed 44 defended ~10.9 h -> ~25.5 h, done around 2026-10-16 02:30-03:00 if uninterrupted.

2026-10-15 ~06:10: thirteenth stop (silent), during `kd_2x_s44` (epoch 151/200); seed-44 prune 8x and dense-small 2x/4x/8x had
finished. Restarted 13:36 (`logs/master_queue_13.log`, PID 5024; same 3 stages, 105 configs); resumed after epoch 151, no work lost.
Remaining: seed-44 KD (~2.5 h), attacks + transfer + certify (~6.3 h), seed-44 defended zoo + certify (~10.9 h) -> ~20 h,
done around 2026-10-16 09:30-10:00 if uninterrupted.

2026-10-15 ~13:44: fourteenth stop (silent), during `kd_2x_s44` (epoch 165/200). Restarted 16:57 (`logs/master_queue_14.log`,
PID 2844; same 3 stages, 105 configs, passed as an explicit file list since Windows does not expand `*.yaml`); resumed after
epoch 165, no work lost. Remaining ~19.5 h -> done around 2026-10-16 12:30 if uninterrupted.

2026-10-15 ~17:40: fifteenth stop (silent), during `kd_4x_s44` (epoch 58/200); `kd_2x_s44` had finished. Found and restarted
2026-10-16 18:43 (`logs/master_queue_15.log`, PID 17848; same 3 stages, 105 configs as an explicit file list); resumed after
epoch 58, no work lost. Remaining: seed-44 KD 4x/8x (~1 h), attacks + transfer + certify (~6.3 h), seed-44 defended zoo +
certify (~10.9 h) -> ~18 h, done around 2026-10-17 13:00 if uninterrupted.

**2026-10-17 15:34: queue finished** (`logs/master_queue_15.log`, no errors in the log or `.err`). `results/` now holds all
3 seeds: 13 vanilla models, universal + LaVAN attacks, transfer, and certification of the vanilla and defended zoos
(79 result files each for seeds 43 and 44). Tables with mean +/- std over seeds come in J3.

## Journal plan (replaces the ICIC 2026 plan, decided 2026-10-09)
Target (confirmed by the user 2026-10-14): **TMLR / JMLR**, not ICIC 2026. **TMLR** first (rewards rigorous, well-supported empirical studies, no page limit, rolling submission).
Backups: Neurocomputing, Pattern Recognition, Computers & Security (Elsevier). JMLR is a poor fit for a
purely empirical study unless the analysis becomes much deeper. The paper is judged on whether every claim
is supported, so multi-seed statistics and a second setting are required, not optional.

### Contributions (unchanged in order, scope widened)
C1 certified robustness under compression (RQ3, lead) | C2 patch vulnerability across families (RQ1) |
C3 transfer, both directions + compressed <-> compressed (RQ2) | C4 sparsity vs capacity (RQ4) |
C5 *why*: link vulnerability to measurable model properties (below) | practical guidance.

**Compute budget (user decision, revised 2026-10-14): after the current queue (seeds 43/44, ends ~2026-10-15), the
~65 GPU-h follow-up phases are approved** (J5-J9, refused on 2026-10-13, now back in). Anything beyond ~65 GPU-h needs asking.

### Work items, in order
| # | Item | Status | Decision / reason |
|---|---|---|---|
| J1 | Part 4 transfer, seed 42 | done | see Part 4 |
| J2 | **Seeds 43, 44**, full pipeline (13 models -> attacks -> transfer -> certification) | done 2026-10-17 | `src/seed_configs.py`. Final-epoch checkpoints (`train.select: last`), so the test set plays no part in training; seed 42 used best-on-test, measured gap <= 0.13 pp in every run. Attack seed unchanged across training seeds -> paired comparisons. Physical-EOT ablation stays single-seed. |
| J3 | Statistics | **in progress (started 2026-10-19)** | mean +/- std over 3 seeds; paired tests vs baseline (per-seed differences); bootstrap 95% CIs over images for single-run numbers. Claims only where the CI excludes 0. |
| J4 | Part 5 PatchCleanser (RQ3), vanilla + defended zoo, 13 models x 3 seeds | done 2026-10-17 | decided: defended zoo, defend-then-compress (see Part 5). |
| J5 | Second setting: **GTSRB**, 4x arms + baseline, 3 seeds | **approved 2026-10-14, after the queue** | est. ~25 GPU-h. |
| J6 | Higher resolution: **ImageNette (224 px), ResNet-50**, 4x arms, 1 seed | **approved 2026-10-14, after the queue** | est. ~12-15 GPU-h. |
| J7 | Second architecture: MobileNetV2 | **approved 2026-10-14, after the queue** | edge-relevant second architecture. |
| J8 | Analysis for C5 | todo (~1 GPU-h covered by the 2026-10-14 budget) | correlate vulnerability with (a) clean acc, (b) occlusion (noise-patch) acc, (c) input-gradient norm / smoothness, (d) prediction agreement / CKA with the baseline vs transfer rate. Explains *why* dense-small is worse and pruning is not. |
| J9 | Attack completeness | **approved 2026-10-14, after the queue** | convergence curves already logged; STE + BPDA adaptive attacks done. Add untargeted patch and LaVAN margin ablation. |
| J10 | Full literature search | todo | Scholar: adversarial patch + quantization / pruning / distillation / TinyML; papers citing PatchCleanser 2023-2026; certified patch defenses (DRS, PatchGuard, ViP, PatchCURE). |

### Positioning (quick literature check, 2026-10-07; do a full Scholar search before writing)
- Closest prior work: Matachana et al. 2021 (AAAI-W, arXiv 2012.06024): universal *perturbations*
  (not patches) on pruned / quantized models, CIFAR-10 + SVHN, transfer pruned <-> full.
  Vora et al. 2023 (arXiv 2308.08160): Lp attacks on pruned models. Review: arXiv 2311.15782.
- PatchCleanser (Xiang et al., USENIX Sec 2022) was evaluated only on full-size classifiers.
- To search: "adversarial patch" + quantization / pruning / TinyML / edge; papers citing PatchCleanser 2023-2026.

### Rigor items reviewers will check (added to the parts below)
- [x] **Seeds:** 3 seeds (42, 43, 44) for all 13 models (J2), done 2026-10-17.
- [x] **Adaptive attacks on quantized models:** STE white-box + FP32-surrogate BPDA, stronger attack reported (Part 3).
- [ ] **Second setting** (GTSRB / ImageNette / MobileNetV2): approved 2026-10-14, runs after the seed 43/44 queue.
- [x] **Attack strength sanity:** patch success vs optimization steps curve, to show attacks converged
      (logged for every universal patch in Part 3).
- [ ] Full literature search (above) before the related-work section is written.

## Later parts
- Part 6: RQ4 analysis, figures, then the interactive 3D explainer site (built only after Parts 3-5 are done, so it shows real results).

## Open items / notes
- Matched-size definition: fixed in Part 2 (nominal ratio for matching, storage ratio reported alongside).
- Checkpoint selection: seed 42 used best-on-test (final epoch within 0.13 pp in every run, checked 2026-10-09 from the logs); seeds 43+ use the final epoch (`train.select: last`).
