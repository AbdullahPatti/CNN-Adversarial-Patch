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
| `src/run_queue.py` | runs configs in order, one process each, logs to `logs/<run>.log`, skips runs with existing results (restartable) |
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
  stay float. Rounding uses a straight-through estimator so attacks can backprop; Part 4 must check
  gradient masking. INT4 uses 8-bit activations (size depends only on weights).
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
- [ ] **Adaptive attacks on quantized models:** STE/BPDA white-box + transfer from FP32 surrogate (Part 4).
- [ ] **Second setting:** GTSRB (traffic signs, the standard physical-patch scenario) with the same pipeline,
      or a second architecture if time is short.
- [ ] **Attack strength sanity:** patch success vs optimization steps curve, to show attacks converged.
- [ ] Full literature search (above) before the related-work section is written.

### Paper layout (6 pages)
Intro + contributions (0.75) | Related work (0.5) | Setup: models, matched-size rule, attacks, defense (1)
| RQ1 (1) | RQ3 (1) | RQ2 + RQ4 (1) | Discussion, guidance, limitations (0.5) | refs (0.25).
Figures: pipeline overview, accuracy-under-patch vs ratio per family, certified-accuracy bars,
transfer heatmap.

## Later parts (per plan)
- Part 3: universal patch (EOT, random location) and LaVAN; patch sizes 2.4% and 6.25% of image area.
- Part 4: transfer matrix + gradient-masking check on quantized models.
- Part 5: PatchCleanser double masking + certification on the 1,000-image subset.
- Part 6: RQ4 analysis, figures, then the interactive 3D explainer site (built only after Parts 3-5 are done, so it shows real results).

## Open items / notes
- Matched-size definition: fixed in Part 2 (nominal ratio for matching, storage ratio reported alongside).
- Baseline checkpoint is selected by best test accuracy (common CIFAR practice); flagged in `src/engine.py`; every arm uses the same rule.
