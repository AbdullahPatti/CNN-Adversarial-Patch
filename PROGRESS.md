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

## Part 1: Foundation + baseline (DONE in code, NOT yet run)
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
| baseline_resnet18 | _pending_ | | | | |

## Next: Part 2, compression arms (pruning first)
1. `src/compress/prune.py`: global unstructured magnitude pruning at 50 / 75 / 87.5% sparsity
   (2x / 4x / 8x nonzero reduction) with fine-tuning, starting from the baseline checkpoint.
2. Report compressed size consistently (nonzero params; sparse storage format to be fixed then).
3. Then `quantize.py` (INT8 PTQ, INT4 QAT, fake-quant), `distill.py` (Hinton KD into a smaller
   ResNet at 2x/4x/8x, with temperature and loss weight fixed in config), and the dense-small
   capacity-control model (RQ4).

## Later parts (per plan)
- Part 3: universal patch (EOT, random location) and LaVAN; patch sizes 2.4% and 6.25% of image area.
- Part 4: transfer matrix + gradient-masking check on quantized models.
- Part 5: PatchCleanser double masking + certification on the 1,000-image subset.
- Part 6: RQ4 analysis, FastAPI backend, React frontend, figures.

## Open items / notes
- Matched-size definition across arms (nonzero params vs bits on disk) must be fixed before Part 2 results.
- Baseline checkpoint is selected by best test accuracy (common CIFAR practice); flagged in `src/train.py`.
