"""
generate_visualizations.py
Research-grade visualization pipeline — Phase 1 through Phase 3.

File → Phase mapping
--------------------
results/baseline_resnet18.json              Phase 1   clean_acc, params, size_mb_fp32, latency
results/{quant,prune,dense_small,kd}_*.json Phase 2   clean_acc, nonzero_params, size_mb_compressed, ratio_vs_baseline, latency
results/universal_*.json                    Phase 3   side_5/8: patched_acc (mean±std), target_success, noise_patch_acc, convergence
results/lavan_*.json                        Phase 3   side_5/8 untargeted robust_acc + targeted success_on_correct
results/universal_physical_*.json           Phase 3   physical-EOT ablation: side_8 patched_acc/target_success
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Matplotlib style ──────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "figure.dpi":              150,
    "savefig.dpi":             300,
    "font.family":             "sans-serif",
    "font.size":               9,
    "axes.titlesize":          10,
    "axes.labelsize":          9,
    "xtick.labelsize":         7.5,
    "ytick.labelsize":         8,
    "legend.fontsize":         7.5,
    "legend.title_fontsize":   8,
    "axes.grid":               True,
    "grid.alpha":              0.3,
    "grid.linestyle":          "--",
    "axes.spines.top":         False,
    "axes.spines.right":       False,
})

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass

# ── Directory constants ───────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
VIZ_DIR     = Path("visualization")
PHASE_DIRS  = {
    "phase_1": VIZ_DIR / "phase_1",
    "phase_2": VIZ_DIR / "phase_2",
    "phase_3": VIZ_DIR / "phase_3",
    "summary": VIZ_DIR / "summary",
}

# ── Visual identity ───────────────────────────────────────────────────────────

ARM_COLOR = {
    "baseline":    "#555555",
    "quant":       "#1f77b4",
    "prune":       "#ff7f0e",
    "dense_small": "#2ca02c",
    "distill":     "#9467bd",
}
ARM_LABEL = {
    "baseline":    "Baseline",
    "quant":       "Quantization",
    "prune":       "Pruning",
    "dense_small": "Dense-small",
    "distill":     "Distillation",
}
ARM_LEGEND = [
    mpatches.Patch(color=ARM_COLOR[a], label=ARM_LABEL[a])
    for a in ["baseline", "quant", "prune", "dense_small", "distill"]
]

MODEL_ORDER = [
    "baseline_resnet18",
    "quant_2x_fp16",  "prune_2x",  "dense_small_2x",  "kd_2x",
    "quant_4x_int8",  "prune_4x",  "dense_small_4x",  "kd_4x",
    "quant_8x_int4",  "prune_8x",  "dense_small_8x",  "kd_8x",
]
MODEL_LABEL = {
    "baseline_resnet18": "Baseline",
    "quant_2x_fp16":     "Q-FP16\n(2×)",
    "prune_2x":          "Prune\n(2×)",
    "dense_small_2x":    "Dense\n(2×)",
    "kd_2x":             "KD\n(2×)",
    "quant_4x_int8":     "Q-INT8\n(4×)",
    "prune_4x":          "Prune\n(4×)",
    "dense_small_4x":    "Dense\n(4×)",
    "kd_4x":             "KD\n(4×)",
    "quant_8x_int4":     "Q-INT4\n(8×)",
    "prune_8x":          "Prune\n(8×)",
    "dense_small_8x":    "Dense\n(8×)",
    "kd_8x":             "KD\n(8×)",
}
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ── Utilities ─────────────────────────────────────────────────────────────────

def setup_dirs():
    for d in PHASE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    print("[setup] Output directories ready.")


def load_all_results():
    data = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data[f.stem] = json.load(fh)
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    return data


def save_fig(fig, phase_key, stem):
    d = PHASE_DIRS[phase_key]
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    print(f"  -> {d}/{stem}.pdf/png")
    plt.close(fig)


def arm_of(run):
    if run == "baseline_resnet18":   return "baseline"
    if run.startswith("quant"):      return "quant"
    if run.startswith("prune"):      return "prune"
    if run.startswith("dense"):      return "dense_small"
    if run.startswith("kd"):         return "distill"
    return "other"


def ratio_of(run):
    for k, v in [("2x", 2), ("4x", 4), ("8x", 8)]:
        if k in run:
            return v
    return 1


def bar_labels(ax, bars, vals, fmt="%.1f%%"):
    labels = [fmt % v if not np.isnan(v) else "" for v in vals]
    ax.bar_label(bars, labels=labels, fontsize=6, padding=2)


def add_group_seps(ax, n):
    for xv in [0.5, 4.5, 8.5]:
        if xv < n - 0.5:
            ax.axvline(xv, color="#cccccc", linestyle=":", linewidth=1.0)


# ── Phase 2 data table ────────────────────────────────────────────────────────

def phase2_table(data):
    rows = []
    for run in MODEL_ORDER:
        d = data.get(run)
        if d is None or d.get("attack") is not None:
            continue
        m = d["metrics"]
        rows.append({
            "run":     run,
            "arm":     arm_of(run),
            "ratio":   ratio_of(run),
            "acc":     m.get("clean_acc", np.nan),
            "nzp":     m.get("nonzero_params", m.get("params", np.nan)),
            "size_mb": m.get("size_mb_compressed", m.get("size_mb_fp32", np.nan)),
            "lat":     m.get("latency_ms_bs1", np.nan),
            "stor":    m.get("ratio_vs_baseline", 1.0),
        })
    return rows


# ── Phase 3 accessors ─────────────────────────────────────────────────────────

def univ(data, run):
    return data.get(f"universal_{run}", {})

def lavan(data, run):
    return data.get(f"lavan_{run}", {})

def phys(data, run):
    return data.get(f"universal_physical_{run}", {})

def univ_m(d, side):
    s = d.get("metrics", {}).get(f"side_{side}", {})
    return {
        "acc":     s.get("effective_patched_acc",    {}).get("mean", np.nan),
        "acc_std": s.get("effective_patched_acc",    {}).get("std",  np.nan),
        "hit":     s.get("effective_target_success", {}).get("mean", np.nan),
        "hit_std": s.get("effective_target_success", {}).get("std",  np.nan),
        "noise":   s.get("noise_patch_acc", np.nan),
    }

def lavan_m(d, side, mode):
    s = d.get("metrics", {}).get(f"side_{side}_{mode}", {})
    e = s.get("effective", {})
    return e.get("robust_acc" if mode == "untargeted" else "success_on_correct", np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_1(data):
    print("\n[Phase 1]")
    b = data.get("baseline_resnet18", {})
    m = b.get("metrics", {})

    metrics = [
        ("Clean Accuracy (%)",   m.get("clean_acc", 0) * 100),
        ("Parameters (M)",       m.get("params", 0) / 1e6),
        ("Model Size (MB, FP32)", m.get("size_mb_fp32", 0)),
        ("Latency (ms / image)", m.get("latency_ms_bs1", 0)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (label, val) in zip(axes, metrics):
        b_ = ax.bar(["ResNet-18"], [val], color=ARM_COLOR["baseline"], width=0.45, zorder=3)
        ax.bar_label(b_, labels=[f"{val:.2f}"], fontsize=9, padding=3)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylim(0, val * 1.4)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Phase 1 — Baseline ResNet-18, CIFAR-10", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "phase_1", "01_baseline_overview")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_2(data):
    print("\n[Phase 2]")
    rows = phase2_table(data)
    if not rows:
        print("  [warn] No Phase 2 data."); return

    base_acc = next((r["acc"] for r in rows if r["run"] == "baseline_resnet18"), np.nan)
    arms = ["quant", "prune", "dense_small", "distill"]
    arm_rows = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: r["ratio"])
                for a in arms}

    x = np.arange(len(rows))
    xlabels = [MODEL_LABEL.get(r["run"], r["run"]) for r in rows]
    colors  = [ARM_COLOR[r["arm"]] for r in rows]

    # ── P2-01: clean acc vs ratio (line) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in arms:
        pts = arm_rows[arm]
        if not pts: continue
        ax.plot([r["ratio"] for r in pts], [r["acc"] * 100 for r in pts],
                "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm], lw=1.8, ms=6)
    ax.axhline(base_acc * 100, color=ARM_COLOR["baseline"], ls="--", lw=1.5, label="Baseline")
    ax.set_xticks([2, 4, 8]); ax.set_xticklabels(["2×", "4×", "8×"])
    ax.set_xlabel("Nominal Compression Ratio"); ax.set_ylabel("Clean Accuracy (%)")
    ax.set_title("Clean Accuracy vs. Compression Ratio", fontweight="bold")
    ax.set_ylim(92, 97); ax.legend(framealpha=0.9, loc="lower left", fontsize=7.5)
    fig.tight_layout(); save_fig(fig, "phase_2", "01_clean_acc_vs_ratio")

    # ── P2-02: clean acc bar (all models) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    vals = [r["acc"] * 100 for r in rows]
    bars = ax.bar(x, vals, color=colors, width=0.65, zorder=3)
    bar_labels(ax, bars, vals, "%.2f%%")
    ax.axhline(base_acc * 100, color="#999", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Clean Accuracy (%)"); ax.set_ylim(91, 98)
    ax.set_title("Clean Accuracy — All Compression Arms", fontweight="bold")
    ax.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5, loc="lower right")
    add_group_seps(ax, len(rows))
    fig.tight_layout(); save_fig(fig, "phase_2", "02_clean_acc_bar")

    # ── P2-03: compressed size bar ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    vals = [r["size_mb"] for r in rows]
    bars = ax.bar(x, vals, color=colors, width=0.65, zorder=3)
    bar_labels(ax, bars, vals, "%.1f")
    base_fp32 = data["baseline_resnet18"]["metrics"]["size_mb_fp32"]
    ax.axhline(base_fp32, color="#999", ls="--", lw=1, alpha=0.7, label=f"Baseline FP32 ({base_fp32:.1f} MB)")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Compressed Size (MB)")
    ax.set_title("Model Storage Size After Compression", fontweight="bold")
    ax.legend(handles=ARM_LEGEND + [mpatches.Patch(color="#999", label=f"Baseline ({base_fp32:.1f} MB)")],
              ncol=5, fontsize=7.5)
    add_group_seps(ax, len(rows))
    fig.tight_layout(); save_fig(fig, "phase_2", "03_model_size_bar")

    # ── P2-04: latency bar ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    vals = [r["lat"] for r in rows]
    bars = ax.bar(x, vals, color=colors, width=0.65, zorder=3)
    bar_labels(ax, bars, vals, "%.2f")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Latency (ms / image, batch-size 1)")
    ax.set_title("Inference Latency per Image", fontweight="bold")
    ax.text(0.99, 0.97,
            "Quant/prune latency = fake-quant overhead,\nnot a deployment number",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#888",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    ax.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5)
    add_group_seps(ax, len(rows))
    fig.tight_layout(); save_fig(fig, "phase_2", "04_latency_bar")

    # ── P2-05: nonzero params bar ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    vals = [r["nzp"] / 1e6 for r in rows]
    bars = ax.bar(x, vals, color=colors, width=0.65, zorder=3)
    bar_labels(ax, bars, vals, "%.2f M")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Non-zero Parameters (M)")
    ax.set_title("Effective Parameter Count After Compression", fontweight="bold")
    ax.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5)
    add_group_seps(ax, len(rows))
    fig.tight_layout(); save_fig(fig, "phase_2", "05_nonzero_params_bar")

    # ── P2-06: accuracy–size scatter ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in rows:
        ax.scatter(r["size_mb"], r["acc"] * 100,
                   color=ARM_COLOR[r["arm"]], s=65, zorder=3,
                   edgecolors="white", linewidths=0.5)
        label = MODEL_LABEL.get(r["run"], r["run"]).replace("\n", " ")
        ax.annotate(label, (r["size_mb"], r["acc"] * 100),
                    textcoords="offset points", xytext=(5, 2), fontsize=6)
    ax.set_xlabel("Compressed Model Size (MB)"); ax.set_ylabel("Clean Accuracy (%)")
    ax.set_title("Accuracy–Size Trade-off", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=ARM_COLOR[a], label=ARM_LABEL[a])
                        for a in ["quant", "prune", "dense_small", "distill"]],
              fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); save_fig(fig, "phase_2", "06_acc_size_tradeoff")

    # ── P2-07: accuracy degradation vs ratio (line) ───────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in arms:
        pts = arm_rows[arm]
        if not pts: continue
        ax.plot([r["ratio"] for r in pts],
                [(base_acc - r["acc"]) * 100 for r in pts],
                "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm], lw=1.8, ms=6)
    ax.axhline(0, color="#999", ls="--", lw=1)
    ax.set_xticks([2, 4, 8]); ax.set_xticklabels(["2×", "4×", "8×"])
    ax.set_xlabel("Nominal Compression Ratio")
    ax.set_ylabel("Accuracy Drop vs. Baseline (percentage points)")
    ax.set_title("Accuracy Degradation Under Compression", fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=7.5)
    fig.tight_layout(); save_fig(fig, "phase_2", "07_accuracy_degradation")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_3(data):
    print("\n[Phase 3]")

    u_models = [m for m in MODEL_ORDER if f"universal_{m}" in data]
    l_models = [m for m in MODEL_ORDER if f"lavan_{m}" in data]
    if not u_models:
        print("  [warn] No universal attack data."); return

    xu = np.arange(len(u_models))
    xl = np.arange(len(l_models))
    xu_labels = [MODEL_LABEL.get(m, m) for m in u_models]
    xl_labels = [MODEL_LABEL.get(m, m) for m in l_models]
    u_colors  = [ARM_COLOR[arm_of(m)] for m in u_models]
    l_colors  = [ARM_COLOR[arm_of(m)] for m in l_models]

    # ── P3-01: Universal patched accuracy (side 5 + 8) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, side, title in zip(axes,
                                [5, 8],
                                ["Side-5 patch (2.4% area)", "Side-8 patch (6.25% area)"]):
        acc_vals  = [univ_m(univ(data, m), side)["acc"]   * 100 for m in u_models]
        acc_std   = [univ_m(univ(data, m), side)["acc_std"] * 100 for m in u_models]
        noise_vals= [univ_m(univ(data, m), side)["noise"] * 100 for m in u_models]
        b1 = ax.bar(xu - 0.2, acc_vals,   0.35, color=u_colors, zorder=3, label="Universal patch")
        b2 = ax.bar(xu + 0.2, noise_vals, 0.35, color=u_colors, zorder=3, alpha=0.4,
                    hatch="///", label="Noise patch (occlusion)")
        ax.errorbar(xu - 0.2, acc_vals, yerr=acc_std,
                    fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(xu); ax.set_xticklabels(xu_labels)
        ax.set_ylabel("Accuracy under Patch (%)"); ax.set_ylim(45, 102)
        ax.set_title(f"Universal Attack — {title}", fontweight="bold")
        add_group_seps(ax, len(u_models))
        if ax is axes[0]:
            ax.legend(fontsize=7.5)
    fig.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Universal Patch: Accuracy Under Attack (Brown et al.)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "phase_3", "01_universal_patched_acc")

    # ── P3-02: Universal target success rate ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, side, title in zip(axes,
                                [5, 8],
                                ["Side-5 (2.4% area)", "Side-8 (6.25% area)"]):
        vals = [univ_m(univ(data, m), side)["hit"]     * 100 for m in u_models]
        stds = [univ_m(univ(data, m), side)["hit_std"] * 100 for m in u_models]
        bars = ax.bar(xu, vals, 0.65, color=u_colors, zorder=3)
        ax.errorbar(xu, vals, yerr=stds, fmt="none", color="black", capsize=3, lw=1)
        bar_labels(ax, bars, vals, "%.1f%%")
        ax.set_xticks(xu); ax.set_xticklabels(xu_labels)
        ax.set_ylabel("Mean Target Success Rate (%) — over 10 targets")
        ax.set_title(f"Universal Patch Target Success — {title}", fontweight="bold")
        add_group_seps(ax, len(u_models))
    fig.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Universal Patch: Target Success Rate",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "phase_3", "02_universal_target_success")

    # ── P3-03: LaVAN robust accuracy (untargeted) ─────────────────────────
    if l_models:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, side, title in zip(axes,
                                    [5, 8],
                                    ["Side-5 (2.4% area)", "Side-8 (6.25% area)"]):
            vals = [lavan_m(lavan(data, m), side, "untargeted") * 100 for m in l_models]
            bars = ax.bar(xl, vals, 0.65, color=l_colors, zorder=3)
            bar_labels(ax, bars, vals, "%.1f%%")
            ax.set_xticks(xl); ax.set_xticklabels(xl_labels)
            ax.set_ylabel("Robust Accuracy (%) — images resisting attack")
            ax.set_title(f"LaVAN Untargeted — {title}", fontweight="bold")
            ymax = max(v for v in vals if not np.isnan(v)) * 1.35 if vals else 20
            ax.set_ylim(0, max(ymax, 5))
            add_group_seps(ax, len(l_models))
        fig.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("LaVAN Per-Image Patch: Robust Accuracy Under Untargeted Attack",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        save_fig(fig, "phase_3", "03_lavan_robust_acc")

        # ── P3-04: LaVAN targeted success ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, side, title in zip(axes,
                                    [5, 8],
                                    ["Side-5 (2.4% area)", "Side-8 (6.25% area)"]):
            vals = [lavan_m(lavan(data, m), side, "targeted") * 100 for m in l_models]
            bars = ax.bar(xl, vals, 0.65, color=l_colors, zorder=3)
            bar_labels(ax, bars, vals, "%.1f%%")
            ax.set_xticks(xl); ax.set_xticklabels(xl_labels)
            ax.set_ylabel("Targeted Attack Success Rate (% images fooled to target)")
            ax.set_title(f"LaVAN Targeted — {title}", fontweight="bold")
            add_group_seps(ax, len(l_models))
        fig.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("LaVAN Per-Image Patch: Targeted Attack Success Rate",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        save_fig(fig, "phase_3", "04_lavan_targeted_success")

    # ── P3-05: Convergence curves (universal, side 5 & 8) ─────────────────
    focus = ["baseline_resnet18", "quant_4x_int8", "prune_4x", "dense_small_4x", "kd_4x"]
    focus = [m for m in focus if f"universal_{m}" in data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, side, title in zip(axes, [5, 8], ["Side-5", "Side-8"]):
        for model in focus:
            ud = univ(data, model)
            per_tgt = ud.get("metrics", {}).get(f"side_{side}", {}).get("per_target", [])
            steps_d: dict = {}
            for t_entry in per_tgt:
                wb = t_entry.get("whitebox", t_entry.get("effective", {}))
                for cp in wb.get("curve", []):
                    s = cp["step"]
                    steps_d.setdefault(s, []).append(cp.get("target_success", 0))
            if not steps_d:
                continue
            steps = sorted(steps_d)
            means = [np.mean(steps_d[s]) * 100 for s in steps]
            lbl   = MODEL_LABEL.get(model, model).replace("\n", " ")
            ax.plot(steps, means, "o-", color=ARM_COLOR[arm_of(model)],
                    label=lbl, lw=1.8, ms=5)
        ax.set_xlabel("Optimization Steps")
        ax.set_ylabel("Mean Target Success Rate (%) — 10 targets")
        ax.set_title(f"Attack Convergence — {title}", fontweight="bold")
        ax.legend(fontsize=7.5, framealpha=0.9)
    fig.suptitle("Universal Patch Optimization Convergence\n(Baseline + 4× models)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); save_fig(fig, "phase_3", "05_convergence_curves")

    # ── P3-06: Physical EOT ablation ──────────────────────────────────────
    phys_models = ["baseline_resnet18", "quant_4x_int8", "prune_4x", "dense_small_4x", "kd_4x"]
    phys_models = [m for m in phys_models if f"universal_physical_{m}" in data]

    if phys_models:
        xp = np.arange(len(phys_models))
        ph_labels = [MODEL_LABEL.get(m, m).replace("\n", " ") for m in phys_models]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        for ax, key, ylabel in zip(
            axes,
            ["acc", "hit"],
            ["Accuracy Under Patch (%)", "Target Success Rate (%)"],
        ):
            dig = [univ_m(univ(data, m), 8)[key] * 100 for m in phys_models]
            peo = [univ_m(phys(data, m),  8)[key] * 100 for m in phys_models]
            w = 0.35
            b1 = ax.bar(xp - w/2, dig, w, color="#1f77b4", label="Location-only (digital)", zorder=3)
            b2 = ax.bar(xp + w/2, peo, w, color="#ff7f0e", label="Physical EOT",            zorder=3)
            bar_labels(ax, b1, dig, "%.1f")
            bar_labels(ax, b2, peo, "%.1f")
            ax.set_xticks(xp); ax.set_xticklabels(ph_labels)
            ax.set_ylabel(ylabel); ax.legend(fontsize=7.5)

        fig.suptitle("Physical EOT Ablation — Universal Patch, Side-8 (4× models)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(); save_fig(fig, "phase_3", "06_physical_eot_ablation")

    # ── P3-07: Per-target heatmap ─────────────────────────────────────────
    hm_models = ["baseline_resnet18", "quant_4x_int8", "prune_4x", "dense_small_4x", "kd_4x"]
    hm_models = [m for m in hm_models if f"universal_{m}" in data]

    if hm_models:
        matrix = np.full((10, len(hm_models)), np.nan)
        for j, model in enumerate(hm_models):
            for t_entry in univ(data, model).get("metrics", {}).get("side_8", {}).get("per_target", []):
                tgt = t_entry["target"]
                eff = t_entry.get("effective", {})
                ts  = eff.get("target_success", np.nan)
                matrix[tgt, j] = ts * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=75)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Target Success Rate (%)", fontsize=9)
        ax.set_xticks(range(len(hm_models)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m).replace("\n", " ") for m in hm_models], fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"→ {c}" for c in CIFAR10], fontsize=8)
        ax.set_xlabel("Model"); ax.set_ylabel("Target Class")
        ax.set_title("Universal Patch (Side-8): Per-Target Success Rate\n"
                     "Rows = target class  ·  Cols = model",
                     fontsize=10, fontweight="bold")
        for j in range(len(hm_models)):
            for i in range(10):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 50 else "white")
        fig.tight_layout(); save_fig(fig, "phase_3", "07_per_target_heatmap")

    # ── P3-08: Noise-patch (occlusion) accuracy, side 5 & 8 ──────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    w = 0.35
    n5 = [univ_m(univ(data, m), 5)["noise"] * 100 for m in u_models]
    n8 = [univ_m(univ(data, m), 8)["noise"] * 100 for m in u_models]
    b5 = ax.bar(xu - w/2, n5, w, color="#5babd6", label="Side-5 noise patch", zorder=3)
    b8 = ax.bar(xu + w/2, n8, w, color="#3f6ea5", label="Side-8 noise patch", zorder=3)
    bar_labels(ax, b5, n5, "%.1f%%")
    bar_labels(ax, b8, n8, "%.1f%%")
    ax.set_xticks(xu); ax.set_xticklabels(xu_labels)
    ax.set_ylabel("Accuracy under Noise Patch (%) — occlusion only")
    ax.set_title("Occlusion Baseline: Noise-Patch Accuracy (Random Patch, No Adversarial Signal)",
                 fontweight="bold")
    ax.set_ylim(78, 95); ax.legend(fontsize=8)
    add_group_seps(ax, len(u_models))
    fig.tight_layout(); save_fig(fig, "phase_3", "08_noise_patch_occlusion")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(data):
    print("\n[Summary]")

    rows2 = phase2_table(data)
    if not rows2:
        print("  [warn] No Phase 2 rows."); return

    base_acc = next((r["acc"] for r in rows2 if r["run"] == "baseline_resnet18"), np.nan)

    # Models that have all three data sources: Phase 2 clean acc, universal, lavan
    models = [m for m in MODEL_ORDER
              if m in data
              and f"universal_{m}" in data
              and f"lavan_{m}" in data]
    if not models:
        print("  [warn] No fully-matched models for summary."); return

    x      = np.arange(len(models))
    labels = [MODEL_LABEL.get(m, m) for m in models]
    colors = [ARM_COLOR[arm_of(m)] for m in models]

    clean = [data[m]["metrics"].get("clean_acc", np.nan) * 100        for m in models]
    u8acc = [univ_m(univ(data, m), 8)["acc"] * 100                    for m in models]
    l5rob = [lavan_m(lavan(data, m), 5, "untargeted") * 100           for m in models]

    # ── S-01: Triple-bar phase progression ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.25
    ax.bar(x - w,   clean, w, color="#2196F3", label="Clean Accuracy",              zorder=3)
    ax.bar(x,       u8acc, w, color="#F44336", label="Acc under U8 Universal Patch", zorder=3)
    ax.bar(x + w,   l5rob, w, color="#4CAF50", label="Robust Acc (LaVAN S5 Untarg.)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Phase 1–3 Summary: Clean / Universal-Patch / LaVAN Robustness",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    add_group_seps(ax, len(models))
    fig.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "summary", "01_phase_progression")

    # ── S-02: Clean acc vs U8 patched acc scatter ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, ca, u8 in zip(models, clean, u8acc):
        r = ratio_of(m)
        sz = {1: 130, 2: 80, 4: 60, 8: 40}[r]
        ax.scatter(ca, u8, color=ARM_COLOR[arm_of(m)], s=sz, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(MODEL_LABEL.get(m, m).replace("\n", " "),
                    (ca, u8), textcoords="offset points", xytext=(5, 2), fontsize=6)
    ax.plot([91, 97], [91, 97], "k--", lw=0.8, alpha=0.35, label="No-drop diagonal")
    ax.set_xlabel("Clean Accuracy (%)"); ax.set_ylabel("Accuracy Under U8 Universal Patch (%)")
    ax.set_title("Clean vs. Adversarial Robustness Trade-off", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=ARM_COLOR[a], label=ARM_LABEL[a])
                        for a in ["baseline", "quant", "prune", "dense_small", "distill"]],
              fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); save_fig(fig, "summary", "02_clean_vs_adversarial")

    # ── S-03: Model size vs U8 robustness scatter ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m, u8 in zip(models, u8acc):
        m_data = data[m]["metrics"]
        sz_mb  = m_data.get("size_mb_compressed", m_data.get("size_mb_fp32", np.nan))
        ax.scatter(sz_mb, u8, color=ARM_COLOR[arm_of(m)], s=65, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(MODEL_LABEL.get(m, m).replace("\n", " "),
                    (sz_mb, u8), textcoords="offset points", xytext=(5, 2), fontsize=6)
    ax.set_xlabel("Compressed Model Size (MB)")
    ax.set_ylabel("Accuracy Under U8 Universal Patch (%)")
    ax.set_title("Model Size vs. Universal-Patch Robustness", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=ARM_COLOR[a], label=ARM_LABEL[a])
                        for a in ["quant", "prune", "dense_small", "distill"]],
              fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); save_fig(fig, "summary", "03_size_vs_robustness")

    # ── S-04: RQ4 — pruning vs dense-small vs distilled ──────────────────
    ratios = [2, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    specs = [
        ("Clean Accuracy (%)",            92, 97,  "acc"),
        ("Acc Under U8 Universal Patch (%)", 45, 70, "u8"),
        ("LaVAN S5 Robust Acc (%)",        0, 15,  "l5"),
    ]

    def _val(run, key):
        if key == "acc":
            return data.get(run, {}).get("metrics", {}).get("clean_acc", np.nan) * 100
        if key == "u8":
            return univ_m(univ(data, run), 8)["acc"] * 100
        if key == "l5":
            return lavan_m(lavan(data, run), 5, "untargeted") * 100

    for ax, (ylabel, ymin, ymax, key) in zip(axes, specs):
        xr = np.arange(3)
        w  = 0.28
        pv = [_val(f"prune_{r}x",       key) for r in ratios]
        dv = [_val(f"dense_small_{r}x", key) for r in ratios]
        kv = [_val(f"kd_{r}x",          key) for r in ratios]
        ax.bar(xr - w, pv, w, color=ARM_COLOR["prune"],       label="Pruned",      zorder=3)
        ax.bar(xr,     dv, w, color=ARM_COLOR["dense_small"], label="Dense-small", zorder=3)
        ax.bar(xr + w, kv, w, color=ARM_COLOR["distill"],     label="Distilled",   zorder=3)
        ax.set_xticks(xr); ax.set_xticklabels(["2×", "4×", "8×"])
        ax.set_xlabel("Compression Ratio"); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ymin, ymax); ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle("RQ4: Sparsity vs. Capacity — Pruned vs. Dense-small vs. Distilled\n"
                 "(Matched nonzero parameters at each ratio)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(); save_fig(fig, "summary", "04_rq4_sparsity_vs_capacity")

    # ── S-05: Vulnerability gap (clean − U8 patched) ─────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    gap  = [ca - u8 for ca, u8 in zip(clean, u8acc)]
    bars = ax.bar(x, gap, color=colors, width=0.65, zorder=3)
    bar_labels(ax, bars, gap, "%.1f pp")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy Drop (Clean − U8 Patched, pp)")
    ax.set_title("Universal Patch Vulnerability Gap (Side-8)", fontweight="bold")
    ax.legend(handles=ARM_LEGEND, ncol=5, fontsize=7.5, loc="upper right")
    add_group_seps(ax, len(models))
    fig.tight_layout(); save_fig(fig, "summary", "05_vulnerability_gap")


# ══════════════════════════════════════════════════════════════════════════════
# MAPPING REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_mapping(data):
    print("\n[Mapping] File -> Phase -> Key Metrics")
    print("-" * 90)
    for key, d in sorted(data.items()):
        attack = d.get("attack")
        m      = d.get("metrics", {})
        if key == "baseline_resnet18":
            phase = "Phase 1"
            info  = (f"clean_acc={m.get('clean_acc',0)*100:.2f}%  "
                     f"params={m.get('params',0)/1e6:.2f}M  "
                     f"size={m.get('size_mb_fp32',0):.1f} MB  "
                     f"lat={m.get('latency_ms_bs1',0):.2f} ms")
        elif attack is None:
            phase = "Phase 2"
            info  = (f"clean_acc={m.get('clean_acc',0)*100:.2f}%  "
                     f"nzp={m.get('nonzero_params', m.get('params',0))/1e6:.2f}M  "
                     f"size={m.get('size_mb_compressed', m.get('size_mb_fp32',0)):.1f} MB")
        elif attack.get("physical"):
            phase = "Phase 3 · Physical EOT"
            s8    = m.get("side_8", {})
            pa    = s8.get("effective_patched_acc", {})
            info  = (f"U8_acc={pa.get('mean',0)*100:.1f}%  "
                     f"hit={s8.get('effective_target_success',{}).get('mean',0)*100:.1f}%")
        elif attack.get("name") == "universal":
            phase = "Phase 3 · Universal"
            s5    = m.get("side_5", {}); s8 = m.get("side_8", {})
            info  = (f"U5_acc={s5.get('effective_patched_acc',{}).get('mean',0)*100:.1f}%  "
                     f"U8_acc={s8.get('effective_patched_acc',{}).get('mean',0)*100:.1f}%  "
                     f"U8_hit={s8.get('effective_target_success',{}).get('mean',0)*100:.1f}%")
        elif attack.get("name") == "lavan":
            phase = "Phase 3 · LaVAN"
            l5u   = m.get("side_5_untargeted", {}).get("effective", {}).get("robust_acc", 0)
            l8u   = m.get("side_8_untargeted", {}).get("effective", {}).get("robust_acc", 0)
            l5t   = m.get("side_5_targeted",   {}).get("effective", {}).get("success_on_correct", 0)
            info  = (f"L5_robust={l5u*100:.1f}%  L8_robust={l8u*100:.1f}%  "
                     f"L5_tgt_hit={l5t*100:.1f}%")
        else:
            phase = "Unknown"
            info  = ""
        print(f"  {key:<50} {phase:<25} {info}")
    print("-" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  CNN Adversarial Patch — Visualization Pipeline")
    print("=" * 70)

    setup_dirs()

    print("\n[Loading] results/*.json ...")
    data = load_all_results()
    print(f"  {len(data)} files loaded.")

    print_mapping(data)

    plot_phase_1(data)
    plot_phase_2(data)
    plot_phase_3(data)
    plot_summary(data)

    print("\n" + "=" * 70)
    print("  Done. Output summary:")
    for key, d in PHASE_DIRS.items():
        pngs = list(d.glob("*.png"))
        print(f"  {str(d):<40} {len(pngs)} figures")
    print("=" * 70)
