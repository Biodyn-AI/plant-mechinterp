"""Regenerate main-paper figures on real data.

Reads JSON/CSV outputs produced by:
  - analysis/probing/run_probing.py          -> data/real/results/probing/
  - analysis/baselines/train_baselines.py    -> data/real/results/baselines/
  - analysis/sae/train_sae.py                -> data/real/results/sae/
  - analysis/sae/motif_enrichment.py         -> data/real/results/sae_enrichment/
  - analysis/mechanistic/activation_patching_v3.py -> data/real/results/patching/
  - analysis/mechanistic/head_ablations.py   -> data/real/results/mechanistic/
  - analysis/cross_species/cross_species_v2.py -> data/real/results/cross_species/

Produces figures under paper/figures_real/ (trusted by the revised manuscript).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = REPO / "data" / "real" / "results"
OUT = REPO / "paper" / "figures_real"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 140, "savefig.dpi": 300,
})


def fig_probing_with_baselines(dataset: str = "region_type") -> Path:
    p = RES / "probing" / f"{dataset}.json"
    if not p.exists():
        print(f"[skip] {p}")
        return None
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    layers_t = sorted(int(k) for k in d["layers"]["trained"].keys())
    acc_t = [d["layers"]["trained"][str(l)]["cv_acc_mean"] for l in layers_t]
    std_t = [d["layers"]["trained"][str(l)]["cv_acc_std"] for l in layers_t]
    ci_lo = [d["layers"]["trained"][str(l)]["cv_acc_ci95"][0] for l in layers_t]
    ci_hi = [d["layers"]["trained"][str(l)]["cv_acc_ci95"][1] for l in layers_t]
    axes[0].errorbar(layers_t, acc_t, yerr=std_t, marker="o", color="C0", label="trained")
    axes[0].fill_between(layers_t, ci_lo, ci_hi, color="C0", alpha=0.15,
                         label="95% CI (CV-fold bootstrap)")

    if "random" in d["layers"]:
        layers_r = sorted(int(k) for k in d["layers"]["random"].keys())
        acc_r = [d["layers"]["random"][str(l)]["cv_acc_mean"] for l in layers_r]
        std_r = [d["layers"]["random"][str(l)]["cv_acc_std"] for l in layers_r]
        axes[0].errorbar(layers_r, acc_r, yerr=std_r, marker="s", color="C3", label="random init")

    base = d.get("baselines", {})
    for name, c in [("3mer", "gray"), ("4mer", "silver"), ("5mer", "dimgray"), ("gc", "orange")]:
        if name in base:
            axes[0].axhline(base[name]["cv_acc_mean"], color=c, ls="--", lw=1.0,
                            label=f"{name}: {base[name]['cv_acc_mean']:.3f}")
    pretty = {
        "region_type": "Region type (5-way)",
        "splice": "Splice sites (3-way)",
        "tss": "TSS detection (binary)",
        "promoter": "Promoter (binary)",
    }.get(dataset, dataset)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("5-fold CV accuracy")
    axes[0].set_title(f"{pretty}: per-layer probing")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.3)

    # Advantage over random
    if "random" in d["layers"]:
        adv = [at - ar for at, ar in zip(acc_t, acc_r)]
        axes[1].bar(layers_t, adv, color="C2")
        axes[1].set_title("Trained − random (Δ accuracy)")
        axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Δ accuracy")
        axes[1].grid(alpha=0.3)
    # Test-set accuracy (chromosome split)
    test_t = [d["layers"]["trained"][str(l)].get("test_acc", np.nan) for l in layers_t]
    axes[2].plot(layers_t, test_t, marker="o", color="C0", label="trained (test chrom)")
    if "random" in d["layers"]:
        test_r = [d["layers"]["random"][str(l)].get("test_acc", np.nan) for l in layers_t]
        axes[2].plot(layers_t, test_r, marker="s", color="C3", label="random (test chrom)")
    for name, c in [("3mer", "gray"), ("gc", "orange")]:
        if name in base and "test_acc" in base[name]:
            axes[2].axhline(base[name]["test_acc"], color=c, ls="--", lw=1.0, label=f"{name} (test)")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Test accuracy (held-out chromosome)")
    axes[2].set_title("Held-out-chromosome evaluation")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Probing on {pretty}", y=1.02)
    plt.tight_layout()
    out = OUT / f"probing_{dataset}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def fig_baselines_comparison(datasets: list[str]) -> Path:
    rows = []
    # Plant-DnaGemma: use best-layer CV test_acc from probing/{ds}.json
    for ds in datasets:
        p_probe = RES / "probing" / f"{ds}.json"
        p_base = RES / "baselines" / f"{ds}.json"
        if p_probe.exists():
            d = json.loads(p_probe.read_text())
            best_l = max(d["layers"]["trained"], key=lambda li: d["layers"]["trained"][li]["cv_acc_mean"])
            r = d["layers"]["trained"][best_l]
            rows.append(dict(dataset=ds, model="Plant-DnaGemma (best layer)",
                             test_acc=r.get("test_acc", r["cv_acc_mean"]), std=r["cv_acc_std"]))
            for name in ("3mer", "4mer", "5mer", "gc"):
                if name in d.get("baselines", {}):
                    rows.append(dict(dataset=ds, model=name,
                                     test_acc=d["baselines"][name].get("test_acc",
                                              d["baselines"][name]["cv_acc_mean"]),
                                     std=d["baselines"][name]["cv_acc_std"]))
        if p_base.exists():
            db = json.loads(p_base.read_text())
            for model, stat in db.get("summary", {}).items():
                rows.append(dict(dataset=ds, model=model, test_acc=stat["test_acc_mean"],
                                 std=stat["test_acc_std"]))
    if not rows:
        print("[skip] no baseline data yet")
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="dataset", columns="model", values="test_acc")
    err = df.pivot_table(index="dataset", columns="model", values="std")
    ax = piv.plot.bar(yerr=err, figsize=(10, 4), capsize=2)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learned baselines vs Plant-DnaGemma (AraReg)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    out = OUT / "baselines_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")
    return out


def fig_cross_species() -> Path:
    p = RES / "cross_species" / "cross_species_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (tag, title) in enumerate([
        ("multispecies", "Natural-GC"),
        ("multispecies_gc_matched", "GC-matched"),
        ("multispecies_heldout", "Held-out species"),
    ]):
        if tag not in d:
            continue
        r = d[tag]
        layers_t = sorted(int(k) for k in r["trained_per_layer"])
        acc = [r["trained_per_layer"][str(l)]["acc_mean"] for l in layers_t]
        std = [r["trained_per_layer"][str(l)]["acc_std"] for l in layers_t]
        axes[i].errorbar(layers_t, acc, yerr=std, marker="o", color="C0", label="trained")
        if "random_per_layer" in r:
            accr = [r["random_per_layer"][str(l)]["acc_mean"] for l in layers_t]
            axes[i].plot(layers_t, accr, marker="s", color="C3", label="random")
        if "gc_only" in r:
            axes[i].axhline(r["gc_only"]["acc_mean"], color="orange", ls="--",
                            label=f"GC only: {r['gc_only']['acc_mean']:.3f}")
        for kname, c in [("3mer", "gray"), ("4mer", "silver"), ("5mer", "dimgray")]:
            if kname in r:
                axes[i].axhline(r[kname]["acc_mean"], color=c, ls=":", lw=0.8,
                                label=f"{kname}: {r[kname]['acc_mean']:.3f}")
        axes[i].set_title(title)
        axes[i].set_xlabel("Layer"); axes[i].set_ylabel("5-fold CV accuracy")
        axes[i].legend(fontsize=7); axes[i].grid(alpha=0.3)
    fig.suptitle("Cross-species v2: probing with controls", y=1.02)
    plt.tight_layout()
    out = OUT / "cross_species_v2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def fig_patching() -> Path:
    p = RES / "patching" / "splice_patching.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, direction in enumerate(["denoise", "noise"]):
        for mode, color in [("canon", "C0"), ("shuf", "C1")]:
            eff = d["layers"][mode][direction]
            layers = sorted(int(k) for k in eff)
            mean = [eff[str(l)]["mean"] for l in layers]
            lo = [eff[str(l)]["ci95_low"] for l in layers]
            hi = [eff[str(l)]["ci95_high"] for l in layers]
            axes[i].plot(layers, mean, marker="o", color=color, label=mode)
            axes[i].fill_between(layers, lo, hi, color=color, alpha=0.15)
        axes[i].axhline(0, color="gray", lw=0.5)
        axes[i].set_xlabel("Layer patched"); axes[i].set_ylabel("Normalized effect")
        axes[i].set_title(f"{direction}")
        axes[i].legend(fontsize=7); axes[i].grid(alpha=0.3)
    fig.suptitle("Activation patching on splice sites")
    plt.tight_layout()
    out = OUT / "patching_splice.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def fig_head_ablations() -> Path:
    for task in ("splice", "tss"):
        p = RES / "mechanistic" / f"head_ablations_{task}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        probe_layer = 7  # known: probe reads at L7; ablating downstream is trivially 0.
        n_layers = 12
        n_heads = 12
        arr = np.full((n_layers, n_heads), np.nan)
        for h in d["heads"]:
            arr[h["layer"], h["head"]] = h["acc_drop"]
        mlp = np.full(n_layers, np.nan)
        for m in d["mlps"]:
            mlp[m["layer"]] = m["acc_drop"]

        # Restrict to layers that actually flow into the probe readout.
        keep = np.arange(probe_layer)  # 0..6
        arr_k = arr[keep]
        mlp_k = mlp[keep]

        fig = plt.figure(figsize=(13, 4.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.4, 2], wspace=0.35)

        # Panel A: head-ablation heatmap, layers 0..6 only.
        ax0 = fig.add_subplot(gs[0, 0])
        vmax = np.nanmax(np.abs(arr_k)) if np.isfinite(arr_k).any() else 1.0
        im = ax0.imshow(arr_k, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax0.set_xlabel("Head index"); ax0.set_ylabel("Layer")
        ax0.set_yticks(keep); ax0.set_yticklabels(keep)
        ax0.set_xticks(range(n_heads))
        ax0.set_title(
            "(a) Head ablation Δ = base − ablated probe accuracy\n"
            f"(layers $\\geq${probe_layer} omitted; downstream of probe readout)"
        )
        plt.colorbar(im, ax=ax0, fraction=0.04, pad=0.03, label="Δ accuracy")

        # Panel B: MLP-ablation bar chart, same layers.
        ax1 = fig.add_subplot(gs[0, 1])
        colors = ["#b31b1b" if v > 0.05 else "#e89191" if v > 0.02 else "#c5c5c5" for v in mlp_k]
        ax1.barh(keep, mlp_k, color=colors, edgecolor="black")
        ax1.set_yticks(keep); ax1.set_yticklabels([f"L{l}" for l in keep])
        ax1.invert_yaxis()
        ax1.set_xlabel("Δ accuracy (test)")
        ax1.set_title("(b) MLP ablation effect")
        ax1.grid(axis="x", alpha=0.3)
        span = max(0.32, float(np.nanmax(mlp_k)) + 0.02)
        ax1.set_xlim(left=-0.02, right=span)
        for y, v in zip(keep, mlp_k):
            # Place number inside bar if bar is long enough, else outside.
            if v > span * 0.25:
                ax1.text(v - span * 0.02, y, f"{v:+.2f}", va="center",
                         ha="right", fontsize=9, color="white", fontweight="bold")
            else:
                ax1.text(v + span * 0.015, y, f"{v:+.2f}", va="center",
                         ha="left", fontsize=9)

        # Panel C: sorted head drops (marginal distribution).
        ax2 = fig.add_subplot(gs[0, 2])
        all_head_drops = sorted([h["acc_drop"] for h in d["heads"]
                                 if h["layer"] < probe_layer], reverse=True)
        x = np.arange(len(all_head_drops))
        ax2.plot(x, all_head_drops, color="#1f77b4", lw=1.2)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axhline(0.03, color="red", lw=0.8, ls="--", label="3% threshold")
        n_crit = sum(d_ > 0.03 for d_ in all_head_drops)
        ax2.set_xlabel(f"Heads ranked by effect (1–{len(all_head_drops)})")
        ax2.set_ylabel("Δ accuracy")
        ax2.set_title(f"(c) Head-drop distribution  ({n_crit}/{len(all_head_drops)} > 3%)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        fig.suptitle(
            f"Per-component ablations on the {task} task "
            f"(baseline acc {d['baseline_acc']:.3f}, test N={d['n_test']})",
            y=1.02,
        )
        out = OUT / f"head_ablations_{task}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def fig_sae() -> Path:
    sae_files = sorted((RES / "sae").glob("*.json"))
    sae_files = [p for p in sae_files if not p.name.startswith("._")]
    if not sae_files:
        return None

    # Parse all SAEs and group by config.
    entries = []
    for sf in sae_files:
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        entries.append(d)

    def config_of(d):
        return f"{d['variant']} ×{int(d['expansion'])}"

    # Aggregate: final var_explained, L0, dead%, per (variant, expansion)
    import collections
    by_cfg = collections.defaultdict(list)
    for d in entries:
        by_cfg[config_of(d)].append(d)

    cfgs = sorted(by_cfg.keys())
    var_mean = [np.mean([d["var_explained"] for d in by_cfg[c]]) for c in cfgs]
    var_sd   = [np.std([d["var_explained"] for d in by_cfg[c]], ddof=1) if len(by_cfg[c])>1 else 0 for c in cfgs]
    l0_mean  = [np.mean([d["L0_mean"] for d in by_cfg[c]]) for c in cfgs]
    l0_sd    = [np.std([d["L0_mean"] for d in by_cfg[c]], ddof=1) if len(by_cfg[c])>1 else 0 for c in cfgs]
    dead_mean= [np.mean([d["dead_fraction"] for d in by_cfg[c]]) * 100 for c in cfgs]
    sparsity_mean = [np.mean([d["sparsity"] for d in by_cfg[c]]) for c in cfgs]

    # Load motif enrichment + annotation enrichment best CSVs for TopK 16× if available
    me_dir = RES / "sae_enrichment"
    tag = "topk_exp16_L7_seed0"
    mot_best = me_dir / f"{tag}_best_per_feature.csv"
    ann_best = me_dir / f"{tag}_annotation_best_per_feature.csv"

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.35)

    # (a) Training reconstruction curves, one line per config (seed-averaged)
    ax_a = fig.add_subplot(gs[0, 0])
    import matplotlib.cm as cm
    color_map = {c: cm.tab10(i) for i, c in enumerate(cfgs)}
    for c in cfgs:
        hists = [d["history"] for d in by_cfg[c]]
        # align by epoch index
        n_ep = min(len(h) for h in hists)
        recon = np.stack([[h[i]["val_recon"] for i in range(n_ep)] for h in hists], axis=0)
        mean = recon.mean(axis=0)
        std = recon.std(axis=0, ddof=1) if len(hists) > 1 else np.zeros_like(mean)
        ep = np.arange(n_ep)
        ax_a.plot(ep, mean, label=c, color=color_map[c], lw=1.5)
        ax_a.fill_between(ep, mean - std, mean + std, color=color_map[c], alpha=0.2)
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Val reconstruction MSE")
    ax_a.set_title("(a) SAE reconstruction curves")
    ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # (b) Quality bar chart: var_explained and L0 per config (log-scale L0)
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(cfgs))
    bars = ax_b.bar(x, var_mean, yerr=var_sd, color=[color_map[c] for c in cfgs],
                    edgecolor="black", capsize=3)
    ax_b.set_ylim(0.80, 1.00)
    ax_b.set_xticks(x); ax_b.set_xticklabels(cfgs, rotation=15)
    ax_b.set_ylabel("Fraction of variance explained")
    ax_b.set_title("(b) Reconstruction quality (mean ± SD over 3 seeds)")
    for xi, vm in zip(x, var_mean):
        ax_b.text(xi, vm + 0.005, f"{vm:.3f}", ha="center", fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    # (c) Sparsity/L0 bar chart — both metrics in one
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c2 = ax_c.twinx()
    bars1 = ax_c.bar(x - 0.18, l0_mean, width=0.36, yerr=l0_sd, color="#4c72b0",
                     edgecolor="black", capsize=3, label="$L_0$ (active features / seq)")
    bars2 = ax_c2.bar(x + 0.18, [s * 100 for s in sparsity_mean], width=0.36,
                      color="#dd8452", edgecolor="black", label="Sparsity (%)")
    ax_c.set_xticks(x); ax_c.set_xticklabels(cfgs, rotation=15)
    ax_c.set_ylabel("$L_0$ (mean active)", color="#4c72b0")
    ax_c2.set_ylabel("Sparsity (%)", color="#dd8452")
    ax_c.set_yscale("log")
    ax_c.set_title("(c) Sparsity vs active-feature count")
    ax_c.grid(axis="y", alpha=0.3, which="both")
    # joint legend
    lines = [bars1, bars2]
    labels = [b.get_label() for b in lines]
    ax_c.legend(lines, labels, fontsize=7, loc="upper left")

    # (d) Motif enrichment: bar chart of top TF-family matches.
    ax_d = fig.add_subplot(gs[1, 0])
    if mot_best.exists():
        import pandas as pd
        df = pd.read_csv(mot_best)
        sig = df[df["q"] < 0.05]
        top_tfs = sig["motif_alt"].value_counts().head(12)
        ax_d.barh(range(len(top_tfs))[::-1], top_tfs.values, color="#2ca02c",
                  edgecolor="black")
        ax_d.set_yticks(range(len(top_tfs))[::-1])
        ax_d.set_yticklabels(top_tfs.index, fontsize=8)
        ax_d.set_xlabel("# SAE features matched (q<0.05)")
        ax_d.set_title(
            f"(d) Top JASPAR plantae TF matches\n"
            f"({len(sig)}/{len(df)} features q<0.05)"
        )
        ax_d.grid(axis="x", alpha=0.3)
    else:
        ax_d.text(0.5, 0.5, "(motif enrichment CSV missing)",
                  ha="center", va="center", transform=ax_d.transAxes)

    # (e) Region-class enrichment breakdown
    ax_e = fig.add_subplot(gs[1, 1])
    if ann_best.exists():
        import pandas as pd
        df = pd.read_csv(ann_best)
        sig = df[df["q"] < 0.05]
        classes = ["exon", "intron", "utr3", "utr5", "intergenic"]
        counts = [int((sig["region"] == c).sum()) for c in classes]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        ax_e.bar(classes, counts, color=colors, edgecolor="black")
        for i, c in enumerate(counts):
            ax_e.text(i, c + 2, str(c), ha="center", fontsize=9)
        ax_e.set_ylabel("# features (q<0.05)")
        ax_e.set_title(
            f"(e) Region-class enrichment\n"
            f"({len(sig)}/{len(df)} features significantly enriched)"
        )
        ax_e.grid(axis="y", alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, "(annotation enrichment CSV missing)",
                  ha="center", va="center", transform=ax_e.transAxes)

    # (f) Fold-enrichment vs q, one dot per feature.
    ax_f = fig.add_subplot(gs[1, 2])
    if ann_best.exists():
        import pandas as pd
        df = pd.read_csv(ann_best)
        for c, col in zip(["exon", "intron", "utr3", "utr5", "intergenic"],
                          ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
            sub = df[df["region"] == c]
            ax_f.scatter(sub["fold_enrichment"], -np.log10(sub["q"] + 1e-300),
                         s=10, alpha=0.6, c=col, label=c)
        ax_f.axhline(-np.log10(0.05), color="gray", lw=0.8, ls="--", label="q=0.05")
        ax_f.set_xlabel("Fold enrichment over base rate")
        ax_f.set_ylabel("$-\\log_{10}$ q")
        ax_f.set_title("(f) Region-class enrichment volcano")
        ax_f.legend(fontsize=7)
        ax_f.grid(alpha=0.3)

    fig.suptitle(
        "Sparse autoencoder analysis on layer-7 activations (AraReg-RegionType)",
        y=0.995, fontsize=11,
    )
    out = OUT / "sae_training.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    for ds in ("region_type", "splice", "tss", "promoter"):
        fig_probing_with_baselines(ds)
    fig_baselines_comparison(["region_type", "splice", "tss", "promoter"])
    fig_cross_species()
    fig_patching()
    fig_head_ablations()
    fig_sae()


if __name__ == "__main__":
    main()
