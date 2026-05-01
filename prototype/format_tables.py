"""
Format multi-seed results as markdown tables for the paper.

Reads:
  ablation_results.json       (from ablation.py)
  lm_experiment_results.json  (from lm_experiment.py)

Emits markdown for:
  - Table 5 (full 2³ ablation, with mean ± std)
  - Table 6 (LM results, with mean ± std)
  - The β-anticipation interaction mini-table for §3.3
  - Inline-number block for abstract / conclusion

Usage (run from this directory):
  uv run python format_tables.py
"""

import json
import os
import sys

HERE = os.path.dirname(__file__)
ABLATION_JSON = os.path.join(HERE, "ablation_results.json")
LM_JSON       = os.path.join(HERE, "lm_experiment_results.json")


def load(path):
    if not os.path.exists(path):
        print(f"Missing: {path}", file=sys.stderr)
        print("  Run the corresponding experiment first.", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def fmt(mean, std, prec=3, plus=False):
    sign = "+" if plus and mean >= 0 else ""
    return f"{sign}{mean:.{prec}f} ± {std:.{prec}f}"


def emit_table5(ab):
    print("### Table 5 — Full 2³ ablation (mean ± std across seeds)")
    print()
    n_seeds = ab["results"]["baseline"]["n_seeds"]
    print(f"*{n_seeds} seeds, sample std (ddof=1). Gain is paired per seed against baseline.*")
    print()
    print("| Condition | acc (all) | acc@transition | Δ vs baseline |")
    print("|---|---|---|---|")

    rows = ["baseline", "β only", "Π only", "Ant only",
            "β + Π", "β + Ant", "Π + Ant", "β + Π + Ant (full)", "oracle"]
    for name in rows:
        s = ab["results"][name]
        all_str   = fmt(s["acc_all_mean"],   s["acc_all_std"])
        trans_str = fmt(s["acc_trans_mean"], s["acc_trans_std"])
        if name == "baseline":
            gain_str = "—"
        else:
            gain_str = fmt(s["gain_trans_mean"], s["gain_trans_std"], plus=True)
        # Highlight β+Ant and oracle in the markdown
        bold = name in {"β + Ant"}
        if bold:
            print(f"| **{name}** | **{all_str}** | **{trans_str}** | **{gain_str}** |")
        else:
            print(f"| {name} | {all_str} | {trans_str} | {gain_str} |")
    print()


def emit_mini_interaction(ab):
    print("### Mini-table for §3.3 — β-anticipation interaction")
    print()
    print("| Condition | acc@transition | gain |")
    print("|---|---|---|")
    for name in ("baseline", "β only", "Ant only", "β + Ant"):
        s = ab["results"][name]
        trans_str = fmt(s["acc_trans_mean"], s["acc_trans_std"])
        gain_str  = "—" if name == "baseline" else fmt(s["gain_trans_mean"], s["gain_trans_std"], plus=True)
        bold = name == "β + Ant"
        if bold:
            print(f"| **{name}** | **{trans_str}** | **{gain_str}** |")
        else:
            print(f"| {name} | {trans_str} | {gain_str} |")
    print()
    sa = ab["super_additivity"]
    print(f"Super-additive interaction: **{fmt(sa['interaction_mean'], sa['interaction_std'], plus=True)}** "
          f"(paired across {len(sa['interaction_runs'])} seeds)")
    print()


def emit_table6(lm):
    print("### Table 6 — Language model results (mean ± std across seeds)")
    print()
    n_seeds = lm["results"]["Standard MoE"]["n_seeds"]
    print(f"*{n_seeds} seeds, sample std (ddof=1).*")
    print()
    print("| Model | BPC (all) | BPC (trans) | p_B@trans | p_B@mid | K (99%) |")
    print("|---|---|---|---|---|---|")
    for name in ("Standard MoE", "β-MoE", "β+Ant MoE"):
        s = lm["results"][name]
        bpc_a = fmt(s["bpc_all_mean"],   s["bpc_all_std"])
        bpc_t = fmt(s["bpc_trans_mean"], s["bpc_trans_std"])
        pc_t  = fmt(s["pc_trans_mean"],  s["pc_trans_std"])
        pc_m  = fmt(s["pc_mid_mean"],    s["pc_mid_std"])
        k_str = fmt(s["K_needed_mean"],  s["K_needed_std"], prec=1)
        bold = name == "β+Ant MoE"
        if bold:
            print(f"| **{name}** | **{bpc_a}** | **{bpc_t}** | **{pc_t}** | **{pc_m}** | **{k_str}** |")
        else:
            print(f"| {name} | {bpc_a} | {bpc_t} | {pc_t} | {pc_m} | {k_str} |")
    print()


def emit_inline_numbers(ab, lm):
    print("### Inline numbers for abstract / conclusion")
    print()
    sa = ab["super_additivity"]

    def gain(name):
        s = ab["results"][name]
        return s["gain_trans_mean"], s["gain_trans_std"]

    bm, bs   = gain("β only")
    am, as_  = gain("Ant only")
    bam, bas = gain("β + Ant")

    print(f"- **β alone gain**:        {fmt(bm,  bs,  plus=True)}")
    print(f"- **Ant alone gain**:      {fmt(am,  as_, plus=True)}")
    print(f"- **β + Ant gain**:        {fmt(bam, bas, plus=True)}")
    print(f"- **Super-additive**:      {fmt(sa['interaction_mean'], sa['interaction_std'], plus=True)} "
          f"(β+Ant minus sum of individuals)")

    if lm is not None:
        s_std = lm["results"]["Standard MoE"]
        s_b   = lm["results"]["β-MoE"]
        s_ba  = lm["results"]["β+Ant MoE"]
        print()
        print(f"- **LM BPC(trans)**: Standard {fmt(s_std['bpc_trans_mean'], s_std['bpc_trans_std'])} "
              f"→ β-MoE {fmt(s_b['bpc_trans_mean'], s_b['bpc_trans_std'])} "
              f"→ β+Ant {fmt(s_ba['bpc_trans_mean'], s_ba['bpc_trans_std'])}")
        print(f"- **LM p_B@trans**:  Standard {fmt(s_std['pc_trans_mean'], s_std['pc_trans_std'])} "
              f"→ β-MoE {fmt(s_b['pc_trans_mean'], s_b['pc_trans_std'])} "
              f"→ β+Ant {fmt(s_ba['pc_trans_mean'], s_ba['pc_trans_std'])}")
    print()


def main():
    ab = load(ABLATION_JSON)
    lm = load(LM_JSON)
    if ab is None and lm is None:
        sys.exit(1)

    print("=" * 72)
    print("Markdown tables for paper (paste into ../affinity_is_not_enough.md)")
    print("=" * 72)
    print()

    if ab is not None:
        emit_table5(ab)
        emit_mini_interaction(ab)
    if lm is not None:
        emit_table6(lm)
    if ab is not None or lm is not None:
        emit_inline_numbers(ab, lm)


if __name__ == "__main__":
    main()
