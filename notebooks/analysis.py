# %% [markdown]
# # Consolidated analysis for paper/thesis
#
# Read-only notebook: loads pre-generated prediction CSVs and produces
# LaTeX tables (.tex) and vector figures (.pdf) for the paper.
#
# Sections:
# 0. Setup + data
# 1. Helper functions
# 2. Axis 1 — Central finding (summary ≈ full text)
# 3. Axis 2 — Model comparison
# 4. Axis 3 — Inter-model agreement
# 5. McNemar (significance of differences)
# 6. Confusion matrix
# 7. Near-miss analysis ("aimed close")
# 8. Closing

# %% Section 0 — Setup
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from statsmodels.stats.contingency_tables import mcnemar

from newsclf.analysis import compute_top1_labels, load_result_csvs
from newsclf.config import LABELS_CNN18, PREDICTIONS_DIR
from newsclf.metrics import compute_adherence

# --- Publication-quality matplotlib defaults ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    }
)

PALETTE = ["#333333", "#777777", "#BBBBBB"]

# --- Output directory ---
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Display labels ---
INPUT_DISPLAY: dict[str, str] = {
    "full_text": "Full text",
    "extractive_summary": "Extractive summary",
    "abstractive_summary": "Abstractive summary",
    "summary_chatgpt": "ChatGPT summary",
    "summary_deepseek": "DeepSeek summary",
    "summary_gemini": "Gemini summary",
}

PROVIDER_DISPLAY: dict[str, str] = {
    "chatgpt": "ChatGPT",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
}

PROVIDERS = ["chatgpt", "deepseek", "gemini"]
COMMON_INPUTS = ["full_text", "extractive_summary", "abstractive_summary"]
FOURTH_INPUT = {
    "chatgpt": "summary_chatgpt",
    "deepseek": "summary_deepseek",
    "gemini": "summary_gemini",
}

# --- Capitalisation normalisation (human classes) ---
CASE_FIX = {
    "Government and politics": "Government and Politics",
    "Society and culture": "Society and Culture",
}
HUMAN_COLS = [
    "Human1Class1", "Human1Class2", "Human1Class3",
    "Hum2Class1", "Hum2Class2", "Hum2Class3",
    "Hum3Class1", "Hum3Class2", "Hum3Class3",
]

print("Setup complete.")

# %% Section 0 — Load data

# Answer key
answer_key = pd.read_csv(
    PREDICTIONS_DIR.parent.parent / "data" / "answer_key" / "answer_key.csv"
)
print(f"Answer key: {len(answer_key)} articles")

# Results
df_single, df_percent = load_result_csvs()
assert df_single is not None, "No single CSV found"
assert df_percent is not None, "No percent CSV found"

# Add top1_label to percent
df_percent = compute_top1_labels(df_percent)

# Raw annotations from 3 annotators
df_raw = pd.read_excel(
    PREDICTIONS_DIR.parent.parent / "data" / "db3" / "classes.xlsx",
    sheet_name="alldiv",
)
df_raw["uid"] = df_raw["News"].apply(
    lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
)
for col in HUMAN_COLS:
    df_raw[col] = df_raw[col].map(lambda x: CASE_FIX.get(x, x) if isinstance(x, str) else x)

# Report counts
print(f"Raw annotations: {len(df_raw)} articles")
print("\n--- Count by model × input × method ---")
print("SINGLE:")
print(df_single.groupby(["provider", "input_type"]).size().to_string())
print("\nPERCENT:")
print(df_percent.groupby(["provider", "input_type"]).size().to_string())

# %% Section 1 — Helper functions


def resolve_y_true(
    row: pd.Series, pred_col: str = "predicted_label"
) -> str | float:
    """Resolve gold label considering pipe-separated ties.

    If the prediction is among the tied labels, use it as y_true
    (counts as correct). Otherwise, use the first label.
    """
    pred = row[pred_col]
    gold = str(row["gold_label"])
    options = gold.split("|")
    if pred in options:
        return pred
    return options[0]


def resolve_percent_ties(pred_tied: str, gold: str) -> tuple[str, str]:
    """Resolve both prediction ties (percent) and gold label ties.

    When the percent prediction has multiple categories tied at the
    maximum percentage, the resolved pred/true pair is:
    - If any category appears in BOTH tied sets → use it as both
      y_pred and y_true (counts as correct).
    - Otherwise → y_pred = first predicted tied (alphabetical),
      y_true = first gold tied (alphabetical).

    Returns (y_true, y_pred).
    """
    pred_options = pred_tied.split("|")
    gab_options = gold.split("|")
    overlap = set(pred_options) & set(gab_options)
    if overlap:
        label = sorted(overlap)[0]
        return label, label
    return gab_options[0], pred_options[0]


def compute_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return accuracy, kappa, and f1_macro for aligned Series."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def prepare_eval_df(
    df: pd.DataFrame,
    answer_key: pd.DataFrame,
    pred_col: str = "predicted_label",
) -> pd.DataFrame:
    """Merge predictions with answer key and resolve y_true.

    For single-label data (pred_col='predicted_label'): resolves
    gold label ties only (if pred ∈ gold options, count as correct).

    For percent data (pred_col='top1_label'): if 'top1_tied' column
    exists, resolves BOTH prediction ties and gold label ties
    symmetrically.
    """
    valid_labels = set(LABELS_CNN18)
    merged = df.merge(answer_key[["uid", "gold_label"]], on="uid", how="inner")

    # For percent data with tied predictions, resolve symmetrically.
    # Exclude large ties (top1_valid == False) as non-responses.
    if pred_col == "top1_label" and "top1_tied" in merged.columns:
        merged = merged[merged[pred_col].isin(valid_labels)].copy()
        if "top1_valid" in merged.columns:
            merged = merged[merged["top1_valid"]].copy()
        resolved = [
            resolve_percent_ties(tied, gab)
            for tied, gab in zip(merged["top1_tied"], merged["gold_label"])
        ]
        merged["y_true"] = [r[0] for r in resolved]
        merged["y_pred_resolved"] = [r[1] for r in resolved]
        return merged

    # Single-label: resolve gold label ties only
    merged = merged[merged[pred_col].isin(valid_labels)].copy()
    merged["y_true"] = merged.apply(resolve_y_true, axis=1, pred_col=pred_col)
    return merged


def metrics_by_group(
    df: pd.DataFrame,
    answer_key: pd.DataFrame,
    pred_col: str = "predicted_label",
) -> pd.DataFrame:
    """Compute metrics per (provider, input_type) group."""
    merged = prepare_eval_df(df, answer_key, pred_col)
    # Use resolved pred column for percent data if available
    actual_pred_col = (
        "y_pred_resolved" if "y_pred_resolved" in merged.columns else pred_col
    )
    rows = []
    for (prov, itype), grp in merged.groupby(["provider", "input_type"]):
        m = compute_all_metrics(grp["y_true"], grp[actual_pred_col])
        m["provider"] = prov
        m["input_type"] = itype
        m["n"] = len(grp)
        rows.append(m)
    return pd.DataFrame(rows)


def mcnemar_paired(
    df: pd.DataFrame,
    filter_a: dict,
    filter_b: dict,
) -> dict:
    """Run McNemar test between two subsets on shared uids.

    filter_a / filter_b: dicts of column→value for filtering df.
    Returns dict with b, c, statistic, p_value, n.
    """
    mask_a = pd.Series(True, index=df.index)
    for col, val in filter_a.items():
        mask_a &= df[col] == val
    mask_b = pd.Series(True, index=df.index)
    for col, val in filter_b.items():
        mask_b &= df[col] == val

    sub_a = df[mask_a][["uid", "correct"]].rename(columns={"correct": "correct_a"})
    sub_b = df[mask_b][["uid", "correct"]].rename(columns={"correct": "correct_b"})
    merged = sub_a.merge(sub_b, on="uid")

    # Discordance table
    b = int(((merged["correct_a"]) & (~merged["correct_b"])).sum())  # A right, B wrong
    c = int(((~merged["correct_a"]) & (merged["correct_b"])).sum())  # A wrong, B right

    # Build 2x2 contingency table for statsmodels
    # [[both_right, b], [c, both_wrong]]
    both_right = int((merged["correct_a"] & merged["correct_b"]).sum())
    both_wrong = int((~merged["correct_a"] & ~merged["correct_b"]).sum())
    table = np.array([[both_right, b], [c, both_wrong]])

    # Use exact test when b+c < 25, otherwise chi-square approximation
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact)

    return {
        "b": b,
        "c": c,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n": len(merged),
    }


def build_reference_sets(
    df_raw: pd.DataFrame, answer_key: pd.DataFrame
) -> pd.DataFrame:
    """Build strict, intermediate, and weighted reference sets per uid."""
    rows = []
    for _, r in df_raw.iterrows():
        uid = r["uid"]

        # Strict: top-1 of each annotator (Class1 of each Human)
        top1_cols = ["Human1Class1", "Hum2Class1", "Hum3Class1"]
        strict = {r[c] for c in top1_cols if pd.notna(r[c])}

        # Intermediate: all classes mentioned by any annotator
        intermediate = {r[c] for c in HUMAN_COLS if pd.notna(r[c])}

        rows.append(
            {
                "uid": uid,
                "ref_strict": strict,
                "ref_intermediate": intermediate,
            }
        )

    ref_df = pd.DataFrame(rows)

    # Weighted: classes with score >= 2 in ranking_json
    # Threshold = 2: a class needs to be cited by at least 2 annotators
    # (ou citada 2x pelo mesmo), excluindo menções isoladas.
    score_threshold = 2
    weighted_sets = []
    for _, r in answer_key.iterrows():
        ranking = json.loads(r["ranking_json"].replace("'", '"'))
        weighted = {cls for cls, score in ranking.items() if score >= score_threshold}
        weighted_sets.append({"uid": r["uid"], "ref_weighted": weighted})

    weighted_df = pd.DataFrame(weighted_sets)
    ref_df = ref_df.merge(weighted_df, on="uid", how="inner")

    return ref_df


# --- Build eval_single and eval_tc (used in sections 5, 6, 7) ---
eval_single = prepare_eval_df(df_single, answer_key)
eval_single["correct"] = eval_single["predicted_label"] == eval_single["y_true"]

# Reference sets for near-miss analysis
ref_sets = build_reference_sets(df_raw, answer_key)
print(f"Reference sets built: {len(ref_sets)} articles")

# Estatísticas descritivas dos conjuntos
for col_name, label in [
    ("ref_strict", "Strict (top-1 of 3 annotators)"),
    ("ref_intermediate", "Intermediate (all 9 positions)"),
    ("ref_weighted", "Weighted (score >= 2)"),
]:
    sizes = ref_sets[col_name].apply(len)
    print(f"  {label}: median {sizes.median():.0f}, "
          f"mean {sizes.mean():.1f}, min {sizes.min()}, max {sizes.max()}")

print("Helper functions defined.")

# %% Section 2 — Axis 1: Central finding (summary ≈ full text)
# %% [markdown]
# ## AXIS 1 — Central finding
#
# Central research question: "does classifying the summary have comparable
# performance to classifying the full text?"

# %% 2.1 — Impact of tie handling on percent top-1

# Count excluded by large tie
_n_excluded = int((~df_percent["top1_valid"]).sum())
print("=" * 60)
print("TIE-BREAKING RULE FOR PERCENT TOP-1")
print("=" * 60)
print(f"Large ties (k >= 6) excluded as invalid: {_n_excluded}")
for prov in PROVIDERS:
    _pmask = df_percent["provider"] == prov
    _pexcl = int((~df_percent.loc[_pmask, "top1_valid"]).sum())
    print(f"  {PROVIDER_DISPLAY[prov]:10s}: {_pexcl}")
print(f"n total before: {len(df_percent)}, after exclusion: {len(df_percent) - _n_excluded}")
del _n_excluded, _pmask, _pexcl

# %% 2.2 — Central finding table (single)

metrics_single = metrics_by_group(df_single, answer_key)

# Filter: for each provider, 4 inputs (3 common + 4th provider-specific)
rows_eixo1 = []
for prov in PROVIDERS:
    inputs = COMMON_INPUTS + [FOURTH_INPUT[prov]]
    sub = metrics_single[
        (metrics_single["provider"] == prov)
        & (metrics_single["input_type"].isin(inputs))
    ]
    rows_eixo1.append(sub)

tab_finding = pd.concat(rows_eixo1, ignore_index=True)
tab_finding["Model"] = tab_finding["provider"].map(PROVIDER_DISPLAY)
tab_finding["Input"] = tab_finding["input_type"].map(INPUT_DISPLAY)
tab_finding_display = tab_finding[
    ["Model", "Input", "accuracy", "kappa", "f1_macro", "n"]
].rename(columns={"accuracy": "Accuracy", "kappa": "Kappa", "f1_macro": "F1-Macro"})

# Format percentages
for col in ["Accuracy", "Kappa", "F1-Macro"]:
    tab_finding_display[col] = tab_finding_display[col].map(lambda x: f"{x:.4f}")

print(tab_finding_display.to_string(index=False))
tab_finding_display.to_latex(
    OUTPUT_DIR / "table_central_finding.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_central_finding.tex'}")

# %% 2.3 — Delta between inputs

# Also compute percent top-1 metrics
metrics_percent = metrics_by_group(df_percent, answer_key, pred_col="top1_label")

delta_rows = []
for metodo_label, mdf in [("single", metrics_single), ("percent top-1", metrics_percent)]:
    for prov in PROVIDERS:
        inputs = COMMON_INPUTS + [FOURTH_INPUT[prov]]
        sub = mdf[
            (mdf["provider"] == prov) & (mdf["input_type"].isin(inputs))
        ]
        if sub.empty:
            continue
        delta_rows.append(
            {
                "Model": PROVIDER_DISPLAY[prov],
                "delta_acc": sub["accuracy"].max() - sub["accuracy"].min(),
                "delta_kappa": sub["kappa"].max() - sub["kappa"].min(),
                "delta_f1": sub["f1_macro"].max() - sub["f1_macro"].min(),
                "Method": metodo_label,
            }
        )

tab_delta = pd.DataFrame(delta_rows)
for col in ["delta_acc", "delta_kappa", "delta_f1"]:
    tab_delta[col] = tab_delta[col].map(lambda x: f"{x:.4f}")

print(tab_delta.to_string(index=False))
tab_delta.to_latex(OUTPUT_DIR / "table_delta_inputs.tex", index=False, escape=False)
print(f"\nSaved: {OUTPUT_DIR / 'table_delta_inputs.tex'}")

# %% 2.4 — FIGURE central_finding.pdf (dot plot with range)

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
metric_cols = ["accuracy", "kappa", "f1_macro"]
metric_labels = ["Accuracy", "Kappa", "F1-Macro"]

for ax, mcol, mlabel in zip(axes, metric_cols, metric_labels):
    for i, prov in enumerate(PROVIDERS):
        inputs = COMMON_INPUTS + [FOURTH_INPUT[prov]]
        sub = metrics_single[
            (metrics_single["provider"] == prov)
            & (metrics_single["input_type"].isin(inputs))
        ]
        vals = sub[mcol].values
        y_pos = i
        # Range (span)
        ax.axhspan(
            y_pos - 0.15, y_pos + 0.15,
            xmin=0, xmax=1,
            color=PALETTE[i], alpha=0.15,
        )
        ax.hlines(y_pos, vals.min(), vals.max(), colors=PALETTE[i], linewidth=2)
        # Points for each input
        for j, (_, row) in enumerate(sub.iterrows()):
            marker = "o" if row["input_type"] in COMMON_INPUTS else "s"
            ax.plot(
                row[mcol], y_pos, marker=marker, color=PALETTE[i],
                markersize=7, zorder=5,
            )
    ax.set_xlabel(mlabel)
    ax.set_yticks(range(len(PROVIDERS)))
    ax.set_yticklabels([PROVIDER_DISPLAY[p] for p in PROVIDERS])
    ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "central_finding.pdf")
print(f"Saved: {OUTPUT_DIR / 'central_finding.pdf'}")
plt.show()

# %% Section 3 — Axis 2: Model comparison
# %% [markdown]
# ## AXIS 2 — Model comparison

# %% 3.1 — Model ranking table (3 common inputs)

# Single + Percent top-1, common inputs only
rank_rows = []
for metodo_label, mdf in [("single", metrics_single), ("percent top-1", metrics_percent)]:
    sub = mdf[mdf["input_type"].isin(COMMON_INPUTS)].copy()
    sub["Model"] = sub["provider"].map(PROVIDER_DISPLAY)
    sub["Input"] = sub["input_type"].map(INPUT_DISPLAY)
    sub["Method"] = metodo_label
    rank_rows.append(sub)

tab_ranking = pd.concat(rank_rows, ignore_index=True).sort_values(
    "kappa", ascending=False
)
tab_ranking_display = tab_ranking[
    ["Model", "Input", "Method", "accuracy", "kappa", "f1_macro", "n"]
].rename(columns={"accuracy": "Accuracy", "kappa": "Kappa", "f1_macro": "F1-Macro"})

for col in ["Accuracy", "Kappa", "F1-Macro"]:
    tab_ranking_display[col] = tab_ranking_display[col].map(lambda x: f"{x:.4f}")

print(tab_ranking_display.to_string(index=False))
tab_ranking_display.to_latex(
    OUTPUT_DIR / "table_model_ranking.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_model_ranking.tex'}")

# %% 3.2 — 4th input (separate)

fourth_rows = []
for metodo_label, mdf in [("single", metrics_single), ("percent top-1", metrics_percent)]:
    for prov in PROVIDERS:
        sub = mdf[
            (mdf["provider"] == prov)
            & (mdf["input_type"] == FOURTH_INPUT[prov])
        ]
        if sub.empty:
            continue
        row = sub.iloc[0].copy()
        fourth_rows.append(
            {
                "Model": PROVIDER_DISPLAY[prov],
                "Input": INPUT_DISPLAY[FOURTH_INPUT[prov]],
                "Method": metodo_label,
                "Accuracy": f"{row['accuracy']:.4f}",
                "Kappa": f"{row['kappa']:.4f}",
                "F1-Macro": f"{row['f1_macro']:.4f}",
                "n": int(row["n"]),
            }
        )

tab_fourth = pd.DataFrame(fourth_rows)
print("Note: each model classifies a different summary (generated by itself).")
print(tab_fourth.to_string(index=False))
tab_fourth.to_latex(
    OUTPUT_DIR / "table_fourth_input.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_fourth_input.tex'}")

# %% 3.3 — FIGURE model_comparison.pdf

fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(COMMON_INPUTS))
width = 0.25

for i, prov in enumerate(PROVIDERS):
    kappas = []
    for itype in COMMON_INPUTS:
        row = metrics_single[
            (metrics_single["provider"] == prov)
            & (metrics_single["input_type"] == itype)
        ]
        kappas.append(row["kappa"].values[0] if len(row) else 0)
    ax.bar(
        x + i * width, kappas, width,
        label=PROVIDER_DISPLAY[prov], color=PALETTE[i], edgecolor="black", linewidth=0.5,
    )

ax.set_ylabel("Kappa")
ax.set_xticks(x + width)
ax.set_xticklabels([INPUT_DISPLAY[it] for it in COMMON_INPUTS])
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "model_comparison.pdf")
print(f"Saved: {OUTPUT_DIR / 'model_comparison.pdf'}")
plt.show()

# %% 3.4 — Adherence rate

tab_aderencia = compute_adherence()

# Adjust percent: subtract large ties from valid count
_large_ties = (
    df_percent[~df_percent["top1_valid"]]
    .groupby(["provider", "input_type"])
    .size()
    .reset_index(name="n_large_tie")
)
tab_aderencia = tab_aderencia.merge(
    _large_ties, on=["provider", "input_type"], how="left"
)
tab_aderencia["n_large_tie"] = tab_aderencia["n_large_tie"].fillna(0).astype(int)
_is_pct = tab_aderencia["metodo"] == "percent"
tab_aderencia.loc[_is_pct, "validos"] = (
    tab_aderencia.loc[_is_pct, "validos"] - tab_aderencia.loc[_is_pct, "n_large_tie"]
)
tab_aderencia["taxa"] = tab_aderencia["validos"] / tab_aderencia["total"]
del _large_ties, _is_pct

tab_aderencia["Model"] = tab_aderencia["provider"].map(PROVIDER_DISPLAY)
tab_aderencia["Input"] = tab_aderencia["input_type"].map(INPUT_DISPLAY)
tab_aderencia["taxa"] = tab_aderencia["taxa"].map(lambda x: f"{x:.4f}")

tab_ader_display = tab_aderencia[
    ["Model", "Input", "metodo", "total", "validos", "taxa"]
].rename(columns={"metodo": "Method", "total": "Total", "validos": "Valid", "taxa": "Rate"})

print(tab_ader_display.to_string(index=False))
tab_ader_display.to_latex(
    OUTPUT_DIR / "table_adherence.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_adherence.tex'}")

# %% Section 4 — Axis 3: Inter-model agreement
# %% [markdown]
# ## AXIS 3 — Inter-model agreement

# %% 4.1 — Agreement table

# Prepare single predictions for the 3 common inputs
common_single = eval_single[eval_single["input_type"].isin(COMMON_INPUTS)]

pairs = [
    ("chatgpt", "deepseek"),
    ("chatgpt", "gemini"),
    ("deepseek", "gemini"),
]

agreement_rows = []
for itype in COMMON_INPUTS:
    sub = common_single[common_single["input_type"] == itype]
    for prov_a, prov_b in pairs:
        preds_a = sub[sub["provider"] == prov_a][["uid", "predicted_label"]].rename(
            columns={"predicted_label": "pred_a"}
        )
        preds_b = sub[sub["provider"] == prov_b][["uid", "predicted_label"]].rename(
            columns={"predicted_label": "pred_b"}
        )
        merged = preds_a.merge(preds_b, on="uid")
        if merged.empty:
            continue
        raw_agreement = (merged["pred_a"] == merged["pred_b"]).mean()
        kappa_inter = cohen_kappa_score(merged["pred_a"], merged["pred_b"])
        agreement_rows.append(
            {
                "Pair": f"{PROVIDER_DISPLAY[prov_a]} × {PROVIDER_DISPLAY[prov_b]}",
                "Input": INPUT_DISPLAY[itype],
                "Raw agreement": f"{raw_agreement:.4f}",
                "Inter-model Kappa": f"{kappa_inter:.4f}",
                "n": len(merged),
            }
        )

# Agreement of all 3 together
for itype in COMMON_INPUTS:
    sub = common_single[common_single["input_type"] == itype]
    pivoted = sub.pivot_table(
        index="uid", columns="provider", values="predicted_label", aggfunc="first"
    )
    pivoted = pivoted.dropna()
    if len(pivoted) > 0 and all(p in pivoted.columns for p in PROVIDERS):
        agree_3 = (
            (pivoted["chatgpt"] == pivoted["deepseek"])
            & (pivoted["deepseek"] == pivoted["gemini"])
        ).mean()
        agreement_rows.append(
            {
                "Pair": "3 models",
                "Input": INPUT_DISPLAY[itype],
                "Raw agreement": f"{agree_3:.4f}",
                "Inter-model Kappa": "—",
                "n": len(pivoted),
            }
        )

tab_agreement = pd.DataFrame(agreement_rows)
print(tab_agreement.to_string(index=False))
tab_agreement.to_latex(
    OUTPUT_DIR / "table_intermodel_agreement.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_intermodel_agreement.tex'}")

# %% 4.2 — FIGURE agreement_heatmap.pdf

# Heatmap 3×3: diagonal = kappa vs ground truth; off-diagonal = inter-model kappa
# Use full_text as reference input for the heatmap
itype_ref = "full_text"
sub_ref = common_single[common_single["input_type"] == itype_ref]

heatmap_data = np.zeros((3, 3))
model_labels = [PROVIDER_DISPLAY[p] for p in PROVIDERS]

# Diagonal: kappa vs ground truth
for i, prov in enumerate(PROVIDERS):
    grp = sub_ref[sub_ref["provider"] == prov]
    if len(grp) > 0:
        heatmap_data[i, i] = cohen_kappa_score(grp["y_true"], grp["predicted_label"])

# Off-diagonal: inter-model kappa
for idx_a, prov_a in enumerate(PROVIDERS):
    for idx_b, prov_b in enumerate(PROVIDERS):
        if idx_a >= idx_b:
            continue
        preds_a = sub_ref[sub_ref["provider"] == prov_a][
            ["uid", "predicted_label"]
        ].rename(columns={"predicted_label": "pred_a"})
        preds_b = sub_ref[sub_ref["provider"] == prov_b][
            ["uid", "predicted_label"]
        ].rename(columns={"predicted_label": "pred_b"})
        merged = preds_a.merge(preds_b, on="uid")
        if len(merged) > 0:
            k = cohen_kappa_score(merged["pred_a"], merged["pred_b"])
            heatmap_data[idx_a, idx_b] = k
            heatmap_data[idx_b, idx_a] = k

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(heatmap_data, cmap="Greys", vmin=0, vmax=1)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(model_labels)
ax.set_yticklabels(model_labels)

# Numeric annotations
for i in range(3):
    for j in range(3):
        val = heatmap_data[i, j]
        color = "white" if val > 0.5 else "black"
        label = "vs. gt" if i == j else "inter"
        ax.text(
            j, i, f"{val:.3f}\n({label})",
            ha="center", va="center", color=color, fontsize=9,
        )

fig.colorbar(im, ax=ax, shrink=0.8, label="Kappa")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "agreement_heatmap.pdf")
print(f"Saved: {OUTPUT_DIR / 'agreement_heatmap.pdf'}")
plt.show()

# %% 4.3 — Ambiguity context

print("=" * 60)
print("CONTEXT: Human ambiguity vs. LLM agreement")
print("=" * 60)
print()
print("Human agreement (3 annotators, CNN corpus):")
print("  ~27% full agreement among 3 annotators")
print()

# Agreement of 3 LLMs (full_text, single)
sub_tc = common_single[common_single["input_type"] == "full_text"]
pivoted_tc = sub_tc.pivot_table(
    index="uid", columns="provider", values="predicted_label", aggfunc="first"
).dropna()

if len(pivoted_tc) > 0 and all(p in pivoted_tc.columns for p in PROVIDERS):
    agree_3_tc = (
        (pivoted_tc["chatgpt"] == pivoted_tc["deepseek"])
        & (pivoted_tc["deepseek"] == pivoted_tc["gemini"])
    ).mean()
    print(f"Agreement of 3 LLMs (full text, single): {agree_3_tc:.4f}")

    # Mean inter-model kappa
    kappas_inter = []
    for prov_a, prov_b in pairs:
        k = cohen_kappa_score(pivoted_tc[prov_a], pivoted_tc[prov_b])
        kappas_inter.append(k)
        print(
            f"  Kappa {PROVIDER_DISPLAY[prov_a]} × {PROVIDER_DISPLAY[prov_b]}: {k:.4f}"
        )
    print(f"  Mean inter-model kappa: {np.mean(kappas_inter):.4f}")

print()
print("Interpretation: LLMs agree substantially more among themselves")
print("than human annotators, suggesting shared bias/contamination")
print("across language models.")

# %% Section 5 — McNemar
# %% [markdown]
# ## McNemar test
#
# McNemar's test compares two **paired** classifiers (same samples).
# Focuses on **discordant** cases: articles one gets right and the other wrong.
# The 2×2 discordance table has:
# - b = model A correct, model B wrong
# - c = model A wrong, model B correct
#
# H₀: b = c (both classifiers have the same performance).
#
# **Bonferroni correction**: when running multiple tests simultaneously,
# the risk of false positives (incorrectly rejecting H₀) increases.
# With k tests, multiply each p-value by k. This controls the
# family-wise error rate at α = 0.05.

# %% 5.1 — McNemar: between inputs (within each model)

mcnemar_input_rows = []
for prov in PROVIDERS:
    inputs = COMMON_INPUTS + [FOURTH_INPUT[prov]]
    for i, inp_a in enumerate(inputs):
        for inp_b in inputs[i + 1 :]:
            res = mcnemar_paired(
                eval_single,
                {"provider": prov, "input_type": inp_a},
                {"provider": prov, "input_type": inp_b},
            )
            mcnemar_input_rows.append(
                {
                    "Model": PROVIDER_DISPLAY[prov],
                    "Input A": INPUT_DISPLAY[inp_a],
                    "Input B": INPUT_DISPLAY[inp_b],
                    "b (A+B-)": res["b"],
                    "c (A-B+)": res["c"],
                    "Statistic": res["statistic"],
                    "Raw p": res["p_value"],
                    "n": res["n"],
                }
            )

tab_mcnemar_inputs = pd.DataFrame(mcnemar_input_rows)

# Bonferroni correction
n_tests_inputs = len(tab_mcnemar_inputs)
tab_mcnemar_inputs["Corrected p"] = (
    tab_mcnemar_inputs["Raw p"] * n_tests_inputs
).clip(upper=1.0)
tab_mcnemar_inputs["Sig."] = tab_mcnemar_inputs["Corrected p"].apply(
    lambda p: "*" if p < 0.05 else ""
)

print(f"McNemar between INPUTS — {n_tests_inputs} comparisons (Bonferroni k={n_tests_inputs})")
print(tab_mcnemar_inputs.to_string(index=False))

# %% 5.2 — McNemar: between models (on 3 common inputs)

model_pairs = [
    ("chatgpt", "deepseek"),
    ("chatgpt", "gemini"),
    ("deepseek", "gemini"),
]

mcnemar_model_rows = []
for itype in COMMON_INPUTS:
    for prov_a, prov_b in model_pairs:
        res = mcnemar_paired(
            eval_single,
            {"provider": prov_a, "input_type": itype},
            {"provider": prov_b, "input_type": itype},
        )
        mcnemar_model_rows.append(
            {
                "Input": INPUT_DISPLAY[itype],
                "Model A": PROVIDER_DISPLAY[prov_a],
                "Model B": PROVIDER_DISPLAY[prov_b],
                "b (A+B-)": res["b"],
                "c (A-B+)": res["c"],
                "Statistic": res["statistic"],
                "Raw p": res["p_value"],
                "n": res["n"],
            }
        )

tab_mcnemar_models = pd.DataFrame(mcnemar_model_rows)

# Bonferroni correction
n_tests_models = len(tab_mcnemar_models)
tab_mcnemar_models["Corrected p"] = (
    tab_mcnemar_models["Raw p"] * n_tests_models
).clip(upper=1.0)
tab_mcnemar_models["Sig."] = tab_mcnemar_models["Corrected p"].apply(
    lambda p: "*" if p < 0.05 else ""
)

print(f"\nMcNemar between MODELS — {n_tests_models} comparisons (Bonferroni k={n_tests_models})")
print(tab_mcnemar_models.to_string(index=False))

# %% 5.3 — Consolidated McNemar LaTeX table

# Combine both tables
tab_mcnemar_inputs["Type"] = "Between inputs"
tab_mcnemar_models["Type"] = "Between models"

# Normalize columns for export
export_inputs = tab_mcnemar_inputs.rename(
    columns={"Model": "Reference", "Input A": "A", "Input B": "B"}
)[
    ["Type", "Reference", "A", "B", "b (A+B-)", "c (A-B+)",
     "Statistic", "Raw p", "Corrected p", "Sig.", "n"]
]

export_models = tab_mcnemar_models.copy()
export_models["Reference"] = export_models["Input"]
export_models["A"] = export_models["Model A"]
export_models["B"] = export_models["Model B"]  # noqa: E501
export_models = export_models[
    ["Type", "Reference", "A", "B", "b (A+B-)", "c (A-B+)",
     "Statistic", "Raw p", "Corrected p", "Sig.", "n"]
]

tab_mcnemar_full = pd.concat([export_inputs, export_models], ignore_index=True)

# Format numeric columns
tab_mcnemar_full["Statistic"] = tab_mcnemar_full["Statistic"].map(lambda x: f"{x:.2f}")
tab_mcnemar_full["Raw p"] = tab_mcnemar_full["Raw p"].map(lambda x: f"{x:.4f}")
tab_mcnemar_full["Corrected p"] = tab_mcnemar_full["Corrected p"].map(lambda x: f"{x:.4f}")

tab_mcnemar_full.to_latex(
    OUTPUT_DIR / "table_mcnemar.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_mcnemar.tex'}")

# %% Section 6 — Confusion matrix
# %% [markdown]
# ## Confusion matrices
#
# An aggregated confusion matrix (3 models together) is more informative than
# 3 separate matrices: it reveals **shared error patterns** across LLMs,
# which are the most relevant finding (common bias). Individual matrices can be
# generated on demand, but the aggregate suffices for the paper.

# %% 6.1 — Aggregated confusion matrix (3 models, full text, single)

# Use full_text, single method, all 3 models
cm_data = eval_single[eval_single["input_type"] == "full_text"]
n_cm = len(cm_data)
print(f"Aggregated confusion matrix: {n_cm} predictions (3 models × full text)")

# Compute confusion matrix normalised by row (recall-style)
labels_order = sorted(cm_data["y_true"].unique())
cm_raw = confusion_matrix(cm_data["y_true"], cm_data["predicted_label"], labels=labels_order)
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

# %% 6.2 — FIGURE confusion_matrix.pdf

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)

n_labels = len(labels_order)
ax.set_xticks(range(n_labels))
ax.set_yticks(range(n_labels))
ax.set_xticklabels(labels_order, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels_order, fontsize=8)
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")

# Annotate cells — only show values >= 0.03 to avoid clutter
for i in range(n_labels):
    for j in range(n_labels):
        val = cm_norm[i, j]
        if val >= 0.03:
            color = "white" if val > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

fig.colorbar(im, ax=ax, shrink=0.7, label="Proportion (row-normalised)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.pdf")
print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.pdf'}")
plt.show()

# %% 6.3 — Strongest confusion patterns

print("=" * 60)
print("STRONGEST CONFUSION PATTERNS (aggregated, full text)")
print("=" * 60)
print()

# Collect off-diagonal confusions, sorted by proportion
confusions = []
for i in range(n_labels):
    for j in range(n_labels):
        if i == j:
            continue
        confusions.append(
            {
                "True": labels_order[i],
                "Predicted": labels_order[j],
                "Proportion": cm_norm[i, j],
                "Count": cm_raw[i, j],
            }
        )

tab_confusions = pd.DataFrame(confusions).sort_values("Proportion", ascending=False)
top_confusions = tab_confusions.head(15)

for _, row in top_confusions.iterrows():
    print(
        f"  {row['True']:30s} -> {row['Predicted']:30s}  "
        f"{row['Proportion']:.2%} ({int(row['Count'])} cases)"
    )

print()
print("Interpretation: the most confused pairs reveal semantic ambiguities")
print("in the corpus (e.g., Economy/Business, Law/Government).")
print("These patterns are consistent across the 3 LLMs, reinforcing")
print("the hypothesis of shared bias at category boundaries.")

# %% Section 7 — Near-miss analysis ("aimed close")
# %% [markdown]
# ## "Aimed close" — Errors within the human space
#
# Investigates whether, when the LLM misses the top-1 ground truth label,
# the predicted class still belongs to the set of categories that human
# annotators considered relevant. Hypothesis: errors concentrate on categories
# that humans also pointed out, reflecting task ambiguity.

# %% 7.1 — Near-miss metric per model

# Merge ref_sets with eval_single (full_text only)
eval_tc = eval_single.merge(ref_sets, on="uid", how="inner")
eval_tc = eval_tc[eval_tc["input_type"] == "full_text"].copy()

print(f"Full text evaluation: {len(eval_tc)} predictions")
print(f"  Correct: {eval_tc['correct'].sum()}")
print(f"  Errors: {(~eval_tc['correct']).sum()}")

# Compute metrics per model
results_rows = []
for prov in PROVIDERS:
    sub = eval_tc[eval_tc["provider"] == prov]
    n_total = len(sub)
    n_correct = int(sub["correct"].sum())
    n_errors = n_total - n_correct
    acc_top1 = n_correct / n_total

    errors = sub[~sub["correct"]]

    for ref_col, ref_label in [
        ("ref_strict", "Strict"),
        ("ref_intermediate", "Intermediate"),
        ("ref_weighted", "Weighted"),
    ]:
        n_near = int(
            errors.apply(
                lambda r, rc=ref_col: r["predicted_label"] in r[rc], axis=1
            ).sum()
        )
        pct_near = n_near / n_errors if n_errors else 0

        results_rows.append(
            {
                "Model": PROVIDER_DISPLAY[prov],
                "n": n_total,
                "Hits (top-1)": n_correct,
                "Errors": n_errors,
                "Acc top-1": acc_top1,
                "Definition": ref_label,
                "Near errors": n_near,
                "% near errors": pct_near,
                "Acc + near": (n_correct + n_near) / n_total,
            }
        )

df_results = pd.DataFrame(results_rows)

print("\n--- Results by model and definition ---")
for prov in PROVIDERS:
    prov_name = PROVIDER_DISPLAY[prov]
    sub = df_results[df_results["Model"] == prov_name]
    print(f"\n{prov_name} (n={sub.iloc[0]['n']}):")
    print(f"  Top-1 accuracy: {sub.iloc[0]['Acc top-1']:.4f}")
    for _, r in sub.iterrows():
        print(
            f"  {r['Definition']:15s}: {r['Near errors']:4d}/{r['Errors']:4d} near errors "
            f"({r['% near errors']:.2%})  |  acc+near = {r['Acc + near']:.4f}"
        )

# %% 7.2 — Comparative LaTeX table

# Pivot: one row per model, columns for each definition
tab_rows = []
for prov in PROVIDERS:
    prov_name = PROVIDER_DISPLAY[prov]
    sub = df_results[df_results["Model"] == prov_name]
    row: dict = {
        "Model": prov_name,
        "n": int(sub.iloc[0]["n"]),
        "Acc top-1": sub.iloc[0]["Acc top-1"],
    }
    for _, r in sub.iterrows():
        defn = r["Definition"]
        row[f"% near ({defn})"] = r["% near errors"]
        row[f"Acc+near ({defn})"] = r["Acc + near"]
    tab_rows.append(row)

tab_mira = pd.DataFrame(tab_rows)

# Format for display
tab_display = tab_mira.copy()
for col in tab_display.columns:
    if col in ("Model", "n"):
        continue
    tab_display[col] = tab_display[col].map(lambda x: f"{x:.4f}")

print("\n--- Comparative table ---")
print("Note: COMPLEMENTARY metric. 'Acc+near' does NOT replace top-1 accuracy.")
print("Shows how much of the error is plausible (within human annotation space).\n")
print(tab_display.to_string(index=False))

tab_display.to_latex(
    OUTPUT_DIR / "table_near_miss.tex", index=False, escape=False
)
print(f"\nSaved: {OUTPUT_DIR / 'table_near_miss.tex'}")

# %% 7.3 — FIGURE near_miss.pdf (stacked bars)

# Use "Intermediate" definition for the figure (most intuitive)
fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(PROVIDERS))
width = 0.5

hits = []
near = []
far = []

for prov in PROVIDERS:
    prov_name = PROVIDER_DISPLAY[prov]
    sub = df_results[
        (df_results["Model"] == prov_name)
        & (df_results["Definition"] == "Intermediate")
    ].iloc[0]
    n = sub["n"]
    hits.append(sub["Hits (top-1)"] / n)
    near.append(sub["Near errors"] / n)
    far.append((sub["Errors"] - sub["Near errors"]) / n)

hits_arr = np.array(hits)
near_arr = np.array(near)
far_arr = np.array(far)

ax.bar(x, hits_arr, width, label="Exact match (top-1)", color="#333333")
ax.bar(
    x, near_arr, width, bottom=hits_arr,
    label="Wrong, but within human space", color="#999999",
)
ax.bar(
    x, far_arr, width, bottom=hits_arr + near_arr,
    label="Wrong, outside human space", color="#DDDDDD", edgecolor="#AAAAAA",
)

ax.set_ylabel("Proportion")
ax.set_xticks(x)
ax.set_xticklabels([PROVIDER_DISPLAY[p] for p in PROVIDERS])
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

# Annotate percentages
for i in range(len(PROVIDERS)):
    # Correct
    ax.text(
        i, hits_arr[i] / 2, f"{hits_arr[i]:.1%}",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold",
    )
    # Near
    mid_near = hits_arr[i] + near_arr[i] / 2
    ax.text(
        i, mid_near, f"{near_arr[i]:.1%}",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold",
    )
    # Far
    mid_far = hits_arr[i] + near_arr[i] + far_arr[i] / 2
    ax.text(
        i, mid_far, f"{far_arr[i]:.1%}",
        ha="center", va="center", color="#555555", fontsize=9,
    )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "near_miss.pdf")
print(f"Saved: {OUTPUT_DIR / 'near_miss.pdf'}")
plt.show()

# %% Section 8 — Paper claims verification
# %% [markdown]
# ## Verification: invalid response rates per model
#
# Trecho do artigo:
# > "o ChatGPT produziu 38 rótulos fora da taxonomia no método single-label
# > e 138 respostas uniformes no método percent-distribution, enquanto no
# > DeepSeek e no Gemini a taxa de respostas inválidas não passou de 0,03%"

# %% 8.1 — Respostas inválidas (single + percent)

from newsclf.config import PREDICTIONS_DIR as _PRED_DIR  # noqa: E402

_valid_labels = set(LABELS_CNN18)

print("=" * 70)
print("VERIFICATION: invalid responses per model")
print("=" * 70)

# --- Single: labels outside taxonomy ---
print("\n--- Single-label: labels outside taxonomy ---")
_single_total = {}
_single_invalid = {}
for prov in PROVIDERS:
    _f = _PRED_DIR / prov / f"results_single_{prov}.csv"
    _df = pd.read_csv(_f)
    _inv = _df[~_df["predicted_label"].isin(_valid_labels)]
    _single_total[prov] = len(_df)
    _single_invalid[prov] = len(_inv)
    _pct = 100 * len(_inv) / len(_df)
    print(f"  {PROVIDER_DISPLAY[prov]:10s}: {len(_inv):3d} / {len(_df)} ({_pct:.2f}%)")

# --- Percent: uniform responses (tie >= 6, top1_valid == False) ---
print("\n--- Percent-distribution: uniform responses (tie k >= 6) ---")
_pct_total = {}
_pct_uniform = {}
for prov in PROVIDERS:
    _mask = df_percent["provider"] == prov
    _sub = df_percent[_mask]
    _n_uniform = int((~_sub["top1_valid"]).sum())
    _pct_total[prov] = len(_sub)
    _pct_uniform[prov] = _n_uniform
    _pct_rate = 100 * _n_uniform / len(_sub)
    print(f"  {PROVIDER_DISPLAY[prov]:10s}: {_n_uniform:3d} / {len(_sub)} ({_pct_rate:.2f}%)")

# --- Consolidated summary ---
print("\n--- Consolidated: total invalid responses ---")
for prov in PROVIDERS:
    _total = _single_total[prov] + _pct_total[prov]
    _inv_total = _single_invalid[prov] + _pct_uniform[prov]
    _rate = 100 * _inv_total / _total
    print(f"  {PROVIDER_DISPLAY[prov]:10s}: {_inv_total:3d} / {_total} ({_rate:.4f}%)")

# --- Claims verification ---
print("\n--- Claim verification ---")
_ok = True

if _single_invalid["chatgpt"] != 38:
    print(f"  DIVERGE: ChatGPT single invalid = {_single_invalid['chatgpt']} (paper says 38)")
    _ok = False
else:
    print(f"  OK: ChatGPT single invalid = {_single_invalid['chatgpt']}")

if _pct_uniform["chatgpt"] != 138:
    print(f"  DIVERGE: ChatGPT percent uniform = {_pct_uniform['chatgpt']} (paper says 138)")
    _ok = False
else:
    print(f"  OK: ChatGPT percent uniform = {_pct_uniform['chatgpt']}")

for prov in ["deepseek", "gemini"]:
    _total = _single_total[prov] + _pct_total[prov]
    _inv_total = _single_invalid[prov] + _pct_uniform[prov]
    _rate = 100 * _inv_total / _total
    if _rate > 0.03:
        print(f"  DIVERGE: {PROVIDER_DISPLAY[prov]} rate = {_rate:.4f}% (paper says <= 0.03%)")
        _ok = False
    else:
        print(f"  OK: {PROVIDER_DISPLAY[prov]} rate = {_rate:.4f}% (<= 0.03%)")

if _ok:
    print("\n  All claims match the data.")
else:
    print("\n  WARNING: divergences found — review the paper text.")

del _valid_labels, _single_total, _single_invalid, _pct_total, _pct_uniform
del _ok, _total, _inv_total, _rate, _f, _df, _inv, _pct, _mask, _sub
del _n_uniform, _pct_rate

# %% Section 9 — Closing

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)

expected_files = [
    # PDFs
    "central_finding.pdf",
    "model_comparison.pdf",
    "agreement_heatmap.pdf",
    "confusion_matrix.pdf",
    "near_miss.pdf",
    # LaTeX
    "table_central_finding.tex",
    "table_delta_inputs.tex",
    "table_model_ranking.tex",
    "table_fourth_input.tex",
    "table_adherence.tex",
    "table_intermodel_agreement.tex",
    "table_mcnemar.tex",
    "table_near_miss.tex",
]

all_ok = True
for fname in expected_files:
    fpath = OUTPUT_DIR / fname
    status = "OK" if fpath.exists() else "MISSING"
    if status == "MISSING":
        all_ok = False
    print(f"  [{status}] {fpath}")

print(f"\nTotal: {len(expected_files)} artifacts ({5} PDFs + {8} .tex)")
if all_ok:
    print("All artifacts generated successfully.")
