"""Report figures from outputs/validation/*.csv. One house style; every figure titled with its takeaway.

Usage:
    uv run python -m pdval.report.figures
Writes report/figures/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
V = Path("outputs/validation")
OUT = Path("report/figures")
COL = {"baseline": "#9aa0a6", "challenger": "#1f77b4", "champion": "#d62728"}
LABEL = {"baseline": "FICO-only baseline", "challenger": "WoE scorecard (challenger)", "champion": "LightGBM (champion)"}
ROLE_ORDER = ["stress", "train", "tune", "oot", "covid_exhibit"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "figure.facecolor": "white",
})


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def fig_timeline():
    """The centrepiece: AUC holds across regimes, calibration does not."""
    s = pd.read_csv(V / "stability_by_vintage.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    for m in ("challenger", "champion"):
        ax.plot(s["vintage"], s[f"auc_{m}"], marker="o", color=COL[m], label=LABEL[m])
    ax.set_ylabel("AUC (36-month default)")
    ax.set_ylim(0.70, 0.90)
    ax.set_title("Rank-ordering survives every regime: AUC 0.77–0.88 across 2006–2019 vintages")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.bar(s["vintage"] - 0.2, s["obs_rate"] * 100, width=0.4, color="#333333", label="Observed 36m default rate")
    ax.bar(s["vintage"] + 0.2, s["mean_pred_champion"] * 100, width=0.4, color=COL["champion"], alpha=0.8, label="Champion mean predicted PD")
    ax.set_ylabel("36-month default rate (%)")
    ax.set_xlabel("Origination vintage")
    st = s[s["role"] == "stress"]; ratios = st["obs_rate"] / st["mean_pred_champion"]
    ax.set_title(f"…but the PD level does not: crisis vintages default {ratios.min():.1f}–{ratios.max():.1f}× what the model predicts")
    ax.legend(loc="center right")
    ax.set_ylim(0, max(s["obs_rate"].max(), s["mean_pred_champion"].max()) * 100 * 1.15)
    for role, (lo, hi) in {"train": (2009, 2015), "tune": (2016, 2016), "OOT": (2017, 2018)}.items():
        ax.axvspan(lo - 0.5, hi + 0.5, color={"train": "#dddddd", "tune": "#c9dcf0", "OOT": "#f6d5c9"}[role], alpha=0.35, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.93, role, ha="center", fontsize=8, color="#555555")
    ax.text(2007, ax.get_ylim()[1] * 0.93, "stress", ha="center", fontsize=8, color="#555555")
    ax.text(2019, ax.get_ylim()[1] * 0.93, "COVID", ha="center", fontsize=8, color="#555555")
    save(fig, "fig1_timeline.png")


def fig_auc_ci():
    d = pd.read_csv(V / "discrimination_by_role.csv")
    d["role"] = pd.Categorical(d["role"], ROLE_ORDER)
    d = d.sort_values(["role", "model"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(ROLE_ORDER))
    for i, m in enumerate(("baseline", "challenger", "champion")):
        g = d[d["model"] == m]
        ax.errorbar(x + (i - 1) * 0.22, g["auc"], yerr=[g["auc"] - g["auc_ci_low"], g["auc_ci_high"] - g["auc"]],
                    fmt="o", color=COL[m], label=LABEL[m], capsize=3)
    ax.set_xticks(x, ["stress\n2006–08", "train\n2009–15", "tune\n2016", "OOT\n2017–18", "COVID\n2019"])
    ax.set_ylabel("AUC (95% bootstrap CI)")
    ax.axhline(0.70, color="k", ls=":", lw=1); ax.text(4.35, 0.703, "floor 0.70", fontsize=7.5, ha="right")
    ax.set_title("Champion leads the scorecard by ~0.007 AUC out of time (DeLong p < 0.001); both beat FICO-only by ~0.05–0.06")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig2_auc_by_role.png")


def fig_reliability():
    r = pd.read_csv(V / "reliability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, role, title in zip(axes, ["oot", "stress", "covid_exhibit"],
                               ["OOT 2017–18: on the diagonal", "Crisis 2006–08: every decile above it",
                                "COVID 2019: below it (relief suppressed defaults)"]):
        for m in ("challenger", "champion"):
            g = r[(r["model"] == m) & (r["role"] == role)]
            ax.errorbar(g["mean_pred"] * 100, g["obs_rate"] * 100,
                        yerr=[(g["obs_rate"] - g["obs_ci_low"]) * 100, (g["obs_ci_high"] - g["obs_rate"]) * 100],
                        fmt="o-", ms=3.5, color=COL[m], label=LABEL[m], capsize=2)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Mean predicted PD (%) by decile"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Observed default rate (%)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Calibration holds out of time, fails under stress, and is not testable on the pandemic cohort", fontsize=10.5, fontweight="bold")
    save(fig, "fig3_reliability.png")


def fig_segments():
    s = pd.read_csv(V / "segments_oot.csv")
    s = s[(s["model"] == "champion") & (s["defaults"] >= 30)]
    dims = ["fico_band", "ltv_band", "dti_band", "loan_purpose", "occupancy", "channel", "region", "metro", "vintage"]
    s = s[s["dimension"].isin(dims)].copy()
    s["label"] = s["dimension"].str.replace("_band", "") + " = " + s["segment"].astype(str)
    s = s.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.errorbar(s["auc"], y, xerr=[s["auc"] - s["auc_ci_low"], s["auc_ci_high"] - s["auc"]], fmt="o", color=COL["champion"], capsize=2)
    ax.set_yticks(y, s["label"], fontsize=7.5)
    ax.axvline(0.65, color="k", ls=":", lw=1); ax.text(0.651, len(s) - 0.5, "segment floor 0.65", fontsize=7, va="top")
    ax.set_xlabel("Champion AUC within segment, OOT 2017–18 (95% CI; segments with ≥30 defaults)")
    ax.set_title("Within-segment ranking is weakest inside the sub-660 FICO band and for investor loans")
    save(fig, "fig4_segments.png")


def fig_psi_csi():
    s = pd.read_csv(V / "stability_by_vintage.csv")
    c = pd.read_csv(V / "csi_by_vintage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.bar(s["vintage"], s["psi_champion"], color=[COL["champion"] if r in ("oot", "covid_exhibit") else "#888888" for r in s["role"]])
    ax.axhline(0.10, color="orange", ls="--", lw=1); ax.axhline(0.25, color="red", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylabel("Score PSI vs training (log)"); ax.set_xlabel("Vintage")
    ax.set_title(f"Score PSI: 2018 red ({s.loc[s['vintage'] == 2018, 'psi_champion'].iloc[0]:.2f}); crisis > 1.5", fontsize=9.5)
    ax = axes[1]
    piv = c[c["vintage"].isin([2016, 2017, 2018, 2019])].pivot(index="variable", columns="vintage", values="csi")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    im = ax.imshow(piv.to_numpy(), cmap="Reds", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7.5)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black" if piv.iloc[i, j] < 0.3 else "white")
    ax.set_title("Input CSI vs training: 2018 shift is rate and purpose, not credit quality", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "fig5_stability.png")


def fig_sensitivity():
    pi = pd.read_csv(V / "perm_importance_oot.csv").head(10)
    sh = pd.read_csv(V / "shap_importance_oot.csv").head(10)
    oat = pd.read_csv(V / "oat_oot.csv")
    pdp = pd.read_csv(V / "pdp_oot.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    ax = axes[0, 0]
    ax.barh(pi["variable"][::-1], pi["auc_drop"][::-1], color=COL["champion"])
    ax.set_xlabel("AUC drop when permuted (OOT)"); ax.set_title("FICO carries most of the ranking", fontsize=9.5)
    ax = axes[0, 1]
    ax.barh(sh["variable"][::-1], sh["mean_abs_shap"][::-1], color="#555555")
    ax.set_xlabel("mean |SHAP| (log-odds), OOT sample"); ax.set_title("SHAP agrees on the top drivers", fontsize=9.5)
    ax = axes[1, 0]
    for v, g in pdp[pdp["variable"].isin(["fico", "ltv", "dti"])].groupby("variable"):
        xx = (g["value"] - g["value"].min()) / (g["value"].max() - g["value"].min())
        ax.plot(xx, g["mean_pred"] * 100, marker="o", ms=3, label=v)
    ax.set_xlabel("Input (rescaled 0–1 across its grid)"); ax.set_ylabel("Mean predicted PD (%)")
    ax.set_title("Partial dependence: monotone as constrained", fontsize=9.5); ax.legend()
    ax = axes[1, 1]
    o = oat[oat["variable"] != "(none)"].copy()
    o["label"] = o["variable"] + " " + o["shock"].map(lambda d: f"{d:+g}")
    ax.barh(o["label"][::-1], o["rel_change"][::-1] * 100, color=COL["champion"])
    ax.set_xlabel("Change in portfolio mean PD (%)"); ax.set_title("One-at-a-time shocks: −50 FICO ≈ doubles PD", fontsize=9.5)
    save(fig, "fig6_sensitivity.png")


def fig_deciles():
    d = pd.read_csv(V / "deciles_oot.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for m in ("baseline", "challenger", "champion"):
        g = d[d["model"] == m]
        ax.plot([0] + list(g["cum_pop"]), [0] + list(g["cum_capture"]), marker="o", ms=3, color=COL[m], label=LABEL[m])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Share of loans (riskiest first)"); ax.set_ylabel("Share of defaults captured")
    top = d[(d["model"] == "champion") & (d["decile"] == 1)]["cum_capture"].iloc[0]
    ax.set_title(f"Champion's riskiest decile captures {top:.0%} of OOT defaults")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig7_capture.png")


def main():
    fig_timeline(); fig_auc_ci(); fig_reliability(); fig_segments(); fig_psi_csi(); fig_sensitivity(); fig_deciles()
    print("figures ->", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
