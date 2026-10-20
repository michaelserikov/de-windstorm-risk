"""Figures for the README (static PNG, light surface)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, DARK = "#2a78d6", "#eb6834", "#123f7a"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def seasons(season_tot: pd.DataFrame, ev: pd.DataFrame, out: Path) -> None:
    """Seasonal aggregate loss index, with the named storms labelled."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(season_tot.index, season_tot["sum"], color=BLUE, width=0.75)
    for _, r in ev[ev["storm"] != ""].iterrows():
        y = season_tot.loc[r["season"], "sum"]
        ax.text(r["season"], y + 0.006, r["storm"], rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_ylim(0, season_tot["sum"].max() * 1.35)
    _style(ax, "Windstorm loss index per winter (Oct-Mar), Germany, DWD stations", "loss index (sum of events)")
    ax.set_xlabel("winter, labelled by the year it ends", color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def ep_curves(sim: pd.DataFrame, season_tot: pd.DataFrame, boot: pd.DataFrame, aal: float, out: Path) -> None:
    """Exceedance-probability curves (aggregate and occurrence) against return period, with observed seasons."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rp = np.logspace(0.05, np.log10(500), 200)
    for col, colour, label in (("aggregate", BLUE, "Aggregate (AEP), model"), ("max_event", ORANGE, "Occurrence (OEP), model")):
        ax.plot(rp, [sim[col].quantile(1 - 1 / t) for t in rp], color=colour, linewidth=2, label=label)
        obs = np.sort(season_tot["sum" if col == "aggregate" else "max"].to_numpy())[::-1]
        emp_rp = (len(obs) + 1) / np.arange(1, len(obs) + 1)
        ax.scatter(emp_rp, obs, s=18, color=colour, edgecolor=SURFACE, zorder=3)
    lo, hi = boot["aep_200"].quantile([0.05, 0.95])
    ax.errorbar([200], [sim["aggregate"].quantile(0.995)], yerr=[[sim["aggregate"].quantile(0.995) - lo],
                [hi - sim["aggregate"].quantile(0.995)]], color=BLUE, capsize=4, fmt="none")
    ax.axhline(aal, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.15, aal, " AAL", va="bottom", fontsize=8, color=INK2)
    ax.set_xscale("log")
    _style(ax, "Exceedance-probability curves: dots are the 35 observed winters", "loss index")
    ax.set_xlabel("return period (winters)", color=INK2)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def footprints(prints: dict[str, pd.Series], stations: pd.DataFrame, out: Path) -> None:
    """Station maximum gust relative to the local 98th percentile, for two storms."""
    fig, axes = plt.subplots(1, len(prints), figsize=(8, 4.6))
    for ax, (name, fp) in zip(np.atleast_1d(axes), prints.items()):
        ratio = (fp / stations["v98"]).reindex(stations.index)
        sc = ax.scatter(stations["lon"], stations["lat"], c=ratio, cmap="RdYlBu_r", vmin=0.6, vmax=1.6, s=22,
                        edgecolor=GRID, linewidth=0.4)
        ax.set_title(name, loc="left", color=INK, fontsize=11)
        ax.set_aspect(1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_facecolor(SURFACE)
    fig.set_facecolor(SURFACE)
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04)
    cb.set_label("event maximum gust / local 98th percentile (damage above 1)", color=INK2)
    fig.suptitle("Storm footprints at DWD stations", x=0.02, ha="left", color=INK, fontsize=12)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --- v0.3: compound wind and rain ----------------------------------------------------------------

def chi_map(chi: pd.DataFrame, out: Path) -> None:
    """Station map of chi(0.98) = P(rain > r98 | gust > v98); 0.02 under independence."""
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    sc = ax.scatter(chi["lon"], chi["lat"], c=chi["chi"], cmap="viridis", vmin=0, vmax=0.2, s=30,
                    edgecolor=GRID, linewidth=0.4)
    ax.set_aspect(1.55)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_facecolor(SURFACE); fig.set_facecolor(SURFACE)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    cb.set_label("P(rain > 98th pct | gust > 98th pct), winter days; independence = 0.02", color=INK2, fontsize=8)
    ax.set_title("Wind and rain extremes on the same day", loc="left", color=INK, fontsize=12)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def wind_rain_scatter(events: pd.DataFrame, out: Path) -> None:
    """Storm events: wind loss index against rain measure, named storms labelled."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(events["loss_index"], events["rain_ratio"], s=10, color=BLUE, alpha=0.45, edgecolor="none")
    named = events[events["storm"].fillna("") != ""]
    ax.scatter(named["loss_index"], named["rain_ratio"], s=30, color=ORANGE, edgecolor=SURFACE, zorder=3)
    for _, r in named.iterrows():
        ax.annotate(r["storm"], (r["loss_index"], r["rain_ratio"]), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(left=1e-6)
    _style(ax, "Winter storm events: wind loss index vs rainfall during the event", "rain / local 98th percentile (weighted mean)")
    ax.set_xlabel("wind loss index (log scale)", color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def joint_rp(jrp: pd.DataFrame, out: Path) -> None:
    """Joint return period of events extreme in both wind and rain, by dependence model."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(jrp))
    width = 0.2
    for i, (col, colour, label) in enumerate((("joint_rp_independence", GRID, "Independence"),
                                               ("joint_rp_gaussian", BLUE, "Gaussian copula"),
                                               ("joint_rp_gumbel", DARK, "Gumbel copula"),
                                               ("joint_rp_empirical", ORANGE, "Observed"))):
        ax.bar(x + (i - 1.5) * width, jrp[col], width * 0.92, color=colour, label=label,
               edgecolor=INK2 if colour == GRID else "none", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{t}-winter level" for t in jrp["marginal_return_period"]])
    _style(ax, "How often is a storm extreme in wind AND rain? Joint return period", "winters (log scale)")
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
