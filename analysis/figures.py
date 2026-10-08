"""Figures for the service's officers: one message per chart, plain titles, saved to results/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from helpers import RESULTS

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, NO_TARGET = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c3"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _judged(interventions: pd.DataFrame) -> pd.DataFrame:
    """Interventions with a binding threshold (Z4 excluded), with a 'late' flag."""
    judged = interventions[interventions["compliant"].notna()]
    return judged.assign(late=~judged["compliant"].astype(bool))


def map_late_share(interventions: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """Where: share of late interventions per commune, with the centres on top."""
    communes, centres = raw["communes"], raw["centres"]
    by_commune = (
        interventions.groupby("insee").size().rename("n").to_frame()
        .join(_judged(interventions).groupby("insee")["late"].mean().mul(100).rename("late_pct"))
        .join(communes.set_index("insee")[["nom_commune", "x_km", "y_km", "zone_sdacr"]])
    )
    size = 25 + 400 * by_commune["n"] / by_commune["n"].max()
    z4 = by_commune["zone_sdacr"] == "Z4"

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.scatter(by_commune.loc[z4, "x_km"], by_commune.loc[z4, "y_km"], s=size[z4], color=NO_TARGET,
               edgecolor=SURFACE, linewidth=1.5, zorder=2)
    pts = ax.scatter(by_commune.loc[~z4, "x_km"], by_commune.loc[~z4, "y_km"], s=size[~z4],
                     c=by_commune.loc[~z4, "late_pct"], cmap=RAMP, vmin=0, vmax=60,
                     edgecolor=SURFACE, linewidth=1.5, zorder=3)
    pro = centres["regime"] == "professionnel"
    ax.scatter(centres.loc[pro, "x_km"], centres.loc[pro, "y_km"], marker="^", s=70, color=INK, zorder=4)
    ax.scatter(centres.loc[~pro, "x_km"], centres.loc[~pro, "y_km"], marker="^", s=70, facecolor=SURFACE,
               edgecolor=INK, linewidth=1.4, zorder=4)
    worst = by_commune.loc[~z4].nlargest(4, "late_pct")
    for i, (_, row) in enumerate(worst.iterrows()):
        ax.annotate(f"{row['nom_commune']}  {row['late_pct']:.0f}% late", (row["x_km"], row["y_km"]),
                    xytext=(8, 7 if i % 2 == 0 else -13), textcoords="offset points", fontsize=8.5,
                    color=INK, zorder=5)
    cb = fig.colorbar(pts, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("% of interventions arriving late", color=INK)
    cb.outline.set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker="^", ls="", color=INK, ms=8, label="Professional centre"),
        Line2D([], [], marker="^", ls="", mfc=SURFACE, mec=INK, ms=8, label="Volunteer centre"),
        Line2D([], [], marker="o", ls="", color=NO_TARGET, ms=9, label="Z4 commune (no target)"),
    ], loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Where: late arrivals concentrate in outlying communes\n"
                 "% of 2025 interventions above threshold, circle size = number of interventions", fontsize=10)
    ax.set_xlabel("km (east)")
    ax.set_ylabel("km (north)")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "map_late_share.png")
    return worst[["nom_commune", "zone_sdacr", "n", "late_pct"]].round(1)


def hourly_on_time(interventions: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """When in the day: % on time by hour, split by the regime of the commune's first centre."""
    centres = raw["centres"].set_index("code_cis")
    first = raw["plan_deploiement"].query("rang == 1").set_index("insee")["code_cis"]
    judged = _judged(interventions)
    hourly = (
        judged.assign(hour=judged["alert_time"].dt.hour, regime=judged["insee"].map(first.map(centres["regime"])))
        .groupby(["regime", "hour"])["late"].apply(lambda s: 100 * (1 - s.mean())).unstack(0)
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for col, color, label in [("professionnel", BLUE, "Communes covered first by a professional centre"),
                              ("volontaire", ORANGE, "Communes covered first by a volunteer centre")]:
        ax.plot(hourly.index, hourly[col], color=color, lw=2, marker="o", ms=4, label=label)
        ax.annotate(f"{hourly[col].min():.0f}% at worst", (hourly[col].idxmin(), hourly[col].min()),
                    xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axvspan(11.5, 19.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(15.5, 101, "daytime peak of calls", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(0, 24, 2), [f"{hh}h" for hh in range(0, 24, 2)])
    ax.set_ylim(50, 104)
    ax.set_ylabel("% of interventions on time")
    ax.set_xlabel("Hour of the alert")
    ax.set_title("When (day): peak hours hurt, and volunteer-covered communes lag all day")
    ax.legend(loc="center left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "hourly_on_time.png")
    return pd.DataFrame({
        "night 0h to 6h": hourly.loc[0:6].mean(),
        "peak 12h to 19h": hourly.loc[12:19].mean(),
        "worst hour": hourly.idxmin().astype(str) + "h",
    }).round(1)


def monthly_unfulfilled(interventions: pd.DataFrame) -> pd.DataFrame:
    """When in the year: interventions nobody could answer, by month and reason."""
    monthly = (
        interventions[interventions["unfulfilled"]]
        .assign(month=lambda d: d["alert_time"].dt.month)
        .pivot_table(index="month", columns="reason", values="id_intervention", aggfunc="count", fill_value=0)
        .reindex(range(1, 13), fill_value=0)[["INC", "SAP", "AVP"]]
    )
    labels = {"INC": "Fire (no FPT available)", "SAP": "Assistance to persons (no VSAV)",
              "AVP": "Road accident (no VSAV)"}

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bottom = np.zeros(12)
    for reason, color in zip(monthly.columns, [ORANGE, BLUE, AQUA]):
        ax.bar(monthly.index, monthly[reason], bottom=bottom, color=color, width=0.72,
               edgecolor=SURFACE, linewidth=2, label=labels[reason])
        bottom += monthly[reason].to_numpy()
    for m, total in zip(monthly.index, bottom):
        ax.text(m, total + 2, f"{total:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Interventions with no vehicle available")
    ax.grid(axis="x", visible=False)
    ax.set_title("When (season): June to August hold most of the interventions nobody could answer")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "monthly_unfulfilled.png")
    monthly.index = MONTHS
    return monthly.T
