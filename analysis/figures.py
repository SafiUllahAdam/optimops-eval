"""Figures for the service's officers: one message per chart, plain titles, saved to results/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from helpers import RESULTS

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, NO_TARGET = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c3"
LATE_CLASSES = ["Under 10%", "10 to 25%", "25 to 50%", "Over 50%"]
LATE_COLORS = ["#f9cfa5", "#f09a5e", "#d65a2e", "#9c2a1a"]   # light to dark: worse is darker
TIME_BLOCKS = {"Night": "0h to 7h", "Morning": "7h to 12h", "Afternoon": "12h to 19h", "Evening": "19h to 24h"}
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
    """Where: each commune coloured by its share of late arrivals, with the fire stations on top."""
    communes, centres = raw["communes"], raw["centres"]
    by_commune = (
        _judged(interventions).groupby("insee")["late"].mean().mul(100).rename("late_pct").to_frame()
        .join(interventions.groupby("insee").size().rename("n"), how="right")
        .join(communes.set_index("insee")[["nom_commune", "x_km", "y_km", "zone_sdacr"]])
    )
    classes = pd.cut(by_commune["late_pct"], [0, 10, 25, 50, 100.01], right=False, labels=LATE_CLASSES)

    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    z4 = by_commune["zone_sdacr"] == "Z4"
    ax.scatter(by_commune.loc[z4, "x_km"], by_commune.loc[z4, "y_km"], s=110, color=NO_TARGET,
               edgecolor=SURFACE, linewidth=1.5, zorder=2)
    handles = []
    for label, color in zip(LATE_CLASSES, LATE_COLORS):
        sel = classes == label
        ax.scatter(by_commune.loc[sel, "x_km"], by_commune.loc[sel, "y_km"], s=110, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        handles.append(Line2D([], [], marker="o", ls="", color=color, ms=10, label=f"{label}  ({sel.sum()} communes)"))
    handles.append(Line2D([], [], marker="o", ls="", color=NO_TARGET, ms=10, label=f"Z4, no deadline set  ({z4.sum()})"))

    pro = centres["regime"] == "professionnel"
    ax.scatter(centres.loc[pro, "x_km"], centres.loc[pro, "y_km"], marker="^", s=85, color=INK, zorder=4)
    ax.scatter(centres.loc[~pro, "x_km"], centres.loc[~pro, "y_km"], marker="^", s=85, facecolor=SURFACE,
               edgecolor=INK, linewidth=1.5, zorder=4)
    stations = [
        Line2D([], [], marker="^", ls="", color=INK, ms=9, label="Professional crews"),
        Line2D([], [], marker="^", ls="", mfc=SURFACE, mec=INK, ms=9, label="Volunteer crews"),
    ]

    worst = by_commune.loc[~z4].nlargest(3, "late_pct")
    for i, (_, row) in enumerate(worst.iterrows()):
        ax.annotate(f"{row['nom_commune']}: {row['late_pct']:.0f}% late", (row["x_km"], row["y_km"]),
                    xytext=(9, [7, -14, -28][i]), textcoords="offset points", fontsize=9, color=INK,
                    zorder=5)

    leg = ax.legend(handles=handles, title="Interventions arriving late", loc="upper left",
                    bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9.5, title_fontsize=10, alignment="left")
    ax.add_artist(leg)
    ax.legend(handles=stations, title="Fire stations", loc="upper left", bbox_to_anchor=(1.01, 0.62),
              frameon=False, fontsize=9.5, title_fontsize=10, alignment="left")

    x0, y0 = by_commune["x_km"].max() - 10, by_commune["y_km"].min() - 3   # 10 km scale bar
    ax.plot([x0, x0 + 10], [y0, y0], color=INK, lw=2)
    ax.text(x0 + 5, y0 + 1, "10 km", ha="center", fontsize=8.5, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Where: late responses are concentrated on the edges of the department", loc="left", pad=22)
    ax.text(0, 1.02, "Share of 2025 interventions arriving after the regulatory deadline, by commune",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(RESULTS / "map_late_share.png", bbox_inches="tight", bbox_extra_artists=[leg])
    return worst[["nom_commune", "zone_sdacr", "n", "late_pct"]].round(1)


def late_by_time_of_day(interventions: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """When in the day: % late by time block, for communes first served by professional vs volunteer crews."""
    centres = raw["centres"].set_index("code_cis")
    first = raw["plan_deploiement"].query("rang == 1").set_index("insee")["code_cis"]
    judged = _judged(interventions)
    blocks = pd.cut(judged["alert_time"].dt.hour, [-1, 6, 11, 18, 23], labels=list(TIME_BLOCKS))
    late = (
        judged.assign(block=blocks, regime=judged["insee"].map(first.map(centres["regime"])))
        .pivot_table(index="block", columns="regime", values="late", aggfunc="mean", observed=True)
        .mul(100)[["professionnel", "volontaire"]]
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(late))
    width = 0.36
    for k, (col, color, label) in enumerate([
        ("professionnel", BLUE, "Communes first served by professional crews"),
        ("volontaire", ORANGE, "Communes first served by volunteer crews"),
    ]):
        bars = ax.bar(x + (k - 0.5) * width, late[col], width=width, color=color, edgecolor=SURFACE,
                      linewidth=2, label=label)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in late[col]], padding=3, fontsize=9.5, color=INK)
    ax.set_xticks(x, [f"{name}\n{hours}" for name, hours in TIME_BLOCKS.items()])
    ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=INK)
    ax.set_ylim(0, 52)
    ax.set_ylabel("% of interventions arriving late")
    ax.grid(axis="x", visible=False)
    ax.set_title("When in the day: afternoons are the hardest, and volunteer areas are late at every hour",
                 loc="left", pad=22)
    ax.text(0, 1.02, "Share of 2025 interventions arriving after the regulatory deadline",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0, 0.98))
    fig.tight_layout()
    fig.savefig(RESULTS / "late_by_time_of_day.png")
    late.columns = ["professional", "volunteer"]
    return late.round(1).T


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
