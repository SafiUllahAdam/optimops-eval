"""Figures for the service's officers: one message per chart, plain titles, saved to results/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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


# -- Task 2 ---------------------------------------------------------------------------------------


VEHICLES_RAMP = LinearSegmentedColormap.from_list("vehicles", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def reinforcement_curves(curve_v: pd.DataFrame, curve_f: pd.DataFrame) -> None:
    """What each extra vehicle buys: P90 against VSAVs added, unanswered fires against FPTs added."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = [
        (axes[0], curve_v, BLUE, "VSAV", "P90 of SAP response times (min)",
         "VSAV: about half a minute off the P90 each, up to the 8th"),
        (axes[1], curve_f, ORANGE, "FPT", "Unanswered fires per year",
         "FPT: the first four remove 94% of unanswered fires"),
    ]
    for ax, curve, color, vehicle, ylabel, title in panels:
        ax.errorbar(curve.index, curve["mean"], yerr=curve["hw"], color=color, lw=2, marker="o", ms=5,
                    capsize=3, elinewidth=1.2)
        ax.set_xticks(curve.index, [f"{k}\n{c}" if k else "0\nnone" for k, c in zip(curve.index, curve["centre"])],
                      fontsize=7.5)
        ax.set_xlabel(f"{vehicle}s added, and the centre that received the last one")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    axes[1].set_ylim(bottom=0)
    fig.text(0.01, 0.01, "Simulator, mean of 10 fresh replays per point (baseline: 50). Bars: 95% interval.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "reinforcement_curves.png")


def tradeoff_front(front: pd.DataFrame, picks: pd.DataFrame) -> None:
    """The menu: every split of up to 12 vehicles, P90 against unanswered fires, coloured by vehicles bought."""
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    dominated = front[~front["pareto"]]
    ax.scatter(dominated["p90"], dominated["inc"], s=22, color=NO_TARGET, zorder=2, label="Dominated (wasteful)")
    for total in (3, 6, 9, 12):
        line = front[front["pareto"] & (front["vehicles"] == total)].sort_values("VSAV")
        ax.plot(line["p90"], line["inc"], color=VEHICLES_RAMP(total / 12), lw=1.2, zorder=2,
                label=f"All splits of {total} vehicles")
    pts = front[front["pareto"]]
    sc = ax.scatter(pts["p90"], pts["inc"], c=pts["vehicles"], cmap=VEHICLES_RAMP, vmin=0, vmax=12, s=34,
                    edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for _, p in picks.iterrows():
        ax.scatter(p["p90"], p["inc"], s=190, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4)
        ax.annotate(p["label"], (p["p90"], p["inc"]), xytext=(9, 4), textcoords="offset points", fontsize=9,
                    color=INK, fontweight="bold", zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Extra vehicles bought", color=INK)
    cb.outline.set_visible(False)
    ax.set_xlabel("P90 of SAP response times (min)")
    ax.set_yscale("symlog", linthresh=10, linscale=0.6)
    ticks = [0, 5, 10, 20, 50, 100, 200, 300]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_ylim(-0.5, 330)
    ax.set_ylabel("Unanswered fires per year (scale stretched at the bottom)")
    ax.set_title("The menu: along each line, the same number of vehicles split differently\n"
                 "between VSAV (moves left) and FPT (moves down)", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.92), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_front.png")


def reinforcement_map(raw: dict, picks: pd.DataFrame, interventions: pd.DataFrame) -> None:
    """Where the extra vehicles go, one small map per recommended configuration."""
    communes, centres = raw["communes"].set_index("insee"), raw["centres"].set_index("code_cis")
    n = interventions.groupby("insee").size().reindex(communes.index, fill_value=0)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.1 * len(picks), 4.6), sharey=True)
    for ax, (_, p) in zip(np.atleast_1d(axes), picks.iterrows()):
        ax.scatter(communes["x_km"], communes["y_km"], s=6 + 60 * n / n.max(), color=NO_TARGET, zorder=1)
        ax.scatter(centres["x_km"], centres["y_km"], marker="^", s=26, color=MUTED, zorder=2)
        for cis, per_type in p["config"].items():
            x, y = centres.loc[cis, ["x_km", "y_km"]]
            text = "\n".join(f"+{k} {t}" for t, k in per_type.items())
            color = ORANGE if set(per_type) == {"FPT"} else BLUE if set(per_type) == {"VSAV"} else INK
            ax.scatter(x, y, marker="^", s=90, color=color, edgecolor=INK, linewidth=0.8, zorder=3)
            # a reinforced neighbour just to the east: put this label on the west side
            crowded = any(0 < centres.loc[c, "x_km"] - x < 8 and abs(centres.loc[c, "y_km"] - y) < 4
                          for c in p["config"] if c != cis)
            ax.annotate(f"{cis}\n{text}", (x, y), xytext=(-5 if crowded else 5, 4), textcoords="offset points",
                        fontsize=6.5, color=INK, ha="right" if crowded else "left", zorder=4)
        ax.set_title(f"{p['label']}: {p['vehicles']} vehicles\nP90 {p['p90']:.1f} min, {p['inc']:.0f} fires",
                     fontsize=9.5)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(handles=[
        Line2D([], [], marker="^", ls="", color=BLUE, mec=INK, ms=8, label="Extra VSAV"),
        Line2D([], [], marker="^", ls="", color=ORANGE, mec=INK, ms=8, label="Extra FPT"),
        Line2D([], [], marker="^", ls="", color=INK, ms=8, label="Extra VSAV and FPT"),
        Line2D([], [], marker="^", ls="", color=MUTED, ms=6, label="Other centres"),
    ], loc="lower left", frameon=False, fontsize=7.5)
    fig.suptitle("Where the extra vehicles go (grey dots: communes, sized by 2025 interventions)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "reinforcement_map.png")
