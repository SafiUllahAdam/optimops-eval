"""Shared helpers for the OptimOps evaluation.

Definitions follow docs/notice_metier.md (the service's rules, not free choices):
- response time  = alert -> arrival of the FIRST vehicle to arrive (all vehicle types);
- unfulfilled    = no vehicle arrived (empty arrival time) -> conventional 60 min;
- compliance     = delay <= threshold(zone, reason); zone Z4 has no binding target
                   and is excluded from the compliance rate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

UNFULFILLED_DELAY_MIN = 60.0
RAW_FILES = [
    "interventions",
    "communes",
    "centres",
    "plan_deploiement",
    "seuils_reglementaires",
    "baseline_officielle",
]


def load_raw() -> dict[str, pd.DataFrame]:
    """Read the six raw CSV files (never modified) and parse timestamps."""
    raw = {name: pd.read_csv(RAW / f"{name}.csv") for name in RAW_FILES}
    iv = raw["interventions"]
    iv["horodatage_alerte"] = pd.to_datetime(iv["horodatage_alerte"])
    iv["horodatage_arrivee"] = pd.to_datetime(iv["horodatage_arrivee"])
    return raw


def build_interventions(raw: dict[str, pd.DataFrame], delay_source: str = "timestamps") -> pd.DataFrame:
    """One row per intervention, judged on its first-arriving vehicle.

    delay_source="timestamps": arrival time - alert time (the guide's definition, to the second).
    delay_source="durations":  delai_depart_min + duree_trajet_s / 60 (same quantity, sub-second).
    """
    iv = raw["interventions"].copy()
    if delay_source == "timestamps":
        iv["delay_min"] = (iv["horodatage_arrivee"] - iv["horodatage_alerte"]).dt.total_seconds() / 60
    elif delay_source == "durations":
        iv["delay_min"] = iv["delai_depart_min"] + iv["duree_trajet_s"] / 60
    else:
        raise ValueError(f"unknown delay_source: {delay_source}")

    # Earliest arrival per intervention; NaN arrivals sort last, so an intervention
    # keeps a NaN delay only if no vehicle arrived at all.
    first = (
        iv.sort_values(["id_intervention", "delay_min"], na_position="last")
        .drop_duplicates("id_intervention")
        .set_index("id_intervention")
    )
    out = pd.DataFrame(
        {
            "alert_time": first["horodatage_alerte"],
            "insee": first["insee"],
            "reason": first["raison"],
            "n_vehicles": iv.groupby("id_intervention").size(),
            "first_vehicle": first["type_engin"].where(first["delay_min"].notna()),
            "first_centre": first["code_cis"].where(first["delay_min"].notna()),
            "observed_delay_min": first["delay_min"],
        }
    )
    out["unfulfilled"] = out["observed_delay_min"].isna()
    out["delay_min"] = out["observed_delay_min"].fillna(UNFULFILLED_DELAY_MIN)

    zones = raw["communes"].set_index("insee")["zone_sdacr"]
    out["zone"] = out["insee"].map(zones)
    thresholds = raw["seuils_reglementaires"].set_index(["zone_sdacr", "raison"])["seuil_min"]
    out["threshold_min"] = thresholds.reindex(pd.MultiIndex.from_arrays([out["zone"], out["reason"]])).to_numpy()
    # Compliance only where a binding threshold exists (Z4 -> NaN, excluded).
    out["compliant"] = (out["delay_min"] <= out["threshold_min"]).where(out["threshold_min"].notna())
    return out.reset_index()


def compute_kpis(interventions: pd.DataFrame) -> dict[str, float]:
    """The three indicators published by the service."""
    sap = interventions.loc[interventions["reason"] == "SAP", "delay_min"]
    inc = interventions["reason"] == "INC"
    judged = interventions["compliant"].dropna()
    return {
        "p90_delai_sap_min": float(np.percentile(sap, 90)),
        "inc_non_honorees": float((inc & interventions["unfulfilled"]).sum()),
        "taux_conformite_pct": float(100 * judged.astype(bool).mean()),
    }


def compare_to_baseline(kpis: dict[str, float], baseline: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side table: official value, recomputed value, gap."""
    table = baseline.rename(columns={"indicateur": "indicator", "valeur": "official"}).set_index("indicator")
    table["recomputed"] = pd.Series(kpis)
    table["gap"] = table["recomputed"] - table["official"]
    return table
