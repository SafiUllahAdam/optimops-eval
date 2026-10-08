"""Corrected version of docuu/adam/audit_a_realiser.py (Act 3).

Same idea as the original (40 random reinforcements, Pareto front, recommendation), with the errors fixed:
  1. unfulfilled calls are kept at 60 min instead of being dropped;
  2. delays are measured per intervention on the first vehicle to arrive, not per vehicle row;
  3. the 90th percentile is np.percentile(x, 90), not np.percentile(x, 0.9);
  4. every objective is minimised in the dominance test (the original maximised the number of vehicles);
  5. each configuration is replayed 5 times, not once, because the simulator is noisy;
  6. the recommendation is a short list of trade-offs, not "lowest P90 whatever the cost".

Simulator calls go through simulation.evaluate(), so re-running this file costs no extra replays.
"""

import numpy as np
import pandas as pd

import helpers as h
import simulation as sim

N_CONFIGURATIONS = 40
SEED = 0
REPLAYS = 5
RESERVE = 300  # safety: never let the audit run use the last 300 replays


def sample_configurations(centres, n=N_CONFIGURATIONS, seed=SEED):
    """Unchanged from the original: random reinforcements of 1 to 12 vehicles (same seed, same 40 plans)."""
    rng = np.random.default_rng(seed)
    configurations = []
    for _ in range(n):
        n_vehicles = rng.integers(1, sim.MAX_VEHICLES + 1)
        config = {}
        for _ in range(n_vehicles):
            cis = str(rng.choice(centres))
            vehicle = str(rng.choice(["VSAV", "FPT"]))
            config.setdefault(cis, {}).setdefault(vehicle, 0)
            config[cis][vehicle] += 1
        configurations.append(config)
    return configurations


def p90_steps(raw):
    """SAP P90 as the original computes it, then with each data error fixed in turn."""
    iv = raw["interventions"].assign(
        delay=lambda d: (d["horodatage_arrivee"] - d["horodatage_alerte"]).dt.total_seconds() / 60)
    rows = iv.dropna(subset=["horodatage_arrivee"])
    sap_rows = rows.loc[rows["raison"] == "SAP", "delay"]
    first = rows.sort_values("delay").drop_duplicates("id_intervention")
    sap_first = first.loc[first["raison"] == "SAP", "delay"]
    correct = h.build_interventions(raw)
    sap_correct = correct.loc[correct["reason"] == "SAP", "delay_min"]
    return pd.Series({
        "Original: rows, unfulfilled dropped, percentile(0.9)": np.percentile(sap_rows, 0.9),
        "+ fix percentile (90)": np.percentile(sap_rows, 90),
        "+ fix one value per call, first arrival": np.percentile(sap_first, 90),
        "+ fix unfulfilled kept at 60 min (official rule)": np.percentile(sap_correct, 90),
    }).round(2)


def front_original(df):
    """The original dominance test, kept to measure its effect: it treats MORE vehicles as better."""
    obj = df[["p90", "inc", "cost"]].to_numpy()
    keep = np.ones(len(obj), dtype=bool)
    for i in range(len(obj)):
        for j in range(len(obj)):
            if i != j and (obj[j][0] <= obj[i][0] and obj[j][1] <= obj[i][1] and obj[j][2] >= obj[i][2]) and (
                    obj[j][0] < obj[i][0] or obj[j][1] < obj[i][1] or obj[j][2] > obj[i][2]):
                keep[i] = False
                break
    return keep


def front_corrected(df):
    """All three objectives minimised."""
    return sim.pareto_mask(df[["p90", "inc", "cost"]].to_numpy())


def evaluate_all(configurations, n, tag):
    rows = []
    for i, config in enumerate(configurations):
        r = sim.evaluate(config, n=n, tag=tag, reserve=RESERVE)
        rows.append({"plan": i, "p90": r["p90"], "inc": r["inc"], "cost": int(r["vehicles"]),
                     "p90_sd": r["p90_sd"], "inc_sd": r["inc_sd"], "config": sim.describe(config)})
    return pd.DataFrame(rows).set_index("plan")


def recommend(front):
    """Trade-offs, not a single winner: the best plan at each vehicle count on the corrected front."""
    return front.sort_values(["cost", "p90"]).groupby("cost").head(1)[["cost", "p90", "inc", "config"]]


def main():
    raw = h.load_raw()
    print("SAP P90 (min):\n", p90_steps(raw).to_string())
    configurations = sample_configurations(raw["centres"]["code_cis"].tolist())
    results = evaluate_all(configurations, n=REPLAYS, tag="E_audit_n5")
    front = results[front_corrected(results)]
    print(f"\n{len(front)} non-dominated plans out of {len(results)}")
    print(recommend(front).to_string())


if __name__ == "__main__":
    main()
