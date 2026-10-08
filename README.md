# OptimOps technical test: code and results

Muhammad Safi Ullah Adam, 8 October 2026. The report is `report.pdf`.

## Contents

| Folder                              | What it holds                                                                                                                                                                    |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis/`                         | `helpers.py` (data and indicators), `simulation.py` (cached simulator calls, Pareto), `figures.py` (charts), `audit_corrected.py` (corrected audit script), one notebook per act |
| `results/`                          | Figures (PNG) and result tables (CSV), including `simulation_runs.csv`, the cache of every simulator answer                                                                      |
| `simulateur/journal_simulateur.csv` | The simulator log, unchanged                                                                                                                                                     |

## Environment

Python 3.14 in a conda environment (`optimops`) with pandas, NumPy, SciPy, Matplotlib, Seaborn and Jupyter.
Python 3.14 is required: the simulator's `.pyc` files do not load on other versions ("bad magic number").

```
conda create -n optimops -c conda-forge python=3.14 pandas numpy scipy matplotlib seaborn jupyter
conda activate optimops
```

## Reproducing the results

The exam material is not included. Place it as follows, from the project root:

- `raw_data/`: the six CSV files of `donnees/`, unchanged;
- `docuu/adam/`: the handout, with `simulateur/` inside (the code reads the simulator from there);
- `data/`: an empty folder, where Task 1 writes its per-intervention table.

Then run the notebooks top to bottom, in order, with the `optimops` kernel:

1. `analysis/task1_baseline.ipynb`: the three indicators, the figures, the commune to first-centre table.
2. `analysis/task2_reinforcement.ipynb`: noise check, search, Pareto front, plans A to D.
3. `analysis/task3_audit.ipynb`: the audit, error by error.

From the command line: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=optimops <notebook>`.

**Simulator budget.** Every simulator call goes through `simulation.evaluate()`, which stores the answer in
`results/simulation_runs.csv`. With that file in place, the notebooks re-run without spending any replay. Deleting
it would spend the budget again. The test used 1,195 of its 2,000 replays.

**Expected check.** Task 1 must give P90 SAP 25.51 min, 290 unanswered fires and 86.46% compliance (86.45 official,
a sub-second rounding gap explained in the report).
