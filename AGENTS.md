# AGENTS.md

Guide for anyone (person or coding assistant) working in this repository.

## What this project is

A three-hour technical test for the OPTIMOPS PhD (FEMTO-ST, AIMOS team): multi-criteria optimisation of fire and
rescue resources under uncertainty. One department, one year of interventions, a stochastic coverage simulator.

- **Act 1**: reproduce the three official indicators, show where and when coverage degrades, build the
  commune to first-centre table.
- **Act 2**: decide where to add up to 12 vehicles (VSAV or FPT) across 30 centres. Three objectives to minimise:
  SAP response time P90, unfulfilled fires, number of vehicles. Deliver a Pareto front and 3 or 4 recommendations.
- **Act 3**: audit `audit_a_realiser.py`, fix its errors and measure how much each one changes the decision.

The work is graded on data analysis, programming, optimisation and scientific reasoning. Reasoning is shown
everywhere: why a choice was made, what was assumed, how reliable the result is, what its limits are.

## Layout

| Folder | Content | Rule |
|---|---|---|
| `docuu/adam/` | Original handout: brief, guide, data, simulator, audit file | Never edit. The simulator runs from here. |
| `docs/` | Copies of the brief and the business guide | Reference only |
| `raw_data/` | Byte-identical copy of `donnees/` | Read only, never modified |
| `data/` | Cleaned or derived tables | Rebuilt by the notebooks |
| `analysis/` | `helpers.py` (data, indicators), `simulation.py` (cached simulator calls, Pareto), `figures.py` (charts), one notebook per task | All code lives here |
| `results/` | Figures (PNG) and result tables (CSV) | Written by code only |
| `reports/` | Final report (3 pages max) | Written last |

Code goes in a shared module when two notebooks need it; otherwise it stays in the notebook. No new file per
small function, no single giant file. Notebooks: `task1_baseline.ipynb`, then `task2_...`, `task3_...`.

## Definitions (from `docs/notice_metier.md`, not negotiable)

- Response time = alert to arrival of the **first vehicle to arrive**, any type. Use arrival time, never row order:
  the data has no dispatch rank and row order often differs from arrival order.
- No arrival = **unfulfilled**, counted with **60 min** in delay indicators.
- On time = delay ≤ threshold(zone, reason). **Z4 is excluded** from the compliance rate.
- Indicators are computed per **intervention**, never per vehicle row.
- Reference results: P90 SAP 25.51, unfulfilled fires 290, compliance 86.45 (86.46 from timestamps, a known
  sub-second rounding effect). Any change to `helpers.py` must keep these.

## Simulator (Act 2)

- Budget: **2000 replays for the whole test**, counted across kernel restarts. `n_replicats=5` costs 5.
- Plan the experiment before spending: estimate noise first, then screen, then confirm the best candidates.
- Never delete or edit `.budget.json` or `journal_simulateur.csv`. The journal is a deliverable, sent as is.
- Call the simulator only through `simulation.evaluate()`. It caches every answer in `results/simulation_runs.csv`,
  so notebooks re-run for free. Never delete that file: re-running would spend the budget again.
- Budget state: Task 2 used 950, Task 3 used 245 (5 of them charged by an interrupted run, not in the journal). 805 left.
- Do not modify the simulator. Results are noisy: report means with their spread, never a single replay.

## Working rules

- Work task by task. Stop after each one and wait for the owner's go before starting the next.
- Keep notebooks short: a little code, the key table or figure, then two or three plain sentences.
- Writing style: plain and human, short sentences, no em dashes, no filler. Figures must read well for a fire
  officer, not only for a data scientist.
- Run notebooks top to bottom in the `optimops` conda environment before committing. From the command line use
  `jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=optimops`: the default `python3` kernel is base
  Python, which cannot load the simulator's `.pyc` files (bad magic number).
- Commits: plain messages, no AI co-author lines. This repository is public: do not commit `docuu/`, `docs/`,
  `raw_data/` or `data/` (exam material, private emails).

## Deliverables (one archive at the end of the session)

1. The code, in the state it ran (`analysis/`).
2. A report of 3 pages max in `reports/`, with a half-page section "steering the assistance": what was asked
   of the AI assistant, what it proposed that was refused, and why. Also state what one more day would add.
3. The figures (`results/`).
4. `docuu/adam/simulateur/journal_simulateur.csv`, unchanged.
