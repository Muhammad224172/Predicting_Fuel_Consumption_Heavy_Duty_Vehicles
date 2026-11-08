# Fuel consumption of heavy-duty trucks from CAN-bus telemetry

Window-level regression of fuel consumption per 100 km from in-vehicle signals
(speed, engine speed, torque, pedal positions, road slope, weight, ...), with
gradient-boosted trees and random forests.

- **Context:** ENS 492 graduation project, Sabancı University, October 2023 to
  June 2024. Supervisor: Prof. Mustafa Ünel. Team: Muhammad Abdulrahman and
  Efe Üykü (group 124). Final report: [`reports/ENS492_final_report_group124.docx`](reports/ENS492_final_report_group124.docx).
- **This repository** is a 2026 refactor of the original Colab notebook
  (kept unchanged in [`archive/ENS492_original.ipynb`](archive/ENS492_original.ipynb))
  into a tested, configurable pipeline. It re-runs the original protocol and
  adds a stricter evaluation. Section [What changed in the refactor](#what-changed-in-the-refactor)
  lists every difference.

## Problem

Fleet operators plan routes and schedules around fuel cost, so they need a
model of how much fuel a truck uses under given driving conditions. The task
here is to estimate the fuel used per 100 km over a window of roughly ten
minutes of driving, from the signals recorded in the same window.

This is concurrent estimation, not forecasting: the inputs (torque, speed,
pedal positions, ...) are measured in the same window as the fuel. It tells
us how well driving conditions explain consumption, not how well consumption
can be predicted before a trip.

## Data

459 records (trips) from 187 trucks, logged between September and November
2015, provided for the course. A record is one CSV of irregularly sampled
CAN-bus signals (0.2-2 s between rows, about one hour per trip). Several
channels are masked by the provider, including the fuel counter, so fuel is in
"masked units", not litres. Records 61 and 66 have too few complete rows
after cleaning to form a 600-row window, and one more record loses all its
windows to the outlier filters, so the models use 456 records. The data is
private and not included; see
[`data/README.md`](data/README.md) for the schema, units and where to put the
files.

## Method

The pipeline (`src/fuelcons/`) reproduces the steps of the original notebook:

1. **Per-record cleaning** (`cleaning.py`). Missing brake-pedal values are set
   to 0 (the pedal is only logged while pressed). All other gaps are filled by
   linear interpolation, then forward fill. Rows recorded before every signal
   has appeared once are dropped. Two slope indicators (uphill, downhill) are
   added.
2. **Sliding windows** (`windowing.py`). Windows of 600 consecutive rows,
   advanced by 120 rows, within each record. At the median sampling interval
   of 1 s this is about 10 minutes advanced by 2 minutes, so consecutive
   windows share 80% of their rows. Because sampling is irregular, a 600-row
   window actually spans 2 to 42 minutes (median 10.4). Time-based windows are
   available with `window.unit: seconds`.
3. **Window features** (`features.py`). Mean, median, max, min, standard
   deviation and range of each signal over the window.
4. **Target.** `fuel_per_100km = fuel used in window * 1e5 / metres driven in
   window`.
5. **Outlier removal** (`cleaning.py`). Three IQR filters (1.5 x IQR), applied
   one after another over all windows: fuel per unit weight, fuel per 100 km,
   fuel used. This also drops windows with zero distance. 10,499 windows
   become 10,087.
6. **Feature set** (`selection.py`). The original final models used 58
   features: every window statistic except medians for ten signals, the
   distance covered, and the slope indicators. Fuel statistics are excluded
   because they define the target.
7. **Models** (`models.py`). XGBoost and random forest with the original
   hyper-parameter grids, plus two baselines that the original did not have:
   predicting the training mean, and ridge regression on standardised
   features.
8. **Evaluation** (`evaluation.py`). 10 outer folds under three schemes:

   | Scheme | Test folds contain | What it measures |
   |---|---|---|
   | `kfold` (original) | random windows | fit to windows of trips already seen in training |
   | `group_record` | whole records | generalisation to new trips |
   | `group_vehicle` | all records of a truck | generalisation to new trucks and trips |

   With shuffled KFold, almost every test window has neighbours in the
   training folds that share up to 80% of its rows, so the score is close to
   an interpolation score. Grouped splits remove that leakage. Hyper-parameters
   are tuned by grid search inside each outer training fold (nested CV, 3 inner
   folds that follow the outer scheme), so no test fold is used for tuning.

## Results

All scores are mean ± standard deviation over 10 outer folds (population std,
as in the original). Target: fuel per 100 km in masked units; over the 10,087
windows it has mean 4.45 and standard deviation 2.53, so an RMSE of 0.55
is about a fifth of the target's spread.

### 1. Numbers in the original project (provenance only)

| Source | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Final report (May 2024) | XGBoost | 0.5582 ± 0.0505 | 0.3290 ± 0.0109 | 0.9506 ± 0.0096 |
| Final report (May 2024) | Random forest | 0.6517 ± 0.0362 | 0.3988 ± 0.0109 | 0.9328 ± 0.0075 |
| Saved notebook output, cell 104 | XGBoost | 0.5386 ± 0.0667 | 0.3019 ± 0.0139 | 0.9538 ± 0.0112 |
| Saved notebook output, cell 127 | Random forest | 0.6239 ± 0.0378 | 0.3803 ± 0.0145 | 0.9388 ± 0.0065 |

The notebook prints XGBoost scores per fold only; the row above is the mean
and std of those ten printed values. Both rows come from the original
protocol (shuffled KFold, hyper-parameters tuned once on the first 80% of the
windows), which leaks across folds as explained above. They are not
comparable to the grouped results below.

### 2. Re-run of the original protocol

Run on 2026-11-07 with `configs/original_protocol.yaml` (seed 42, code at
commit `1895de6`); full output in [`results/original_protocol/`](results/original_protocol/summary.md).
The re-run uses the notebook's 58 features, its grids, its record order and
its tuning procedure. It produces exactly the notebook's window counts (10,499
windows, 10,087 after outlier removal) and matches the window values printed
in the notebook.

| Model | Tuned hyper-parameters | RMSE | MAE | R² |
|---|---|---|---|---|
| XGBoost | 500 trees, depth 5, learning rate 0.05 | 0.5449 ± 0.0574 | 0.3068 ± 0.0130 | 0.9530 ± 0.0091 |
| Random forest | 200 trees, sqrt features, min. leaf 2, min. split 5 | 0.6317 ± 0.0395 | 0.3819 ± 0.0149 | 0.9373 ± 0.0065 |

These agree with the notebook's saved outputs to within about 0.01 RMSE, and
the per-fold XGBoost scores follow the notebook's fold by fold. The remaining
differences are expected: XGBoost 3.4 instead of 2.0 (the grid search now picks
500 instead of 700 trees), and the notebook's random forest had no seed.

### 3. Leakage-aware evaluation (main result)

Run on 2026-11-07 with `configs/default.yaml`: seed 42, code at commit
`1dfb7e8`, the same 58 features and 10,087 windows, and nested tuning
throughout. The full output is in
[`results/default/`](results/default/summary.md).

| Model | `kfold` RMSE | `kfold` R² | `group_record` RMSE | `group_record` R² | `group_vehicle` RMSE | `group_vehicle` R² |
|---|---|---|---|---|---|---|
| Training mean | 2.527 ± 0.060 | -0.002 ± 0.002 | 2.526 ± 0.112 | -0.005 ± 0.006 | 2.529 ± 0.126 | -0.011 ± 0.010 |
| Ridge | 0.742 ± 0.046 | 0.913 ± 0.011 | 0.750 ± 0.173 | 0.908 ± 0.038 | 0.933 ± 0.504 | 0.826 ± 0.224 |
| Random forest | 0.634 ± 0.037 | 0.937 ± 0.008 | 0.726 ± 0.135 | 0.915 ± 0.030 | 0.735 ± 0.203 | 0.911 ± 0.040 |
| **XGBoost** | **0.554 ± 0.057** | **0.951 ± 0.010** | **0.643 ± 0.204** | **0.929 ± 0.047** | **0.634 ± 0.212** | **0.931 ± 0.038** |

MAE for every cell is in `summary.md`. For XGBoost it is 0.305 (`kfold`),
0.351 (`group_record`) and 0.361 (`group_vehicle`).

- **Window overlap inflates the original score.** When whole trips are held
  out, XGBoost's RMSE rises from 0.554 to 0.643 (+16%) and its R² falls from
  0.951 to 0.929. Nested tuning alone changes almost nothing: the original
  protocol gives 0.545 / 0.953.
- **New trucks are about as hard as new trips.** Holding out every record of
  a truck gives RMSE 0.634, the same as holding out trips. The gap comes from
  overlapping windows within a trip, not from truck identity.
- **Grouped scores vary more between folds.** The fold standard deviation of
  RMSE grows from 0.06 to about 0.21, because some trips and trucks are much
  harder than others. For held-out trucks, XGBoost's per-fold R² ranges from
  0.88 to 0.98.
- **XGBoost wins in every scheme.** It leads the random forest by 0.08 to 0.10
  RMSE. Ridge regression comes close to the random forest only on
  trip-grouped folds (0.750 against 0.726). It is unreliable on held-out
  trucks: one fold collapses to R² 0.17, while the other nine lie between
  0.82 and 0.97.

The figure to quote is therefore R² ≈ 0.93 and RMSE ≈ 0.63 (masked units)
on trucks not seen in training. The original 0.95 is the score on windows
whose neighbouring windows were in the training set.

### Why the report and the notebook disagree

The report's summary numbers are not reproduced by any saved output, and
they do not match the report's own figures:

- **The report's figures come from the notebook run.** The "Model
  Performance Across Folds (XGBOOST)" figure in the report shows the same ten
  fold values as notebook cell 104, whose mean is RMSE 0.539, MAE 0.302,
  R² 0.954, not the 0.558 / 0.329 / 0.951 stated in the text. The random-forest
  fold figure averages to about RMSE 0.63, MAE 0.38, R² 0.938 (read off the
  plot), again not the stated 0.652 / 0.399 / 0.933.
- **The text's hyper-parameters differ from the notebook's.** The report lists
  300 XGBoost trees; the notebook's grid search chose 700. The saved random
  forest cell also prints "48 candidates" for a grid that has 24, so the code
  was edited after that run.
- **Seeds, tree count and fold assignment do not close the gap**
  ([`results/discrepancy/`](results/discrepancy/discrepancy.md), same protocol
  with fixed hyper-parameters). XGBoost with the report's 300 trees gives RMSE
  0.552, MAE 0.315, R² 0.952: close on RMSE and R², but not on MAE. Changing
  only the record order (and with it the folds) moves XGBoost's RMSE by 0.016.
  The random forest with the report's hyper-parameters stays within RMSE
  0.623-0.637 and MAE 0.378-0.383 over five seeds and two record orders, below
  the reported 0.652 and 0.399.

The most likely explanation is that the summary numbers come from an earlier
run, probably on a different version of the window table or feature set, that
was not saved. This cannot be confirmed from the surviving files. This
repository therefore cites the report's numbers only as provenance, and uses
the re-run above for any comparison.

## Reproducing

```bash
pip install -r requirements.txt          # Python 3.11; versions as tested
python scripts/run_pipeline.py --raw-dir "/path/to/ENS 492 - Fuel Consumption Dataset/Records"
python scripts/discrepancy_check.py --raw-dir "/path/to/.../Records"   # optional, section "Why ... disagree"
```

`run_pipeline.py` runs `configs/original_protocol.yaml` and then
`configs/default.yaml`; pass `--config` to run one of them. The raw folder can
also be given as `FUELCONS_RAW_DIR` or by placing the files in `data/raw/`.
Building the window table takes about 2 minutes (it is cached in
`data/processed/`); the nested CV of `default.yaml` took about 2 h 20 min with 5
parallel workers on a 6-core laptop. Each run writes to `results/<config>/`:

- `summary.md`, `summary.csv`: mean ± std per scheme and model
- `folds.csv`: per-fold metrics and the hyper-parameters chosen in each fold
- `run_info.json`: config, seed, package versions, git commit, data counts
- `figures/`: CV comparison, out-of-fold predictions, target correlations,
  XGBoost feature importance
- `predictions.csv`: out-of-fold predictions (git-ignored)

Tests run on synthetic data generated in `tests/synthetic.py` (invented values
that mimic the schema; no result in this README comes from them):

```bash
pytest
ruff check .
```

## Repository layout

```
archive/ENS492_original.ipynb    original Colab notebook, unchanged (outputs included)
reports/                         original final report (docx)
configs/                         original_protocol.yaml, default.yaml, notebook_record_order.txt
data/README.md                   data schema; the data itself is git-ignored
scripts/run_pipeline.py          one-command entry point
scripts/discrepancy_check.py     report-vs-notebook analysis
src/fuelcons/                    io, cleaning, windowing, features, selection, models, evaluation, plots, pipeline, cli
tests/                           pytest suite on synthetic data
results/                         summaries and figures from the runs above
```

## What changed in the refactor

What the original notebook did, and where this repository deviates:

| Step | Original notebook | This repository |
|---|---|---|
| File order | `os.listdir` order on Colab | numeric order; the original protocol re-uses the notebook's order (`configs/notebook_record_order.txt`, read from the notebook output) so its splits match |
| Cleaning | brake NaN -> 0, interpolate, forward fill, drop incomplete rows; files overwritten in place | same steps, in memory; raw files are never modified |
| Windows | 600 rows / step 120, commented as "5 minutes window" / "1 minute shift"; the report says 5-minute windows with 1-minute overlap | same 600/120 rows, documented as such; time-based windows optional |
| Slope features | `slope_negative` aggregated twice, `slope_positive` never; the two indicators at the first row of each window used as features | same 58 features for the reproduction; the `no_median` set aggregates both indicators and drops the point values |
| Scaling | MinMax fitted on all windows before modelling | none for trees (no effect on them); standardisation inside the ridge pipeline |
| Feature selection | correlations plotted and a top-30 list printed, but not used; final models use all non-median statistics | same 58 features; an optional correlation filter (`features.corr_top_k`) is fitted inside each training fold |
| Tuning | `GridSearchCV(cv=5)` once on the first 80% of windows, default R² scoring | nested: grid search inside each outer training fold, RMSE scoring; the original procedure is kept as `tuning: original` |
| Evaluation | shuffled 10-fold KFold over all windows, including the tuning data | the same, plus GroupKFold by record and by vehicle, and two baselines |
| Seeds | random forest without `random_state` | every model and splitter seeded (`seed: 42`) |
| Figures | made interactively | generated from saved results by `plots.py` |

Things the notebook imports or the report mentions that the code does not do:
KNNImputer, DBSCAN and PCA are imported but never used; there is no early
stopping; the report's "four models based on different feature subsets" are
not in the notebook, which only contains the final no-median models (plus two
exploratory XGBoost models on a different target, `EngTotalFuelUsed_range`,
evaluated on an unshuffled 80/20 split).

## Limitations

- The target is a ratio of two counter differences over a short window, so
  it is noisy for windows with little distance. The IQR filters remove the
  extremes, but they are applied to the target over the whole dataset before
  cross-validation, so the scores describe "typical" windows only.
- The features are measured in the same window as the fuel, so the model is
  a soft sensor, not a forecaster. Route-level use would need inputs that are
  known in advance (planned speed, slope, load).
- Fuel and several channels are masked, so errors cannot be stated in litres.
- One provider, two months of 2015, 187 trucks: no claim is made about other
  fleets or vehicle types.

## License

MIT, see [LICENSE](LICENSE). The license covers the code only, not the data.
