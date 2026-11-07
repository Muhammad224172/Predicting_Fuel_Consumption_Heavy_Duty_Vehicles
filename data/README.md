# Data

The telemetry was provided for the ENS 491/492 graduation project. It is not
redistributable, so **no data is committed to this repository**. Everything in
`data/` except this file is git-ignored.

## What to place where

```
data/
  raw/
    Record1.csv
    Record2.csv
    ...
    Record459.csv
  processed/            # created by the pipeline (window-table cache)
```

- File names must match `Record<N>.csv` (N is an integer). Other files in
  `data/raw/` are ignored, so the provider's `Dataset Description.xlsx` can sit
  there too.
- If the files are somewhere else (for example the original hand-off folder
  `ENS 492 - Fuel Consumption Dataset/Records/`), either copy them into
  `data/raw/` or pass the folder on the command line:
  `python scripts/run_pipeline.py --raw-dir "ENS 492 - Fuel Consumption Dataset/Records"`.
- The window table is cached in `data/processed/windows_<unit>_<size>_<step>.csv`.
  Use `--rebuild` after changing the raw files.

## Dataset at a glance

From the provider's description sheet and a profile of the files:

| | |
|---|---|
| Records (files) | 459, numbered 1-459 |
| Vehicles | 187 (1-27 records per vehicle, median 2) |
| Period | 2015-09-01 to 2015-11-23 |
| Record length | 2,108-16,560 rows (median 3,139); 47-280 min (median 54 min) |
| Sampling | irregular, 0.2-2 s between rows, median 1.0 s |

A **record** is one continuous logging session (a trip) of one vehicle: every
file has a single `VehicleID`, and rows are sorted by `DateTime`. There are no
trip or segment markers inside a file.

## Columns

Each CSV has a header row and these 14 columns. Empty cells are written as
`NaN`. Units are from the provider's description sheet; "masked" means the
provider rescaled the values, so their physical unit is unknown.

| Column | Description | Unit | Rows with a value |
|---|---|---|---|
| `DateTime` | timestamp, `yyyy-MM-dd HH:mm:ss.SSS` | - | 100% |
| `VehicleID` | vehicle identifier, 1-187 | - | 100% |
| `HghRslutionTotalVehicleDistance` | cumulative odometer | m | 3% |
| `TachographVehicleSpeed` | vehicle speed | km/h | 20% |
| `EngSpeed` | engine speed | rpm | 51% |
| `ActualEngPercentTorque` | engine torque | % | 20% |
| `AccelPedalPos1` | accelerator pedal position | % | 20% |
| `BrakePedalPos` | brake pedal position; logged only while pressed | % | 11% |
| `PCCM_Slope` | road slope | - | 34% |
| `DStgy_dmRdcAgAct` | AdBlue (urea) consumption indicator | masked, 0-1 | 20% |
| `EngOilTemp1` | engine oil temperature | masked, 0-1 | 3% |
| `EngCoolantTemp` | engine coolant temperature | masked, 0-1 | 20% |
| `GrossCombinationVehicleWeight` | gross combination weight incl. load | masked, 0-1 | 2% |
| `EngTotalFuelUsed` | cumulative fuel used by the engine | masked | 2% |

The signals arrive on the CAN bus at different rates, so a row typically holds
only one or two of them. The cumulative counters (distance, fuel) and the
weight are the sparsest.

## How the target is derived

There is no target column in the raw data. For each window the pipeline computes

```
fuel_per_100km = (EngTotalFuelUsed_max - EngTotalFuelUsed_min) * 1e5
               / (HghRslutionTotalVehicleDistance_max - HghRslutionTotalVehicleDistance_min)
```

that is, fuel used per 100 km driven in the window. Because `EngTotalFuelUsed`
is masked, the unit is "masked fuel units per 100 km", not litres per 100 km.
Windows with zero distance (vehicle standing) give an infinite value and are
removed by the outlier filter. See the main README for the full pipeline.
