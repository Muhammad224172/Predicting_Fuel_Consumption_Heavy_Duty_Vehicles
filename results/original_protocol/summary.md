# Results: `original_protocol`

- Run started (UTC): 2026-11-07T17:15:46+00:00; git commit: `1895de6`
- Data: 456 records, 187 vehicles, 10087 windows after outlier removal (10499 before)
- Windows: 600 rows, step 120 rows; feature set `original` (58 features), correlation filter k=None
- CV: 10 outer folds, tuning `original`
- Target `fuel_per_100km`: mean 4.4475, std 2.5274 (masked fuel units per 100 km)

Mean ± standard deviation over outer folds.

| CV scheme | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| kfold | random_forest | 0.6317 ± 0.0395 | 0.3819 ± 0.0149 | 0.9373 ± 0.0065 |
| kfold | xgboost | 0.5449 ± 0.0574 | 0.3068 ± 0.0130 | 0.9530 ± 0.0091 |
