# Results: `default`

- Run started (UTC): 2026-11-07T17:26:34+00:00; git commit: `1dfb7e8`
- Data: 456 records, 187 vehicles, 10087 windows after outlier removal (10499 before)
- Windows: 600 rows, step 120 rows; feature set `original` (58 features), correlation filter k=None
- CV: 10 outer folds, tuning `nested` (3 inner folds)
- Target `fuel_per_100km`: mean 4.4475, std 2.5274 (masked fuel units per 100 km)

Mean ± standard deviation over outer folds.

| CV scheme | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| kfold | dummy | 2.5270 ± 0.0600 | 2.0608 ± 0.0449 | -0.0015 ± 0.0020 |
| kfold | ridge | 0.7417 ± 0.0460 | 0.4403 ± 0.0104 | 0.9133 ± 0.0110 |
| kfold | random_forest | 0.6338 ± 0.0368 | 0.3811 ± 0.0100 | 0.9367 ± 0.0076 |
| kfold | xgboost | 0.5537 ± 0.0567 | 0.3049 ± 0.0126 | 0.9513 ± 0.0099 |
| group_record | dummy | 2.5256 ± 0.1116 | 2.0606 ± 0.0871 | -0.0046 ± 0.0060 |
| group_record | ridge | 0.7502 ± 0.1730 | 0.4575 ± 0.0668 | 0.9077 ± 0.0380 |
| group_record | random_forest | 0.7264 ± 0.1346 | 0.4312 ± 0.0463 | 0.9145 ± 0.0297 |
| group_record | xgboost | 0.6430 ± 0.2043 | 0.3514 ± 0.0477 | 0.9287 ± 0.0465 |
| group_vehicle | dummy | 2.5294 ± 0.1259 | 2.0646 ± 0.0918 | -0.0109 ± 0.0101 |
| group_vehicle | ridge | 0.9328 ± 0.5035 | 0.5076 ± 0.1227 | 0.8261 ± 0.2237 |
| group_vehicle | random_forest | 0.7346 ± 0.2028 | 0.4447 ± 0.0787 | 0.9109 ± 0.0400 |
| group_vehicle | xgboost | 0.6343 ± 0.2118 | 0.3612 ± 0.0713 | 0.9312 ± 0.0384 |
