# Original protocol with fixed hyper-parameters

Shuffled 10-fold KFold (seed 42), 58 original features, 10,087 windows. Mean ± std over folds.

| Model | Setting | Row order | RMSE | MAE | R² |
|---|---|---|---|---|---|
| xgboost | report hyper-parameters (300 trees) | notebook | 0.5522 ± 0.0545 | 0.3151 ± 0.0131 | 0.9518 ± 0.0088 |
| xgboost | notebook hyper-parameters (700 trees) | notebook | 0.5417 ± 0.0586 | 0.3031 ± 0.0127 | 0.9535 ± 0.0093 |
| xgboost | notebook hyper-parameters (700 trees) | numeric | 0.5579 ± 0.0563 | 0.3025 ± 0.0113 | 0.9505 ± 0.0099 |
| random_forest | report hyper-parameters, seed 0 | notebook | 0.6279 ± 0.0427 | 0.3808 ± 0.0154 | 0.9380 ± 0.0072 |
| random_forest | report hyper-parameters, seed 1 | notebook | 0.6231 ± 0.0419 | 0.3778 ± 0.0152 | 0.9390 ± 0.0068 |
| random_forest | report hyper-parameters, seed 2 | notebook | 0.6274 ± 0.0441 | 0.3798 ± 0.0162 | 0.9381 ± 0.0075 |
| random_forest | report hyper-parameters, seed 3 | notebook | 0.6350 ± 0.0428 | 0.3828 ± 0.0152 | 0.9366 ± 0.0071 |
| random_forest | report hyper-parameters, seed 4 | notebook | 0.6262 ± 0.0422 | 0.3804 ± 0.0153 | 0.9384 ± 0.0070 |
| random_forest | report hyper-parameters, seed 0 | numeric | 0.6368 ± 0.0374 | 0.3831 ± 0.0107 | 0.9361 ± 0.0079 |
