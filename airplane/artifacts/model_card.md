# Model Card: airplane

## Task
Regression — target: `price`.

## Features
age, pass, fixgear, tdrag, horse, fuel, ceiling, cruise

## Estimator
RandomForestRegressor

## Holdout metrics
- R2: 0.9292
- RMSE: 10846.3617
- MAE: 7273.8823
- Test rows: 39

## Notes
All preprocessing (log scaling, standardization, encoding) is embedded in the
persisted scikit-learn pipeline. Metrics computed on a held-out split.
