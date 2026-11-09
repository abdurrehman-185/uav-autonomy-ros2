| Estimator | RMSE with GNSS (m) | RMSE in 60 s outage (m) | Error at outage end (m) | Recovery after GNSS returns (s) | Time per epoch (us) |
|---|---|---|---|---|---|
| Dead reckoning | 32.79 | 197.22 | 337.83 | none | 330 |
| Error-state EKF | 0.75 | 11.08 | 23.51 | 0.4 | 1263 |
| Bootstrap PF | 59.98 (19.04-92.11) | 438.51 (117.65-678.73) | 809.97 (202.39-1205.91) | none | 1084 |
| Roughened PF | 2.57 (2.52-2.76) | 5705.25 (2992.57-15102.89) | 12707.08 (6848.58-33603.26) | none | 1450 |
| Marginalized PF | 3.62 (1.19-4.23) | 167.87 (45.25-187.66) | 345.07 (98.50-380.71) | 2.6 | 3868 |
