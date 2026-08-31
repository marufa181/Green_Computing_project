# Results: Synthetic (Before) vs. Real Data (Now)

## Forecasting

| Model | MAE (Synthetic) | MAE (Real) | RMSE (Synthetic) | RMSE (Real) | MAPE (Synthetic) | MAPE (Real) |
|---|---|---|---|---|---|---|
| LSTM | 8.31 | 8.69 | 9.76 | 11.23 | 1.26% | 1.32% |
| Linear Regression | **4.75** | **8.28** | **5.97** | **10.53** | **0.73%** | **1.26%** |
| Persistence | 10.64 | 13.93 | 13.06 | 17.14 | 1.62% | 2.12% |

**Takeaway:** Linear Regression still wins on both datasets, but the LSTM is much more competitive on real data (8.69 vs. 8.28) than it was on the smoother synthetic series (8.31 vs. 4.75) — exactly what you'd expect, since real data is noisier and less purely periodic.

## Carbon-Aware Scheduler

| Policy | Carbon Savings (Synthetic) | Carbon Savings (Real) | SLA Miss (Synthetic) | SLA Miss (Real) |
|---|---|---|---|---|
| Round-Robin | 0.02% | -0.02% | 6.07% | 1.70% |
| Time-of-Day | ≈0% | 0.07% | 6.40% | 1.27% |
| **Carbon-Aware (proposed)** | **0.21%** | **0.54%** | 7.23% | 2.07% |

**Takeaway:** The carbon savings from our proposed scheduler are *larger* on real data (0.54% vs. 0.21%) — because the real grid's carbon intensity swings much more (std. dev. 34.2 vs. 12.3 gCO2/kWh in the synthetic version), giving the scheduler more genuine variation to exploit.

## Bottom line for your paper / presentation
Both the direction and the story of your results **hold up on real data** — if anything, the real data makes the core finding (temporal carbon-aware scheduling gives modest-but-real savings) a bit stronger. You do not need to walk back any earlier claims; you just update the numbers and change "synthetic" to "real" throughout.
