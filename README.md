# Carbon-Aware Computing for Bangladesh's Power Grid

**Machine Learning-Based Carbon Intensity Forecasting and Carbon-Aware Task Scheduling for Sustainable Cloud Computing in Bangladesh**

Green Computing course project — United International University (UIU)

**Team:** Marufa Akter · Jakia Tunnesa Rifa · MD. Abdullah · Fahim Faysal · MD. Hasibul Hasan Emon

---

## 1. What is this project about?

Electricity is not equally "dirty" at every hour of the day. When more power comes from coal and oil, one kWh carries more CO₂ than when it comes from gas, hydro, or solar. This CO₂ per kWh is called the grid's **carbon intensity**.

If a computing job is *not urgent* (a backup, a batch job, model training), we can **run it later, at an hour when the grid is cleaner** — same job, less carbon. This idea is called **carbon-aware computing**, and companies like Google and Microsoft already use it.

**The problem:** those systems depend on live carbon-intensity data services that barely cover South Asia. For Bangladesh there is no ready-made carbon-intensity signal, and its grid (gas + rising coal, small renewable share) is very different from the US/EU grids these tools were built for.

**What we did:** using **real, measured hourly generation data of Bangladesh's grid**, we

1. built an hourly **carbon-intensity index** for Bangladesh,
2. trained models to **forecast** it 1–6 hours ahead, and
3. built a **scheduler** that uses the forecast to move flexible jobs to cleaner hours, and measured how much carbon it saves.

---

## 2. The whole project at a glance

```
 RESEARCH FOUNDATION            STEP 1                  STEP 2                 STEP 3
 proposal + literature   -->    DATASET          -->    FORECASTING     -->    SCHEDULING
 (8 core papers)                real PGCB data           predict carbon         shift flexible jobs
                                -> clean it              intensity 1-6h         to cleaner hours
                                -> carbon intensity      ahead                  and compare policies
                                per hour
                                     |                        |                      |
                                     v                        v                      v
                          bd_grid_carbon_intensity   forecast_metrics.csv    scheduler_comparison.csv
                          _real.csv (21,565 rows)    forecast_vs_actual.png  scheduler_comparison.png
```

---

## 3. Everything done, first to last

| # | Stage | What was done | Output |
|---|---|---|---|
| 0 | Research foundation | Proposal (title, problem statement, Green Computing fit); literature review of 8 core papers; wider list of 25 reference papers | `paper/Research_Paper.docx` (Related Work + References) |
| 1 | Dataset | Took the real PGCB hourly generation data, cleaned it, computed carbon intensity for every hour | `scripts/01_load_real_dataset.py`, `data/bd_grid_carbon_intensity_real.csv` |
| 2 | Forecasting | Trained and compared 3 models (Persistence, Linear Regression, Random Forest) on a strict train / validation / test split | `scripts/02_carbon_intensity_forecasting.py`, `results/forecast_*` |
| 3 | Scheduling | Simulated 3,000 flexible jobs and compared 4 scheduling policies using the forecasts | `scripts/03_carbon_aware_scheduler_simulation.py`, `results/scheduler_*` |
| 4 | Write-up | Full research paper (Abstract → Conclusion) using the real results | `paper/Research_Paper.docx` |

All numbers in this README and in the paper come from running the three scripts below on the real dataset. Everything is seeded, so re-running gives the same numbers.

---

## 4. Step 1 — The dataset

### Where the data comes from
**PGCB Hourly Generation Dataset (Bangladesh)** — M. T. Islam, S. A. Turja, A. Habib, UCI Machine Learning Repository, 2025, DOI [10.24432/C59P6V](https://doi.org/10.24432/C59P6V) (CC BY 4.0).

It contains hourly generation and demand for Bangladesh's national grid, split by source: **gas, liquid fuel (oil), coal, hydro, solar, wind**, plus four **import routes** (India–Bheramara HVDC, India–Tripura, India–Adani, Nepal). The raw file covers April 2015 – June 2025 (92,650 rows).

> **How we obtained the file:** the UCI website could not be reached from our coding environment, so we took the identical file from a public GitHub project that uses the same dataset and cites the same UCI source (`rolaseba/energy-demand-forecasting-lstm`), and checked that its columns match the official dataset description exactly. The raw file is included here as `data/PGCB_raw_source.xlsx` so Step 1 can be re-run. Please cite the original dataset (above), not the mirror.

### How we cleaned it (`scripts/01_load_real_dataset.py`)
1. Kept **2023-01-01 onward** — the most complete period.
2. Kept only **on-the-hour** readings and removed duplicate timestamps.
3. Blank `wind`, `india_adani`, `nepal` values (small/new sources) were treated as 0 MW.
4. **Removed 19 impossible rows** (total generation outside 3,000–18,000 MW, i.e. clear data-entry errors).
5. Filled the resulting small gaps (57 hours = **0.26%** of the series, every gap only 1–2 hours) by linear interpolation.

### How carbon intensity is calculated
For each hour: `carbon intensity = Σ (source's share of generation × source's emission factor)`

| Source | gCO₂/kWh | Note |
|---|---|---|
| Coal | 950 | standard value |
| Oil (liquid fuel) | 700 | standard value |
| Gas | 450 | standard value |
| Solar | 40 | life-cycle value |
| Hydro | 20 | life-cycle value |
| Wind | 12 | life-cycle value |
| India–Bheramara, India–Tripura | 700 | broad Indian grid-mix average |
| India–Adani | 950 | comes from one dedicated coal plant, so it is priced as coal |
| Nepal | 20 | Nepal's grid is almost entirely hydro |

### What the final dataset looks like
- **21,565 hourly rows**, 2023-01-01 → 2025-06-17
- Mean **637.1 gCO₂/kWh** (std 34.2), range 479.7 – 850.7
- **Daily pattern:** cleanest around **10:00** (~620), dirtiest around **19:00** (~655)
- **Yearly trend (rising):** 2023 → **613**, 2024 → **649**, 2025 → **663** gCO₂/kWh — matching news reports that coal's share of Bangladesh's generation keeps growing

---

## 5. Step 2 — Forecasting carbon intensity

**Goal:** given the last 48 hours, predict carbon intensity for each of the next **1 to 6 hours**.

**Inputs (per hour):** carbon intensity, demand (MW), and time-of-day / day-of-year encoded as sin/cos.

**Split (by time, no shuffling, no leakage):**

| Part | Period | Use |
|---|---|---|
| Train (80%) | 2023-01-01 → 2024-12-19 | learn |
| Validation (10%) | 2024-12-19 → 2025-03-19 | check |
| Test (10%) | 2025-03-19 → 2025-06-17 | final scores |

**Models compared:**
- **Persistence** — "the next hours will be the same as now" (the simplest possible baseline)
- **Linear Regression** — a straight-line model over the last 48 hours
- **Random Forest** — a tree-ensemble ML model (60 trees, depth 10)

### Results (test period, averaged over the 1–6 hour horizons; lower is better)

| Model | MAE (gCO₂/kWh) | RMSE | MAPE |
|---|---|---|---|
| **Linear Regression** | **8.28** | **10.53** | **1.26%** |
| Random Forest | 9.32 | 11.72 | 1.42% |
| Persistence | 13.93 | 17.14 | 2.12% |

![forecast](results/forecast_vs_actual.png)

**What this tells us**
- Both learned models clearly beat the naive baseline.
- The simple **Linear Regression wins** — Random Forest's own feature importance shows why: the most recent carbon-intensity values account for **95.7%** of what it uses (demand ≈ 1.5%, hour-of-day ≈ 2.5%). Short-term carbon intensity is mostly "what it was a moment ago", which a linear model captures well. More complex ≠ automatically better.

---

## 6. Step 3 — Carbon-aware scheduling

**Setup (simulation on the test period, 2025-03-19 → 2025-06-17, 2,152 hours):**
- **3,000 flexible jobs** arrive at random times; each uses ~2 kWh on average (mean 2.06) and has a **deadline 2–12 hours** after arrival (mean 7 h).
- The system can **start only 2 jobs per hour** (limited capacity), so jobs really compete for good hours.
- Every policy gets the **same jobs and the same capacity**. Carbon is scored with the **actual** carbon intensity of the hour the job ran, so a bad forecast really costs the policy.

**Four policies compared:**

| Policy | How it decides |
|---|---|
| Run-Immediately (baseline) | run as soon as the job arrives (next free hour if full) |
| Round-Robin | spread jobs over the allowed hours, ignoring carbon |
| Time-of-Day | always aim for the historically cleanest hour (10:00) |
| **Carbon-Aware (proposed)** | pick the hour with the lowest **forecast** carbon intensity before the deadline (Random Forest forecast for the next 6 h, typical hour-of-day values beyond that), plus a small penalty for hours that are already filling up — to stop all jobs crowding into the same "clean" hour |

### Results

| Policy | Total carbon (kgCO₂) | Saving vs Run-Immediately | Avg delay | Missed deadlines |
|---|---|---|---|---|
| Run-Immediately | 4107.46 | — | 0.45 h | 1.23% |
| Round-Robin | 4108.19 | −0.02% | 3.57 h | 1.70% |
| Time-of-Day | 4104.40 | 0.07% | 0.69 h | 1.27% |
| **Carbon-Aware (proposed)** | **4085.01** | **0.55%** | 5.11 h | 2.13% |

![scheduler](results/scheduler_comparison.png)

**What this tells us**
- The proposed scheduler emits the **least carbon** of all four policies.
- The saving is **small (0.55%)** and it is **not free**: jobs wait longer (5.1 h vs 0.45 h) and slightly more deadlines are missed.
- The reason the saving is small is a finding in itself: Bangladesh's grid is still almost entirely fossil-based, so the carbon intensity moves only within a modest band during the day (std 34 gCO₂/kWh). Grids with lots of solar/wind swing much more, which is where carbon-aware scheduling saves the most.
- **Why the "crowding penalty"?** A purely greedy version (always pick the cleanest forecast hour) saves slightly more carbon (0.64%) but too many jobs pile into the same clean hours and **3.40%** miss their deadline. Adding a small penalty for already-busy hours (`LOAD_PENALTY_PER_JOB = 15`) gives 0.55% saving with only **2.13%** missed deadlines (`results/ablation_crowding_penalty.csv`; reproduce by setting `LOAD_PENALTY_PER_JOB = 0` in script 03). The value 15 was chosen up front, not tuned.

---

## 7. Key findings (summary)

1. Real hourly data lets us build a carbon-intensity index for Bangladesh that did not exist as a ready signal.
2. Bangladesh's grid carbon intensity is **rising year by year** (613 → 649 → 663 gCO₂/kWh, 2023–2025).
3. Short-term carbon intensity is very predictable; a plain **Linear Regression** matches or beats a Random Forest.
4. Carbon-aware scheduling **works but saves little on today's Bangladesh grid (0.55%)**, and costs some delay — its benefit is limited by how much the grid's carbon intensity actually varies.

---

## 8. Repository structure

```
.
├── README.md                                  <- this file
├── data/
│   ├── PGCB_raw_source.xlsx                   <- raw real dataset (Step 1 input)
│   └── bd_grid_carbon_intensity_real.csv      <- cleaned dataset + carbon intensity (Step 1 output)
├── scripts/
│   ├── 01_load_real_dataset.py                <- Step 1: clean data, compute carbon intensity
│   ├── 02_carbon_intensity_forecasting.py     <- Step 2: train/evaluate forecasting models
│   └── 03_carbon_aware_scheduler_simulation.py<- Step 3: simulate & compare scheduling policies
├── results/
│   ├── forecast_metrics.csv                   <- accuracy per model and per forecast hour
│   ├── forecast_metrics_overall.csv           <- accuracy averaged over 1-6 h
│   ├── rf_feature_importance.csv              <- what the Random Forest relied on
│   ├── test_period_forecasts.csv              <- forecasts saved for Step 3
│   ├── forecast_vs_actual.png                 <- forecast plot
│   ├── scheduler_comparison.csv               <- scheduling results
│   └── scheduler_comparison.png               <- scheduling plot
└── paper/
    └── Research_Paper.docx                    <- full research paper
```

---

## 9. How to run

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl

python scripts/01_load_real_dataset.py                 # ~seconds  -> data/bd_grid_carbon_intensity_real.csv
python scripts/02_carbon_intensity_forecasting.py      # ~1-3 min  -> results/forecast_*, test_period_forecasts.csv
python scripts/03_carbon_aware_scheduler_simulation.py # ~seconds  -> results/scheduler_*
```

Run them **in this order** (Step 3 reads the forecasts saved by Step 2). Scripts use paths relative to the project folder, so they run from any location. All random seeds are fixed, so you should get exactly the numbers above.

---

## 10. Honest notes and limitations

- **Data source:** the real PGCB dataset was obtained through a public GitHub copy (see Section 4) because the UCI site was unreachable from our environment; the column structure matches the official dataset. Cite the original UCI dataset.
- **Random Forest instead of LSTM:** we originally planned an LSTM neural network, but our environment could not install the deep-learning library (disk space). Random Forest is a standard, strong method for this kind of data. Trying an LSTM on a bigger machine is a clear next step.
- **Emission factors** are standard literature values, not plant-by-plant measurements for Bangladesh. Import routes use route-specific reasoning (Adani = coal, Nepal = hydro) but are still estimates.
- **The jobs in Step 3 are simulated** (random arrivals, sizes, deadlines, and a capacity of 2 jobs/hour), not real cloud traces. The 0.55% figure applies to this setup.
- **Possible future work:** try an LSTM/other sequence models; test other capacities and job mixes; a joint carbon-and-deadline optimiser instead of the simple crowding penalty; test on real workload traces.

---

## 11. References (core literature review)

1. A. Radovanović et al., "Carbon-Aware Computing for Datacenters," *IEEE Trans. Power Systems*, vol. 38, no. 2, pp. 1270–1280, 2022.
2. Microsoft & Green Software Foundation, "Carbon-Aware Computing White Paper," 2023.
3. "CarbonCast: Multi-Day Forecasting of Electric Grid Carbon Intensity Using Machine Learning," *ACM Energy Informatics Review*.
4. "A Node-Aware Graph Neural Network-Based Carbon Intensity Forecasting Model for Cross-Border Power Grids," ScienceDirect.
5. "Green Task: A Carbon-Aware Scheduling Algorithm for Edge-Fog Networks," IJSAT, 2025.
6. A. Beloglazov, J. Abawajy, R. Buyya, "Energy-Aware Resource Allocation Heuristics for Efficient Management of Data Centers for Cloud Computing," *Future Generation Computer Systems*, vol. 28, no. 5, pp. 755–768, 2012.
7. A. P. Piyal et al., "Energy Demand Forecasting Using Machine Learning Perspective Bangladesh," IEEE GlobConHT, 2023.
8. M. T. Islam, S. A. Turja, A. Habib, "PGCB Hourly Generation Dataset (Bangladesh)," UCI Machine Learning Repository, 2025, DOI: 10.24432/C59P6V.
