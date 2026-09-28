"""
03_carbon_aware_scheduler_simulation.py

Simulates a stream of delay-tolerant compute jobs arriving during the
held-out TEST period, and compares four scheduling policies:

  1. Run-Immediately   -- execute at arrival hour (FCFS, spill if full)
  2. Round-Robin       -- cycle jobs across hours in [arrival, deadline],
                          ignoring carbon intensity
  3. Time-of-Day Heuristic -- always aim for the fixed hour-of-day with the
                          lowest historical (training-period) average CI
  4. Carbon-Aware Scheduler (proposed) -- pick the lowest ESTIMATED-CI hour
                          within [arrival, deadline] that still has spare
                          capacity. Estimated CI = the LSTM's forecast for
                          hours <=6h ahead of "now"; for hours further out,
                          falls back to the training-period hour-of-day
                          climatology average (a common practical choice
                          when a forecast horizon is limited).

All four policies are evaluated on the SAME job stream and the SAME
per-hour execution capacity, using the ACTUAL (not forecast) carbon
intensity to compute the real carbon emitted once a job is placed.

Outputs:
  - ../results/scheduler_comparison.csv
  - ../results/scheduler_comparison.png
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

ROOT = Path(__file__).resolve().parent.parent   # project root (works from any folder)
DATA_PATH = ROOT / "data" / "bd_grid_carbon_intensity_real.csv"
FORECASTS_PATH = ROOT / "results" / "test_period_forecasts.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
HORIZON = 6
CAPACITY_PER_HOUR = 2   # max jobs that can start execution in any given hour
N_JOBS = 3000

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
full_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
fc = pd.read_csv(FORECASTS_PATH, parse_dates=["timestamp"])

# Training-period hour-of-day climatology (used as (a) fallback for forecasts
# beyond the 6h horizon, and (b) the whole basis of the Time-of-Day baseline)
n_train = int(len(full_df) * 0.8)
train_df = full_df.iloc[:n_train]
climatology = train_df.groupby(train_df["timestamp"].dt.hour)["carbon_intensity_gco2_per_kwh"].mean()
best_hour_of_day = int(climatology.idxmin())
print(f"Training-period climatology: lowest-CI hour-of-day = {best_hour_of_day}:00 "
      f"({climatology.min():.1f} gCO2/kWh avg)")

# Build a lookup: timestamp -> actual CI (for the whole test period, so we can
# look arbitrarily far ahead of any job's arrival time)
test_period = full_df[full_df["timestamp"].isin(fc["timestamp"])].reset_index(drop=True)
# fc rows are a subset (windows need LOOKBACK history) -- use fc's own actual_ci
# column plus full_df for any timestamps slightly outside fc's range.
actual_ci_by_ts = dict(zip(full_df["timestamp"], full_df["carbon_intensity_gco2_per_kwh"]))
forecast_row_by_ts = fc.set_index("timestamp")

sim_timestamps = fc["timestamp"].reset_index(drop=True)
sim_start, sim_end = sim_timestamps.min(), sim_timestamps.max()
print(f"Simulation window: {sim_start} to {sim_end}  ({len(sim_timestamps)} hours)")

# ---------------------------------------------------------------
# 2. Generate synthetic job stream (delay-tolerant batch jobs)
# ---------------------------------------------------------------
usable_hours = (sim_end - sim_start).total_seconds() / 3600 - HORIZON - 12  # leave room for deadlines
arrival_offsets = rng.integers(0, int(usable_hours), size=N_JOBS)
arrival_times = sim_start + pd.to_timedelta(arrival_offsets, unit="h")
energy_kwh = rng.lognormal(mean=0.6, sigma=0.5, size=N_JOBS)   # ~1-3 kWh typical batch job
deadline_hours = rng.integers(2, 13, size=N_JOBS)              # deadline: 2-12h after arrival

jobs = pd.DataFrame({
    "job_id": np.arange(N_JOBS),
    "arrival": arrival_times,
    "energy_kwh": energy_kwh,
    "deadline_hours": deadline_hours,
})
jobs["deadline_time"] = jobs["arrival"] + pd.to_timedelta(jobs["deadline_hours"], unit="h")
print(f"\nGenerated {N_JOBS} jobs. Mean energy/job: {energy_kwh.mean():.2f} kWh, "
      f"mean deadline: {deadline_hours.mean():.1f}h")

# ---------------------------------------------------------------
# 3. Estimated-CI lookup used by the proposed scheduler at decision time
# ---------------------------------------------------------------
def estimated_ci(now_ts, target_ts):
    """CI estimate for target_ts, as available at decision time now_ts."""
    delta_h = int(round((target_ts - now_ts).total_seconds() / 3600))
    if delta_h <= 0:
        return actual_ci_by_ts.get(now_ts, np.nan)
    if now_ts in forecast_row_by_ts.index and delta_h <= HORIZON:
        return forecast_row_by_ts.loc[now_ts, f"rf_forecast_h{delta_h}"]
    # fallback: training-period hour-of-day climatology
    return climatology.loc[target_ts.hour]

# ---------------------------------------------------------------
# 4. Scheduling policies
# ---------------------------------------------------------------
def run_policy(name, choose_hour_fn):
    capacity_used = {}  # ts -> count
    records = []
    for row in jobs.itertuples():
        candidate_hours = pd.date_range(row.arrival, row.deadline_time, freq="h")
        chosen, sla_violation = choose_hour_fn(row, candidate_hours, capacity_used)
        capacity_used[chosen] = capacity_used.get(chosen, 0) + 1
        actual_ci = actual_ci_by_ts.get(chosen, np.nan)
        if pd.isna(actual_ci):
            # overflow spilled past the end of our known-CI range -> use the
            # last available actual value as a reasonable stand-in
            actual_ci = full_df["carbon_intensity_gco2_per_kwh"].iloc[-1]
        delay_h = (chosen - row.arrival).total_seconds() / 3600
        records.append((row.job_id, chosen, actual_ci, delay_h, sla_violation))
    res = pd.DataFrame(records, columns=["job_id", "assigned_hour", "actual_ci", "delay_h", "sla_violation"])
    res["carbon_g"] = res["actual_ci"] * jobs.set_index("job_id").loc[res["job_id"], "energy_kwh"].values
    total_carbon_kg = res["carbon_g"].sum() / 1000
    print(f"[{name:20s}] total carbon={total_carbon_kg:9.2f} kgCO2  "
          f"avg delay={res['delay_h'].mean():5.2f}h  "
          f"SLA violations={res['sla_violation'].mean()*100:5.2f}%")
    return name, res, total_carbon_kg

OVERFLOW_SEARCH_HOURS = 48  # how far past the deadline a job may be pushed

def has_room(ts, capacity_used):
    return capacity_used.get(ts, 0) < CAPACITY_PER_HOUR

def pick_with_fallback(candidate_hours, capacity_used, order, deadline_time):
    """order: candidate hours (within [arrival, deadline]) sorted by
    preference. If every in-window hour is saturated, spill forward past
    the deadline until a free slot is found -- this is what actually
    produces a genuine SLA violation."""
    for ts in order:
        if has_room(ts, capacity_used):
            return ts, False
    overflow = pd.date_range(deadline_time + pd.Timedelta(hours=1),
                              deadline_time + pd.Timedelta(hours=OVERFLOW_SEARCH_HOURS), freq="h")
    for ts in overflow:
        if has_room(ts, capacity_used):
            return ts, True
    # extremely unlikely: still nothing free -> force into least-loaded overflow hour
    return min(overflow, key=lambda t: capacity_used.get(t, 0)), True

# --- Policy 1: Run-Immediately (FCFS, spill forward if arrival hour is full)
def choose_run_immediately(row, candidate_hours, capacity_used):
    order = list(candidate_hours)  # arrival first, then next hours if needed
    return pick_with_fallback(candidate_hours, capacity_used, order, row.deadline_time)

# --- Policy 2: Round-Robin across the candidate window (carbon-agnostic)
_rr_counter = {"i": 0}
def choose_round_robin(row, candidate_hours, capacity_used):
    n = len(candidate_hours)
    order_start = _rr_counter["i"] % n
    order = list(candidate_hours[order_start:]) + list(candidate_hours[:order_start])
    _rr_counter["i"] += 1
    return pick_with_fallback(candidate_hours, capacity_used, order, row.deadline_time)

# --- Policy 3: Time-of-Day heuristic (fixed target hour-of-day, carbon-naive)
def choose_time_of_day(row, candidate_hours, capacity_used):
    on_target = [t for t in candidate_hours if t.hour == best_hour_of_day]
    order = on_target + [t for t in candidate_hours if t not in on_target]
    return pick_with_fallback(candidate_hours, capacity_used, order, row.deadline_time)

# --- Policy 4: Proposed carbon-aware scheduler (ML forecast-driven,
# load-aware). A purely greedy "always pick the single lowest-CI hour"
# variant was tested first and, as expected from the demand-response
# literature, produced a "herding" effect: many jobs converged on the same
# popular low-carbon hour, driving up SLA violations. This version adds a
# small congestion penalty so the scheduler still favours low-carbon hours
# but spreads load when a preferred hour is already filling up.
LOAD_PENALTY_PER_JOB = 15.0  # gCO2/kWh-equivalent penalty per job already booked in that hour

def choose_carbon_aware(row, candidate_hours, capacity_used):
    now_ts = row.arrival
    def score(t):
        return estimated_ci(now_ts, t) + LOAD_PENALTY_PER_JOB * capacity_used.get(t, 0)
    scored = sorted(candidate_hours, key=score)
    return pick_with_fallback(candidate_hours, capacity_used, scored, row.deadline_time)

print("\n=== Running scheduler simulation (4 policies, same job stream) ===")
results = {}
for name, fn in [("Run-Immediately", choose_run_immediately),
                 ("Round-Robin", choose_round_robin),
                 ("Time-of-Day", choose_time_of_day),
                 ("Carbon-Aware (proposed)", choose_carbon_aware)]:
    _rr_counter["i"] = 0
    n, res, total_carbon = run_policy(name, fn)
    results[n] = (res, total_carbon)

# ---------------------------------------------------------------
# 5. Summary table + % savings vs Run-Immediately
# ---------------------------------------------------------------
baseline_carbon = results["Run-Immediately"][1]
summary_rows = []
for name, (res, total_carbon) in results.items():
    pct_savings = 100 * (baseline_carbon - total_carbon) / baseline_carbon
    summary_rows.append({
        "policy": name,
        "total_carbon_kgCO2": round(total_carbon, 2),
        "pct_carbon_savings_vs_run_immediately": round(pct_savings, 2),
        "avg_delay_hours": round(res["delay_h"].mean(), 2),
        "sla_violation_pct": round(res["sla_violation"].mean() * 100, 2),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{RESULTS_DIR}/scheduler_comparison.csv", index=False)
print("\n=== Summary ===")
print(summary.to_string(index=False))

# ---------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].bar(summary["policy"], summary["total_carbon_kgCO2"], color="#3a7d44")
axes[0].set_title("Total Carbon Emitted (kgCO2)")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(summary["policy"], summary["avg_delay_hours"], color="#4472c4")
axes[1].set_title("Average Job Delay (hours)")
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(summary["policy"], summary["sla_violation_pct"], color="#c0504d")
axes[2].set_title("SLA Violation Rate (%)")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/scheduler_comparison.png", dpi=150)
print(f"\nSaved plot to {RESULTS_DIR}/scheduler_comparison.png")
