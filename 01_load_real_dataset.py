"""
01_load_real_dataset.py

Loads the REAL PGCB Hourly Generation Dataset (Bangladesh) -- Islam, Turja &
Habib (2025), UCI ML Repository, DOI: 10.24432/C59P6V -- and derives an
hourly carbon-intensity index from it.

DATA PROVENANCE: This exact dataset could not be downloaded directly from
UCI/Mendeley inside the sandboxed execution environment (no internet access
to archive.ics.uci.edu or data.mendeley.com). It was instead obtained from
a public GitHub repository (rolaseba/energy-demand-forecasting-lstm) whose
README explicitly cites the same UCI dataset (id 1175) as its source, and
whose columns exactly match the dataset's documented schema (gas,
liquid_fuel, coal, hydro, solar, wind, india_bheramara_hvdc, india_tripura,
india_adani, nepal). This is REAL, measured grid data -- not synthetic.

Cleaning steps (all documented in the paper's Methodology):
  1. Keep only on-the-hour (:00) timestamps, 2023-01-01 onward (most
     complete / highest-quality period), drop duplicate timestamps.
  2. Fill missing wind / india_adani / nepal with 0 (columns are heavily
     zero/absent in periods before those sources existed or reported).
  3. Flag and null out rows where total fuel-mix MW is outside a plausible
     range for Bangladesh's grid (3,000-18,000 MW) -- removes a small
     number (~16) of clear data-entry/scraping artifacts.
  4. Reindex to a continuous hourly index and linearly interpolate the
     small number of resulting gaps (<0.3% of hours, all 1-2 hours long).

Output: ../data/bd_grid_carbon_intensity_real.csv
"""

import numpy as np
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # project root (works from any folder)
SRC = ROOT / "data" / "PGCB_raw_source.xlsx"
OUT = ROOT / "data" / "bd_grid_carbon_intensity_real.csv"

FUEL_COLS = ["gas", "liquid_fuel", "coal", "hydro", "solar", "wind",
             "india_bheramara_hvdc", "india_tripura", "india_adani", "nepal"]

# Emission factors (gCO2/kWh). Same standard values as our earlier synthetic
# version for gas/coal/oil/hydro/solar/wind. Import links are now given
# SOURCE-SPECIFIC factors, since the real data breaks imports out by route:
#   - india_bheramara_hvdc, india_tripura: broad Indian grid-mix average
#   - india_adani: dedicated coal plant (Godda, Jharkhand) built to supply
#     Bangladesh -> use the coal factor, not a blended grid average
#   - nepal: Nepal's grid is overwhelmingly hydroelectric -> use the hydro
#     factor
EF = {
    "gas": 450.0, "liquid_fuel": 700.0, "coal": 950.0, "hydro": 20.0,
    "solar": 40.0, "wind": 12.0,
    "india_bheramara_hvdc": 700.0, "india_tripura": 700.0,
    "india_adani": 950.0, "nepal": 20.0,
}

# ---------------------------------------------------------------
# 1. Load & filter to on-the-hour timestamps, 2023 onward
# ---------------------------------------------------------------
raw = pd.read_excel(SRC)
raw["datetime"] = pd.to_datetime(raw["datetime"])
df = raw[(raw["datetime"] >= "2023-01-01") & (raw["datetime"].dt.minute == 0)].copy()
df = df.drop_duplicates(subset="datetime", keep="first").sort_values("datetime").reset_index(drop=True)
print(f"After :00-only filter + dedup: {len(df)} rows "
      f"({df['datetime'].min()} to {df['datetime'].max()})")

# ---------------------------------------------------------------
# 2. Fill sparse columns with 0 (absence = not yet online / not reported)
# ---------------------------------------------------------------
for c in ["wind", "india_adani", "nepal"]:
    df[c] = df[c].fillna(0)

# ---------------------------------------------------------------
# 3. Flag implausible rows (data-entry/scraping artifacts) as missing
# ---------------------------------------------------------------
fuel_sum = df[FUEL_COLS].sum(axis=1)
bad = (fuel_sum < 3000) | (fuel_sum > 18000)
print(f"Flagging {bad.sum()} rows outside the plausible 3,000-18,000 MW range "
      f"as data errors -> will be interpolated")
df.loc[bad, FUEL_COLS] = np.nan
df.loc[bad, "demand_mw"] = np.nan

# ---------------------------------------------------------------
# 4. Reindex to a continuous hourly index; interpolate small gaps
# ---------------------------------------------------------------
full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
df = df.set_index("datetime").reindex(full_index)
n_missing_before = df[FUEL_COLS].isna().any(axis=1).sum()
print(f"Hours needing interpolation (originally missing + flagged bad): {n_missing_before} "
      f"({100 * n_missing_before / len(df):.2f}% of hours)")

for c in FUEL_COLS + ["demand_mw", "load_shedding"]:
    df[c] = df[c].interpolate(method="linear", limit=6)

df = df.dropna(subset=FUEL_COLS)  # drop any rare gap too long to interpolate
df.index.name = "timestamp"
df = df.reset_index()

# ---------------------------------------------------------------
# 5. Carbon-intensity index = weighted sum of (fuel MW / total MW) * EF
# ---------------------------------------------------------------
total_mw = df[FUEL_COLS].sum(axis=1)
carbon_intensity = sum(df[c] / total_mw * EF[c] for c in FUEL_COLS)

out = pd.DataFrame({
    "timestamp": df["timestamp"],
    "demand_mw": df["demand_mw"].round(1),
    "loadshedding_mw": df["load_shedding"].round(1),
    "gas_share": (df["gas"] / total_mw).round(4),
    "coal_share": (df["coal"] / total_mw).round(4),
    "oil_share": (df["liquid_fuel"] / total_mw).round(4),
    "hydro_share": (df["hydro"] / total_mw).round(4),
    "wind_share": (df["wind"] / total_mw).round(4),
    "solar_share": (df["solar"] / total_mw).round(4),
    "import_share": ((df["india_bheramara_hvdc"] + df["india_tripura"]
                       + df["india_adani"] + df["nepal"]) / total_mw).round(4),
    "carbon_intensity_gco2_per_kwh": carbon_intensity.round(2),
})

out.to_csv(OUT, index=False)
print(f"\nWrote {len(out)} rows to {OUT}")
print(out.head())
print("\nCarbon intensity stats (gCO2/kWh):")
print(out["carbon_intensity_gco2_per_kwh"].describe())
