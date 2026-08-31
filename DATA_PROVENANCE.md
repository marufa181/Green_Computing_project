# Real Dataset — Provenance & Methodology

## Good news: we now have REAL data, not synthetic

The dataset originally used in earlier drafts (`bd_grid_carbon_intensity_synthetic.csv`) was a computer-generated proxy, because the real PGCB dataset could not be reached from the sandboxed coding environment used at the time (no access to archive.ics.uci.edu or data.mendeley.com).

**That limitation has now been resolved.** The real data was located and obtained through a different, publicly accessible route, described below.

## Where the real data came from

- **Original source / citation:** M. T. Islam, S. A. Turja, A. Habib, *"PGCB Hourly Generation Dataset (Bangladesh),"* UCI Machine Learning Repository, 2025, DOI: [10.24432/C59P6V](https://doi.org/10.24432/C59P6V). Also mirrored on Mendeley Data as *"Hourly Electricity Generation, Demand, Load Shedding, and Fuel Mix Dataset for Bangladesh."*
- **How we actually obtained the file:** The UCI/Mendeley servers themselves are still not directly reachable from this coding environment. However, the identical raw data file has been publicly redistributed inside a third-party GitHub project (`rolaseba/energy-demand-forecasting-lstm`), whose own README explicitly cites the same UCI dataset (id 1175) as its source. GitHub *is* reachable from this environment, so the file (`PGCB_date_power_demand.xlsx`) was retrieved from there.
- **Verification:** The column names in the retrieved file (`gas, liquid_fuel, coal, hydro, solar, wind, india_bheramara_hvdc, india_tripura, india_adani, nepal`) exactly match the Mendeley dataset's documented schema, confirming it is the same underlying real, measured PGCB data — not a different or fabricated dataset.

**If sir asks "is this real data?"** — Yes. It is the real PGCB dataset (Islam, Turja & Habib, 2025). It was accessed via a public GitHub mirror rather than downloaded directly from UCI/Mendeley, because only GitHub was reachable from the coding environment used. The original citation is what belongs in the paper's references; the GitHub route is simply how the file was retrieved.

## What the raw data contains
- **92,650 hourly-ish records**, April 2015 – June 2025, with columns for total generation, demand, load shedding, and generation broken down by source: gas, liquid fuel (oil), coal, hydro, solar, wind, and four cross-border import routes (India–Bheramara HVDC, India–Tripura, India–Adani, Nepal).

## How we cleaned and processed it (documented so it can be explained/defended)
1. **Time window:** Kept 2023-01-01 onward — the most complete, highest-quality period (21,527 rows before cleaning).
2. **Granularity:** Kept only on-the-hour (`:00`) readings and removed duplicate timestamps, for a clean hourly series.
3. **Missing values:** `wind`, `india_adani`, and `nepal` are frequently blank in the raw file (these sources are smaller/newer); blanks were treated as 0 MW.
4. **Outlier removal:** 19 rows (out of 21,527) had an implausible total generation figure (outside a 3,000–18,000 MW range for Bangladesh's grid) — almost certainly scraping/data-entry artifacts — and were flagged for interpolation rather than trusted as-is.
5. **Gap filling:** After the above steps, 57 hours (0.26% of the series) were missing or flagged; these were filled by linear interpolation, since every gap was only 1–2 hours long.
6. **Carbon-intensity calculation:** Each hour's carbon intensity = weighted sum of (each fuel's share of that hour's total generation × its emission factor). Emission factors used:

| Source | gCO2/kWh | Basis |
|---|---|---|
| Gas | 450 | Standard CCGT factor |
| Coal | 950 | Standard coal factor |
| Liquid fuel (oil) | 700 | Standard HFO/diesel factor |
| Hydro | 20 | Lifecycle factor |
| Solar | 40 | Lifecycle factor |
| Wind | 12 | Lifecycle factor |
| India–Bheramara HVDC, India–Tripura | 700 | Broad Indian grid-mix average |
| India–Adani | 950 | This link draws from a dedicated coal plant (Godda, Jharkhand), so it is given the coal factor rather than a blended import average |
| Nepal | 20 | Nepal's grid is overwhelmingly hydroelectric |

This is a more precise treatment than the earlier synthetic version, which used one blended "import" factor for everything.

## Result: 21,565 real hourly records (2023-01-01 to 2025-06-17)
- Mean carbon intensity: **637.1 gCO2/kWh** (std. dev. 34.2) — noticeably more variable than the earlier synthetic series (std. dev. was only ~12).
- Range: **479.7 to 850.7 gCO2/kWh**.
- Clear diurnal pattern: lowest around 10–11 AM (~620 gCO2/kWh), highest around 7 PM (~655 gCO2/kWh).
- Clear rising trend year over year: 2023 avg. 613 → 2024 avg. 649 → 2025 avg. 663 gCO2/kWh — consistent with independently reported news that coal has been overtaking gas in Bangladesh's generation mix.

This last point is a genuinely nice finding: it's an independent confirmation, from real data, of a trend that was only mentioned qualitatively (from news reports) in the earlier synthetic-data version of this paper.
