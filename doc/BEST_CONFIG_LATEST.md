# Best Config Latest

## Repo

`dynamic_port_opt`

## Main Notebook

- `notebook/multi_factor_copula_poc.ipynb`

## Current Default Strategy

Current default strategy:

- `One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure`

Why this is the default now:

- selected low-concentration one-model US/TH configuration from the concentration sweep
- keeps US cap `70%`, TH cap `30%`, stock cap `5%`, concentration penalty `0.02`, and `50` selected assets per side
- keeps Gold/BTC overlay assets and daily exposure active
- baseline row has no AI/theme/segment cap; AI-tech and segment caps are separate guardrail experiments

Sources:

- `result/us_th_one_model_us70_th30_concentration_sweep_summary_thb.csv`
- `result/us_th_one_model_us70_th30_concentration_sweep_period_compare_thb.csv`
- `result/us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv`
- `result/us_th_one_model_us70_th30_concentration_sweep_effective_weights_thb.csv`
- `result/us_th_one_model_us70_th30_concentration_sweep_concentration_history_thb.csv`

Config:

- US group cap: `70.00%`
- TH group cap: `30.00%`
- Stock cap: `5.00%`
- Concentration penalty: `0.02`
- US selected assets: `50`
- TH selected assets when TH signal is on: `50`
- Gold cap: `30.00%`
- BTC cap: `10.00%`
- Daily exposure: `True`

Full-period THB metrics:

- Period: `2007-12-31` to `2026-04-29`
- Total Return: `29.2300`
- CAGR: `19.77%`
- Annual Vol: `16.05%`
- Sharpe: `1.0179`
- Sortino: `1.4737`
- Max Drawdown: `-18.01%`
- Hit Rate: `0.5426`

10Y THB baseline check:

- Period: `2016-04-29` to `2026-04-29`
- CAGR: `26.39%`
- Sharpe: `1.3896`
- Max Drawdown: `-18.01%`

Average effective weights:

- US stocks: `60.13%`
- TH stocks: `10.23%`
- Gold: `12.90%`
- BTC: `2.16%`
- Cash / Reduced Exposure: `14.58%`

Latest effective weights on `2026-04-29`:

- US stocks: `60.00%`
- TH stocks: `30.00%`
- Gold: `5.00%`
- BTC: `0.00%`
- Cash / Reduced Exposure: `5.00%`

Latest holdings:

| Asset | Effective Weight |
|---|---:|
| `GC=F` | `5.00%` |
| `Cash / Reduced Exposure` | `5.00%` |
| `AMAT` | `5.00%` |
| `COST` | `5.00%` |
| `CVX` | `5.00%` |
| `GEV` | `5.00%` |
| `INTC` | `5.00%` |
| `JNJ` | `5.00%` |
| `LRCX` | `5.00%` |
| `MRK` | `5.00%` |
| `MU` | `5.00%` |
| `TXN` | `5.00%` |
| `WMT` | `5.00%` |
| `XOM` | `5.00%` |
| `PTTEP.BK` | `4.34%` |
| `BCP.BK` | `4.32%` |
| `TOP.BK` | `4.30%` |
| `HANA.BK` | `4.27%` |
| `DELTA.BK` | `4.27%` |
| `IVL.BK` | `4.27%` |
| `PTTGC.BK` | `4.23%` |

Concentration diagnostics:

- Latest max single-stock weight: `5.00%`
- Latest top-5 stock weight: `25.00%`
- Latest top-10 stock weight: `50.00%`
- Latest effective stock count: `23.3329`

## Previous Preferred Baseline

- `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure` is retained in `best_confirmed_joint_config` for historical comparison and notebook compatibility, but it is no longer the default strategy.
