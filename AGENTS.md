# dynamic_port_opt

## Repo Purpose

This repository is a proof-of-concept for:

- dynamic multi-factor copula portfolio construction
- HMM-style cluster assignment for equity sleeves
- point-in-time S&P 500 universe selection
- overlay portfolios that combine:
  - S&P 500 daily exposure
  - Gold
  - BTC
  - Static HMM equity sleeve

The main deliverable is the notebook:

- `notebook/multi_factor_copula_poc.ipynb`

That notebook shows:

- backtest metrics
- equity curves
- daily exposure charts
- optimizer weight history
- mix sweep
- momentum attribution
- lookback sweep
- strategic rebalance sweep
- factor ablation

## Important Files

- Core model/backtest logic: `src/dynamic_factor_copula.py`
- Notebook generator: `scripts/build_poc_notebook.py`
- Main notebook: `notebook/multi_factor_copula_poc.ipynb`
- Latest overlay summary: `result/overlay_comparison_summary.csv`
- Latest preferred config record: `result/best_config_latest.json`
- Human-readable best config notes: `doc/BEST_CONFIG_LATEST.md`

## Current Preferred Baseline

The current default/preferred strategy is:

- `One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure`

Why this is the preferred baseline now:

- selected low-concentration one-model US/TH configuration from the concentration sweep
- keeps US cap `70%`, TH cap `30%`, stock cap `5%`, concentration penalty `0.02`, and `50` selected assets per side
- keeps Gold/BTC overlay assets and daily exposure active
- baseline row has no AI/theme/segment cap; AI-tech and segment caps are guardrail experiments

Baseline reproduction numbers to check before new experiments:

- 10Y window (`2016-04-29` to `2026-04-29`): CAGR `26.39%`, Sharpe `1.3896`, Max Drawdown `-18.01%`
- Full period (`2007-12-31` to `2026-04-29`): CAGR `19.77%`, Sharpe `1.0179`, Max Drawdown `-18.01%`
- Source files:
  - `result/us_th_one_model_us70_th30_concentration_sweep_period_compare_thb.csv`
  - `result/us_th_one_model_us70_th30_concentration_sweep_summary_thb.csv`
  - `result/us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv`

## Equity Sleeve Backtest Defaults

Unless a notebook section says otherwise, the main Static/Dynamic HMM backtest uses:

- point-in-time S&P 500 universe
- top `30` liquid names at each rebalance
- `4` clusters
- monthly rebalance for the equity sleeve
- `max_weight = 0.08`
- momentum enabled

## Best Known Results

### Current default one-model US/TH strategy

- Strategy: `One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure`
- Source: `result/us_th_one_model_us70_th30_concentration_sweep_period_compare_thb.csv`
- Full period THB metrics:
  - CAGR: `19.77%`
  - Sharpe: `1.0179`
  - Max Drawdown: `-18.01%`
- 10Y THB metrics:
  - CAGR: `26.39%`
  - Sharpe: `1.3896`
  - Max Drawdown: `-18.01%`
- Latest effective weights source: `result/us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv`

### Historical best confirmed joint config

This historical US-only overlay config is retained for comparison:

- Strategy: `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure`
- Equity sleeve lookback: `504` trading days
- Overlay strategic rebalance: `1` month
- Source files:
  - `result/joint_confirm_603010_504d_1m_overlay_summary_usd.csv`
  - `result/joint_confirm_603010_504d_1m_overlay_summary_thb.csv`
- USD metrics:
  - CAGR: `25.47%`
  - Sharpe: `1.8512`
  - Max Drawdown: `-11.36%`
- THB metrics:
  - CAGR: `24.85%`
  - Sharpe: `1.6490`
  - Max Drawdown: `-12.21%`

### Legacy confirmed baseline run

The older fully embedded notebook baseline is still available in the main overlay summary:

- Strategy: `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure`
- Equity sleeve lookback: `756` trading days
- Overlay strategic rebalance: `3` months
- Source: `result/overlay_comparison_summary.csv`
- CAGR: `25.20%`
- Sharpe: `1.7491`
- Max Drawdown: `-9.81%`

### THB baseline view

THB-translated overlay results are available in:

- `result/overlay_comparison_summary_thb.csv`
- `result/overlay_comparison_curves_thb.csv`

The historical THB baseline counterpart of the previous preferred strategy is:

- `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure`
- CAGR: `24.59%`
- Sharpe: `1.5712`

### Sweep context

The best sweep observations were later validated jointly:

- Lookback sweep best Sharpe:
  - `504` trading days
  - Sharpe: `1.7749`
- Strategic rebalance sweep best Sharpe:
  - `1` month
  - Sharpe: `1.8275`

Those two choices are now confirmed together in the joint config above.

## Thailand PIT Status

Thailand SET100 point-in-time membership logic is wired in code and a local Thai cache can now be built into this repo:

- `src/dynamic_factor_copula.py` supports `universe_mode="set100_pit"`
- membership intervals come from `port_opt_advance/data/thai_stock/set100_ticker_start_end.csv`
- `scripts/build_thai_set100_cache.py` downloads Thailand SET100 history plus `^SET.BK` into:
  - `data/cache/dynamic_factor_copula/extra_prices.parquet`
  - `data/cache/dynamic_factor_copula/extra_volumes.parquet`

Important data note:

- the default `20Y` cache remains US-only
- Thai prices are layered on top through the local extra cache

### Latest Thailand PIT Equity Sleeve Run

Files:

- `result/thai_set100_cache_status.csv`
- `result/thai_set100_pit_metrics.csv`
- `result/thai_set100_pit_universe_history.csv`

Latest tested Thailand config:

- `universe_mode="set100_pit"`
- top `30` liquid SET100 names at each rebalance
- `4` clusters
- lookback `504` trading days
- equity sleeve rebalance `monthly`
- benchmark ticker `^SET.BK`
- no vol proxy ticker
- momentum features `on`
- momentum signal `on`
- `momentum_signal_mode="mom_63"`
- dropped features:
  - `resid_vol = False`
  - `drawdown = False`
  - `downside_beta = False`

Latest Thailand equity sleeve metrics:

- `Dynamic HMM Copula`: CAGR `2.22%`, Sharpe `0.2087`, Max Drawdown `-50.33%`
- `Static Copula`: CAGR `2.12%`, Sharpe `0.2038`, Max Drawdown `-50.56%`

## US + Thailand Blend

The repo also has a blended sleeve experiment script:

- `scripts/run_us_th_blended.py`

It compares:

- US Static HMM sleeve from `sp500_pit`
- Thailand Static HMM sleeve from `set100_pit`
- Gold
- BTC

Latest files:

- `result/us_th_gold_btc_blended_summary_thb.csv`
- `result/us_th_gold_btc_blended_summary_usd.csv`
- `result/us_th_gold_btc_blended_curves_thb.csv`
- `result/us_th_gold_btc_blended_curves_usd.csv`
- `result/latest_us_hmm_members.csv`
- `result/latest_th_hmm_members.csv`

Latest comparison period:

- `2016-01-04` to `2026-04-29`

Latest THB result summary:

- best Sharpe: `US/TH/Gold/BTC 45/15/30/10`
- CAGR: `30.94%`
- Sharpe: `1.9740`
- Max Drawdown: `-10.90%`

Important interpretation:

- this is a shorter overlap-period comparison than the full US-only joint confirm
- adding Thailand reduced CAGR versus US-only in this window, but improved drawdown and gave the best THB Sharpe at a modest `15%` Thailand sleeve

## How To Refresh Results

Regenerate the notebook and embedded outputs with:

- `python scripts\\build_poc_notebook.py`

## Notes For Future Agents

- Keep naming consistent between:
  - notebook cells
  - hydrated notebook outputs
  - result CSV files
- When adding notebook summary tables, configure pandas/display output so all relevant columns are visible; do not leave important metric or weight columns hidden behind truncated display settings.
- Strategy names and descriptions should omit assets with zero or display-rounded-zero weights; keep full weight columns in CSV/metric tables, but keep human-readable labels focused on active allocations.
- Extended allocation experiments should exclude assets with less than `10` years of usable price history unless a section explicitly labels them as short-history/tactical tests.
- Chart titles and legends must not overlap; move dense legends below the plot or otherwise increase margins before saving notebook/html outputs.
- The current naming standard for overlay strategies is:
  - `S&P 500 daily exposure`
  - `S&P/Gold/BTC 60/30/10 daily exposure`
  - `Static HMM daily exposure`
  - `Static HMM/Gold/BTC 60/30/10 daily exposure`
  - `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure`
- If changing the baseline mix again, update both:
  - notebook cell code in `scripts/build_poc_notebook.py`
  - hydrate logic in the same file
