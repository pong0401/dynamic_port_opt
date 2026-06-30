# Best Sharpe Handoff For `port_opt_advance`

This note is the current handoff reference from `dynamic_port_opt` to `port_opt_advance`.

Purpose:

- identify the best-Sharpe strategy in each headline category
- record the exact config used to produce it
- list the main functions involved
- record the metric snapshot
- point to the saved precompute files

Important data-status note:

- use this file as the current source of truth for the categories below
- `result/best_config_latest.json` still contains stale pre-fix values for the old confirmed daily-exposure baseline and should not be treated as authoritative until it is refreshed

## 1. Best Sharpe Buy And Hold

### Strategy

- `S&P 500 buy and hold (full)`

### Config

- asset universe: `S&P 500 benchmark series`
- benchmark source: `source cache benchmark.parquet`
- PIT asset selection: not applicable
- vol clustering / copula / HMM: none
- optimizer: none
- weights: `100%` S&P 500
- momentum: none
- lookback: not applicable
- rebalance: none
- daily exposure: `No`
- report currency: `USD`
- window: `2012-01-01` to `2026-04-30`

### Main Functions

- `curve_from_returns`
- `compute_port_opt_style_metrics`

### Metrics

- Sharpe: `0.7366`
- CAGR: `0.1485`

### Precompute / Source Location

- no dedicated precompute summary file
- source cache:
  - `../port_opt_advance/data/cache/portopt_optimizer_proof/20Y/benchmark.parquet`

### Last Weight Note

- not applicable

## 2. Best Sharpe Buy And Hold S&P + Gold + BTC

### Strategy

- `S&P/Gold/BTC buy and hold 60/30/10`

### Config

- assets:
  - `SPY`
  - `GC=F`
  - `BTC-USD`
- asset selection PIT: none
- vol clustering / copula / HMM: none
- optimizer: none
- weights:
  - `SPY = 60%`
  - `Gold = 30%`
  - `BTC = 10%`
- momentum: none
- lookback: not applicable
- rebalance: `monthly`
- daily exposure: `No`
- report currency: `USD`
- window: `2016-01-01` to `2026-04-29`

### Main Functions

- `load_overlay_compare_prices`
- `compare_rebalanced_portfolio`
- `curve_from_returns`
- `compute_port_opt_style_metrics`

### Metrics

- Sharpe: `0.9200`
- CAGR: `0.1474`

### Precompute / Source Location

- no dedicated precompute summary file
- source cache:
  - `data/cache/dynamic_factor_copula/overlay_compare_prices.parquet`

### Last Weight Note

- deterministic strategic weights only:
  - `SPY = 0.60`
  - `Gold = 0.30`
  - `BTC = 0.10`

## 3. Best Sharpe US Stock

### Strategy

- `Static Copula`

### Config

- function entrypoint: `backtest_dynamic_factor_copula`
- universe mode: `sp500_pit`
- asset selection: top `30` liquid names from true S&P 500 members at each rebalance
- PIT asset selection: `Yes`
- vol clustering: `4 clusters`
- copula: `Static Copula`
- HMM: `No`
- optimizer objective: `mean_variance`
- max weight: `0.08`
- momentum:
  - momentum features: `On`
  - momentum signal: `On`
  - signal mode: `mom_63`
- dropped feature flags:
  - `resid_vol = False`
  - `drawdown = False`
  - `downside_beta = False`
- lookback: `756 trading days`
- rebalance: `ME`
- daily exposure: `No`
- report currency: `USD`
- window: `2012-01-01` to `2026-04-30`

### Main Functions

- `backtest_dynamic_factor_copula`
- `initialize_static_clusters`
- `build_factor_covariance`
- `optimize_portfolio`

### Metrics

- Sharpe: `1.1329`
- CAGR: `0.2529`

### Precompute / Source Location

- summary:
  - `result/multi_factor_copula_metrics.csv`
- weight history:
  - `result/static_copula_weight_history.csv`
- latest weight comparison:
  - `result/latest_weight_comparison.csv`

### Last Weight Note

- use:
  - `result/static_copula_weight_history.csv`
- latest rebalance row gives the last saved portfolio weights

## 4. Best Sharpe Momentum

### Strategy

- `Static HMM mix 60/30/10`

### Config

- equity sleeve source: `results["nav"]["Static Copula"]` from the main US PIT backtest
- overlay side assets:
  - `Gold`
  - `BTC`
- asset selection PIT for the equity sleeve: `Yes`
- vol clustering: `4 clusters` in the underlying equity sleeve
- copula / HMM:
  - sleeve name in the notebook section is `Static HMM with Momentum`
  - practical construction uses the static equity sleeve return stream plus Gold/BTC sleeves in the tested mix sweep code path
- optimizer for the equity sleeve: inherited from the main US PIT run
- strategic weights:
  - `Static sleeve = 60%`
  - `Gold = 30%`
  - `BTC = 10%`
- momentum:
  - equity sleeve momentum features: `On`
  - equity sleeve momentum signal: `On`
  - signal mode: `mom_63`
- lookback: inherited US sleeve lookback `756 trading days`
- rebalance: `monthly` strategic rebalance
- daily exposure:
  - `No` at the category definition level here
  - Gold and BTC sleeves use the tested overlay-return construction from the sweep code
- report currency: `THB`

### Main Functions

- `compare_trend_exposure`
- `compare_apply_returns`
- `compare_rebalanced_portfolio`
- `curve_from_returns`
- `compute_port_opt_style_metrics`

### Metrics

- Sharpe: `1.0796`
- CAGR: `0.1846`

### Precompute / Source Location

- summary:
  - `result/static_hmm_momentum_mix_sweep.csv`
- combined strategic weight history:
  - `result/static_hmm_603010_weight_history.csv`

### Last Weight Note

- use:
  - `result/static_hmm_603010_weight_history.csv`
- latest rebalance row gives the last saved combined sleeve + Gold + BTC weights

## 5. Best Sharpe Daily Exposure

### Strategy

- `Static HMM/Gold/BTC 60/30/10 daily exposure`

### Config

- function entrypoint:
  - `backtest_dynamic_factor_copula`
  - then `build_overlay_comparison`
- universe mode: `sp500_pit`
- asset selection: top `30` liquid names from true S&P 500 members at each rebalance
- PIT asset selection: `Yes`
- vol clustering: `4 clusters`
- copula / HMM:
  - equity sleeve = `Static HMM`
  - HMM-style clustering with momentum-enabled features in the base run
- optimizer objective: `mean_variance`
- max weight: `0.08`
- strategic weights:
  - `Static HMM sleeve = 60%`
  - `Gold = 30%`
  - `BTC = 10%`
- momentum:
  - momentum features: `On`
  - momentum signal: `On`
  - signal mode: `mom_63`
- lookback: `504 trading days`
- rebalance:
  - equity sleeve rebalance: `ME`
  - strategic overlay rebalance: `1 month`
- daily exposure: `Yes`
- daily exposure implementation:
  - signal at close
  - applied next session
  - no same-day close lookahead
- report currency: `USD`
- window: overlap window used by the overlay comparison

### Main Functions

- `backtest_dynamic_factor_copula`
- `build_overlay_comparison`
- `apply_daily_exposure_overlay`
- `compare_sp_exposure`
- `compare_trend_exposure`
- `compare_rebalanced_portfolio`

### Metrics

- Sharpe: `1.0622`
- CAGR: `0.1531`

### Precompute / Source Location

- summary:
  - `result/joint_confirm_603010_504d_1m_overlay_summary_usd.csv`
- THB companion:
  - `result/joint_confirm_603010_504d_1m_overlay_summary_thb.csv`
- curves:
  - `result/joint_confirm_603010_504d_1m_overlay_curves_usd.csv`
  - `result/joint_confirm_603010_504d_1m_overlay_curves_thb.csv`

### Last Weight Note

- use:
  - `result/static_hmm_603010_weight_history.csv`
- note:
  - this file gives the strategic rebalance weights
  - daily exposure is a time-varying overlay cap on top of those strategic weights

## 6. Best Sharpe Cap Weight

### Strategy

- `All-assets static capped [US6/TH6/Gold40/BTC10]`

### Config

- script entrypoint: `scripts/run_us_th_all_asset_cap_sweep.py`
- model family: one static model containing:
  - US equities
  - TH equities
  - Gold
  - BTC
- asset selection:
  - US tickers from `result/latest_us_hmm_members.csv`
  - TH tickers from `result/latest_th_hmm_members.csv`
- PIT asset selection: inherited from the saved latest member lists
- vol clustering: `4 clusters` through `_run_model_on_prices`
- copula: `Static Copula`
- HMM: `No` in the final selected strategy
- optimizer objective: `mean_variance`
- momentum:
  - momentum features: `On`
  - momentum signal: `On`
  - signal mode inside `_run_model_on_prices`: `mom_63`
- lookback: `504 trading days`
- rebalance: `ME`
- daily exposure: `No`
- cap structure:
  - US equity max weight = `0.06`
  - TH equity max weight = `0.06`
  - Gold max weight = `0.40`
  - BTC max weight = `0.10`
- report currency: `THB`
- window: `2017-12-29` to `2026-04-29`

### Main Functions

- `run_us_th_all_asset_cap_sweep.py`
- `_load_thb_panel`
- `_run_model_on_prices`
- `optimize_portfolio`
- `compute_port_opt_style_metrics`

### Metrics

- Sharpe: `1.4935`
- CAGR: `0.3741`

### Precompute / Source Location

- summary:
  - `result/us_th_all_asset_cap_sweep_summary_thb.csv`
- curves:
  - `result/us_th_all_asset_cap_sweep_curves_thb.csv`
- latest weights:
  - `result/us_th_all_asset_cap_sweep_latest_weights_thb.csv`

### Last Weight Note

- use:
  - `result/us_th_all_asset_cap_sweep_latest_weights_thb.csv`
- filter:
  - `Case = US6/TH6/Gold40/BTC10`

## 7. Best Sharpe All

### Strategy

- same as section 6:
  - `All-assets static capped [US6/TH6/Gold40/BTC10]`

### Why It Is The Overall Winner

- highest Sharpe across the rerun result families reviewed in `dynamic_port_opt`
- includes the broadest in-model opportunity set among the reviewed winners:
  - US equities
  - TH equities
  - Gold
  - BTC

### Config

- same config as section 6

### Main Functions

- same function chain as section 6

### Metrics

- Sharpe: `1.4935`
- CAGR: `0.3741`

### Precompute / Source Location

- same as section 6

### Last Weight Note

- same as section 6

## 8. Best Objective Inside The Top 5 Cap Cases

This is an extra helper section for `port_opt_advance` because the top cap cases were re-tested across the objective menu.

### Winner

- `All-assets static capped [US6/TH6/Gold40/BTC10] [mean_variance]`

### Config

- same cap structure as section 6
- objective set tested:
  - `mean_variance`
  - `max_sharpe_mom`
  - `min_vol_mom_tilt`
  - `risk_parity_mom_tilt`
- top-5 cap cases by Sharpe were selected first, then objective sweep was run on those 5 cases

### Main Functions

- `run_us_th_all_asset_cap_top5_objective_sweep.py`
- `_run_model_on_prices`
- `compute_port_opt_style_metrics`

### Metrics

- Sharpe: `1.4935`
- CAGR: `0.3741`

### Precompute / Source Location

- full sweep:
  - `result/us_th_all_asset_cap_top5_objective_sweep_thb.csv`
- best by case:
  - `result/us_th_all_asset_cap_top5_objective_best_by_case_thb.csv`
- curves:
  - `result/us_th_all_asset_cap_top5_objective_curves_thb.csv`

### Last Weight Note

- no dedicated latest-weight file was written for the top-5 objective sweep
- use the base cap-sweep latest-weight file from section 6 for the same cap case
