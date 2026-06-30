# Best Config Latest

## Repo

`dynamic_port_opt`

## Main Notebook

- `notebook/multi_factor_copula_poc.ipynb`

## Preferred Baseline

Current preferred baseline:

- `Static HMM with momentum/Gold/BTC 60/30/10 daily exposure`

Reason:

- better Sharpe than `70/20/10` in the tested mix sweep
- `mom_63` remains the best practical momentum signal mode
- `resid_vol`, `drawdown`, and `downside_beta` are dropped from the current baseline config

## Best Confirmed Joint Config

Source:

- `result/joint_confirm_603010_504d_1m_overlay_summary_usd.csv`
- `result/joint_confirm_603010_504d_1m_overlay_summary_thb.csv`

Config:

- Universe mode: `sp500_pit`
- Universe selection: top `30` liquid names from true S&P 500 members at each rebalance
- Clusters: `4`
- Equity sleeve rebalance: `monthly`
- Lookback: `504` trading days
- Max weight: `0.08`
- Momentum features: `on`
- Momentum signal: `on`
- Momentum signal mode: `mom_63`
- Strategic overlay rebalance: `1 month`
- Dropped feature flags:
  - `resid_vol = False`
  - `drawdown = False`
  - `downside_beta = False`

USD metrics:

- Total Return: `28.7567`
- CAGR: `25.47%`
- Annual Vol: `10.97%`
- Sharpe: `1.8512`
- Sortino: `2.4701`
- Max Drawdown: `-11.36%`
- Hit Rate: `0.5056`

THB metrics:

- Total Return: `26.6477`
- CAGR: `24.85%`
- Annual Vol: `12.09%`
- Sharpe: `1.6490`
- Sortino: `2.2513`
- Max Drawdown: `-12.21%`
- Hit Rate: `0.4973`

## Legacy Embedded Notebook Baseline

Source:

- `result/overlay_comparison_summary.csv`

Metrics:

- Total Return: `27.8389`
- CAGR: `25.20%`
- Annual Vol: `11.52%`
- Sharpe: `1.7491`
- Sortino: `2.3072`
- Max Drawdown: `-9.81%`
- Hit Rate: `0.5024`

## Legacy THB Baseline Metrics

Source:

- `result/overlay_comparison_summary_thb.csv`

Metrics:

- Total Return: `25.7818`
- CAGR: `24.59%`
- Annual Vol: `12.60%`
- Sharpe: `1.5712`
- Sortino: `2.1203`
- Max Drawdown: `-11.61%`
- Hit Rate: `0.4952`

## Notebook Baseline Equity Sleeve Config

- Universe mode: `sp500_pit`
- Universe selection: top `30` liquid names from true S&P 500 members at each rebalance
- Clusters: `4`
- Equity sleeve rebalance: `monthly`
- Max weight: `0.08`
- Momentum features: `on`
- Momentum signal: `on`
- Momentum signal mode: `mom_63`
- Dropped feature flags:
  - `resid_vol = False`
  - `drawdown = False`
  - `downside_beta = False`

## Overlay Mix

- Static HMM sleeve: `60%`
- Gold: `30%`
- BTC: `10%`

## Best Sweep Observations

### Lookback Sweep

Source:

- `result/static_hmm_603010_lookback_sweep.csv`

Best Sharpe in that sweep:

- Lookback: `504` trading days
- CAGR: `25.84%`
- Sharpe: `1.7749`
- Max Drawdown: `-11.29%`

### Strategic Rebalance Sweep

Source:

- `result/static_hmm_603010_rebalance_sweep.csv`

Best Sharpe in that sweep:

- Strategic rebalance: `1 month`
- CAGR: `24.86%`
- Sharpe: `1.8275`
- Max Drawdown: `-9.80%`

## Validation Status

The old caveat is no longer the active state:

- `504` lookback and `1 month` strategic rebalance have now been validated together
- use the joint confirmed config above as the current best config

## Thailand PIT Status

Thailand SET100 point-in-time membership is supported in code and the repo can now build a local Thai cache on top of the US base cache.

Current state:

- `universe_mode="set100_pit"` exists in `src/dynamic_factor_copula.py`
- membership history comes from `port_opt_advance/data/thai_stock/set100_ticker_start_end.csv`
- default source cache `port_opt_advance/data/cache/portopt_optimizer_proof/20Y` is still US-only
- Thai prices are now added through:
  - `data/cache/dynamic_factor_copula/extra_prices.parquet`
  - `data/cache/dynamic_factor_copula/extra_volumes.parquet`
- cache build script:
  - `scripts/build_thai_set100_cache.py`

## Thailand PIT Equity Sleeve Baseline

Source:

- `result/thai_set100_pit_metrics.csv`
- `result/thai_set100_pit_universe_history.csv`
- `result/thai_set100_cache_status.csv`

Config:

- Universe mode: `set100_pit`
- Universe selection: top `30` liquid names from true SET100 members at each rebalance
- Clusters: `4`
- Equity sleeve rebalance: `monthly`
- Lookback: `504` trading days
- Max weight: `0.08`
- Benchmark ticker: `^SET.BK`
- Vol proxy ticker: none
- Momentum features: `on`
- Momentum signal: `on`
- Momentum signal mode: `mom_63`
- Dropped feature flags:
  - `resid_vol = False`
  - `drawdown = False`
  - `downside_beta = False`

Latest Thailand equity sleeve metrics:

- `Equal Weight`: CAGR `1.04%`, Sharpe `0.1459`, Max Drawdown `-49.05%`
- `Risk Parity`: CAGR `1.46%`, Sharpe `0.1684`, Max Drawdown `-45.55%`
- `Static Copula`: CAGR `2.12%`, Sharpe `0.2038`, Max Drawdown `-50.56%`
- `Dynamic HMM Copula`: CAGR `2.22%`, Sharpe `0.2087`, Max Drawdown `-50.33%`

Next Thai step:

- build a Thailand-specific overlay comparison if you want `SET benchmark / Thai HMM / Thai HMM + Gold/BTC` curves in THB

## US + Thailand Blended Experiment

Source:

- `result/us_th_gold_btc_blended_summary_thb.csv`
- `result/us_th_gold_btc_blended_summary_usd.csv`
- `result/us_th_gold_btc_blended_curves_thb.csv`
- `result/us_th_gold_btc_blended_curves_usd.csv`
- `result/latest_us_hmm_members.csv`
- `result/latest_th_hmm_members.csv`

Script:

- `scripts/run_us_th_blended.py`

Comparison period:

- `2016-01-04` to `2026-04-29`

THB results:

- `US HMM/Gold/BTC 60/30/10`: CAGR `36.06%`, Sharpe `1.9578`, Max Drawdown `-13.06%`
- `US/TH/Gold/BTC 50/10/30/10`: CAGR `32.64%`, Sharpe `1.9724`, Max Drawdown `-11.63%`
- `US/TH/Gold/BTC 45/15/30/10`: CAGR `30.94%`, Sharpe `1.9740`, Max Drawdown `-10.90%`
- `US/TH/Gold/BTC 40/20/30/10`: CAGR `29.26%`, Sharpe `1.9696`, Max Drawdown `-10.17%`
- `US/TH/Gold/BTC 30/30/30/10`: CAGR `25.93%`, Sharpe `1.9331`, Max Drawdown `-8.70%`

Interpretation:

- `US-only + Gold/BTC` still has the highest CAGR in this overlap window
- adding Thailand improves drawdown
- `15%` Thailand is the current best THB Sharpe point in this small sweep
- this is not yet replacing the main best confirmed config; it is a new blended-market experiment
