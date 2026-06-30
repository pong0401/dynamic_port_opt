# PIT Reselect By Step - Port Opt Advance Handoff

## Overall Best Sharpe

- Step: `2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63`
- Strategy: `Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure`
- Sharpe: `1.4003`
- CAGR: `0.2942`
- Max Drawdown: `-0.2328`
- Precompute port growth file path: `result\mean_covariance_stock_cap_sweep_daily_exposure_curves.csv`
- Precompute period: `2018-01-02 to 2026-04-29`

## US/TH/JP Index-Signal Cash-Inactive Candidate

Purpose: add the new US/Thailand/Japan sleeve allocation candidate for the next port-opt continuation pass. This is a short-history tactical candidate because Japan PIT data currently only overlaps from `2024-07-01` to `2026-01-30`.

Recommended strategy family:

- Base strategy: `Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash`
- Daily candidate: `Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash + daily exposure all assets + gold drawdown 252d warn10 crash20`
- Weekly candidate: `Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash + weekly exposure all assets + gold drawdown 252d warn10 crash20`
- Script: `scripts\run_us_th_jp_allocation_models.py`
- Summary file: `result\us_th_jp_allocation_models_summary_thb.csv`
- Curve file: `result\us_th_jp_allocation_models_curves_thb.csv`
- Weight history file: `result\us_th_jp_allocation_models_weight_history_thb.csv`
- Exposure history file: `result\us_th_jp_allocation_models_exposure_variant_history.csv`
- Japan internal weight history file: `result\us_th_jp_allocation_models_jp_internal_weight_history.csv`

Core allocation:

- Equity budget: `60%`
- Gold budget: `30%`
- BTC budget: `10%`
- Currency view: THB
- Equity sleeves:
  - US Equity: existing US PIT optimized sleeve return from `result\us_th_tactical_perf_momentum_comparison_curves_thb.csv`
  - TH Equity: existing Thailand PIT optimized sleeve return from `result\us_th_tactical_perf_momentum_comparison_curves_thb.csv`
  - JP Equity: Japan PIT equal-selected sleeve from `data\cache\dynamic_factor_copula\japan_daily_bars.parquet` and `data\cache\dynamic_factor_copula\japan_pit_universe_history.parquet`
- Japan signal ticker: `^N225`; fallback: Japan PIT equal-weight proxy when Nikkei is unavailable
- Japan FX: `JPYTHB=X`; fallback: JPY local return if FX cannot be loaded

Index-signal cash-inactive equity rule:

- US signal: SPY trend and momentum
  - Trend: price above MA300
  - Momentum: 63-day return above 0
  - Score: `(trend + momentum) / 2`, shifted by 1 trading session
- TH signal: SET trend and momentum
  - Trend: `^SET.BK` above MA200
  - Momentum: 63-day return above 0
  - Score: `(trend + momentum) / 2`, shifted by 1 trading session
- JP signal: Nikkei trend and momentum
  - Trend: `^N225` above MA120
  - Momentum: 63-day return above 0
  - Score: `(trend + momentum) / 2`, shifted by 1 trading session
- Equity budget assignment:
  - With JP: each country receives `equity_budget / 3 * country_score`
  - No JP comparison: US/TH each receive `equity_budget / 2 * country_score`
  - Inactive equity is not redistributed; it goes to `Cash / Reduced Exposure`
- This cash-inactive rule beat the tested stock-bucket cutout rule, because the cutout rule redistributed stock risk into remaining active countries and raised volatility/drawdown.

Asset exposure variants:

- Daily exposure applies the exposure signal every trading day.
- Weekly exposure samples the already-lagged exposure on weekly Friday cadence and forward-fills it, reducing whipsaw/turnover risk.
- US Equity exposure: SPY MA300 below-trend exposure `50%`
- TH Equity exposure: SET MA200 below-trend exposure `0%`
- JP Equity exposure: Nikkei MA120 below-trend exposure `0%`
- BTC exposure: BTC MA50 below-trend exposure `0%`
- Gold uses drawdown exposure, not MA80 trend exposure, for the recommended candidate:
  - Rolling high window: `252` trading days
  - Warn drawdown: `-10%`
  - Crash drawdown: `-20%`
  - Warn exposure: `75%`
  - Crash exposure: `50%`
  - Recovery threshold: drawdown improves to `-5%`
  - Panic fail-safe: if drawdown <= `-30%`, Gold < MA200, and Gold 63-day momentum < 0, exposure can go to `0%`
  - Gold signal is shifted by 1 trading session before use

Same-window comparison versus PIT handoff best:

- Comparison period: `2024-07-01 to 2026-01-30`
- PIT handoff overall best on this same window:
  - Strategy: `Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure`
  - Sharpe: `1.6448`
  - CAGR: `42.03%`
  - Annual Vol: `20.91%`
  - Max Drawdown: `-18.45%`
- Best PIT-linked same-window variant found in the referenced PIT curve files:
  - Strategy: `Mean Covariance Gold30 stock-cap sweep stockcap6 mom_63 + asset-level daily exposure`
  - Sharpe: `1.7583`
  - CAGR: `41.33%`
  - Annual Vol: `19.06%`
  - Max Drawdown: `-15.50%`
- New daily candidate:
  - Sharpe: `1.9501`
  - CAGR: `25.35%`
  - Annual Vol: `10.36%`
  - Max Drawdown: `-6.12%`
- New weekly candidate:
  - Sharpe: `1.9341`
  - CAGR: `25.00%`
  - Annual Vol: `10.29%`
  - Max Drawdown: `-6.01%`

Selected JP-optimized continuation candidate:

- Strategy: `Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash + weekly exposure all assets + gold drawdown 252d warn10 crash20`
- JP sleeve mode: `JP optimized min_vol_mom_tilt top10 cap15%`
- JP optimizer script: `scripts\run_us_th_jp_optimized_sleeve_sweep.py`
- JP optimizer output files:
  - Summary: `result\us_th_jp_optimized_sleeve_sweep_focus_summary_thb.csv`
  - Curves: `result\us_th_jp_optimized_sleeve_sweep_curves_thb.csv`
  - Top-level weight history: `result\us_th_jp_optimized_sleeve_sweep_weight_history_thb.csv`
  - Latest top-level weights: `result\us_th_jp_optimized_sleeve_sweep_latest_weights_thb.csv`
  - JP internal weights: `result\us_th_jp_optimized_sleeve_sweep_jp_internal_weight_history.csv`
- JP optimized sleeve config:
  - JP PIT selected assets: `10`
  - JP internal max weight: `15%`
  - JP objective: `min_vol_mom_tilt`
  - JP covariance lookback: `120` trading days
  - Minimum training history: `40` trading days
  - Momentum signal: trailing return up to `63` trading days
  - Concentration penalty: `0.01`
  - If the JP optimizer does not have enough usable history, fallback is equal weight for that rebalance only
  - JP internal weights are applied with a 1-session lag to avoid same-close lookahead
- Same-window metrics, `2024-07-01 to 2026-01-30`:
  - Sharpe: `2.1295`
  - CAGR: `30.03%`
  - Annual Vol: `11.26%`
  - Max Drawdown: `-6.92%`
- Concentration:
  - Latest JP sleeve weight in the selected top-level strategy: `20%`
  - Latest JP internal max weight: `15%`
  - Latest max JP stock effective weight: `3%`
- Why this is selected:
  - It is the best `top10 cap15%` weekly candidate by Sharpe.
  - It keeps JP single-stock concentration lower than the more aggressive `top10 cap20%` candidate, whose latest max JP stock effective weight is `4%`.
  - It materially improves the equal-weight JP top10 weekly candidate while keeping drawdown near the same range.
- More aggressive but less conservative reference:
  - `JP optimized mean_variance top10 cap20% + weekly exposure + gold DD252`
  - Sharpe: `2.2509`
  - CAGR: `32.68%`
  - Max Drawdown: `-6.97%`
  - Latest max JP stock effective weight: `4%`

Interpretation for continuation:

- Use the daily candidate when selecting strictly by same-window Sharpe.
- Prefer the weekly candidate for live-style follow-up if turnover/whipsaw control matters; Sharpe is slightly lower, but max drawdown is slightly better.
- For the JP-optimized path, use `JP optimized min_vol_mom_tilt top10 cap15% + weekly exposure + gold DD252` as the conservative selected candidate for the next continuation pass.
- The new strategy is more retirement-friendly than the PIT mean-covariance best on this short overlap because it cuts volatility and drawdown materially.
- The PIT mean-covariance strategies still have higher CAGR on the same window, but with much higher volatility and drawdown.
- Do not treat the Japan result as full-period proof until Japan PIT history is extended beyond the current free-data overlap.

## US/TH Tactical Final Best Sharpe

- Notebook: `notebook\us_th_tactical_perf_momentum.ipynb`
- Strategy: `Tactical TH/Gold/BTC 65/25/10 Gold crash protection`
- Tactical TH rule: `proxy_regime relative_return binary lb1 cap30 entry0 exit0 hold0 confirm1`
- Overlay mix: `Equity/Gold/BTC 65/25/10`
- Daily exposure:
  - US Equity: `SPY MA300 below 50%`
  - TH Equity: `SET MA200 below 0%`
  - Gold: `DD252 warn -8% -> 50%; crash -20% -> 50%; panic -30% and below MA200 and mom63 < 0 -> 0%; recover -5%`
  - BTC: `BTC MA50 below 0%`
- Timing-aligned period: `2018-01-02 to 2026-04-29`
- Timing-aligned Sharpe: `1.3835`
- Timing-aligned CAGR: `0.2400`
- Timing-aligned Annual Vol: `0.1492`
- Timing-aligned Max Drawdown: `-0.1906`
- Timing-aligned Total Return: `5.3316`
- Full-history period: `2006-12-29 to 2026-04-29`
- Full-history Sharpe: `0.9836`
- Full-history CAGR: `0.1728`
- Full-history Max Drawdown: `-0.1906`
- Average Gold Weight: `0.1892`
- Average Cash / Reduced Exposure Weight: `0.1665`
- Gold Zero Days: `10`
- Precompute port growth file path: `result\us_th_tactical_perf_momentum_gold_crash_protection_sweep_curves_thb.csv`
- Summary file: `result\us_th_tactical_perf_momentum_gold_crash_protection_sweep_thb.csv`
- Period compare file: `result\us_th_tactical_perf_momentum_gold_btc_overlay_period_compare_thb.csv`
- PIT aligned metric file: `result\us_th_tactical_perf_momentum_gold_btc_overlay_pit_aligned_metrics_thb.csv`
- Precompute period: `2006-12-29 to 2026-04-29`

Gold crash-protection candidate:

- Rule: `Gold max 25%; DD252 warn -8% -> Gold exposure 50%; panic fail-safe can set Gold to 0 when DD <= -30%, Gold < MA200, and 63D momentum < 0; recover when drawdown improves to -5%`
- Interpretation: use this when the main concern is a Gold crash. It reduces Gold exposure without fully exiting, so it cuts volatility/drawdown slightly while giving up some CAGR.
- Timing-aligned period: `2018-01-02 to 2026-04-29`
- Timing-aligned Sharpe: `1.3835`
- Timing-aligned CAGR: `0.2400`
- Timing-aligned Annual Vol: `0.1492`
- Timing-aligned Max Drawdown: `-0.1906`
- Timing-aligned Total Return: `5.3316`
- Full-history period: `2006-12-29 to 2026-04-29`
- Full-history Sharpe: `0.9836`
- Full-history CAGR: `0.1728`
- Full-history Max Drawdown: `-0.1906`
- Average Gold Weight: `0.1892`
- Average Cash / Reduced Exposure Weight: `0.1665`
- Gold Zero Days: `10`
- Summary file: `result\us_th_tactical_perf_momentum_gold_crash_protection_sweep_thb.csv`
- Curve file: `result\us_th_tactical_perf_momentum_gold_crash_protection_sweep_curves_thb.csv`
- Weight history file: `result\us_th_tactical_perf_momentum_gold_crash_protection_sweep_weight_history_thb.csv`

One-model optimizer test from this config:

- Script: `scripts\run_us_th_tactical_one_model.py`
- Strategy: `One-model US+TH signal stocks + Gold/BTC caps`
- Rule: US PIT top 30 stocks always eligible; TH PIT top 30 stocks eligible only when tactical TH signal is on; Gold/BTC always eligible.
- Caps: stock `8%`, Gold `30%`, BTC `10%`
- Objective: `mean_variance` with `mom_63`
- Full-history period: `2006-12-29 to 2026-04-29`
- Raw one-model Sharpe: `0.7491`
- Raw one-model CAGR: `0.1908`
- Raw one-model Max Drawdown: `-0.3891`
- One-model with asset-level daily exposure Sharpe: `0.8323`
- One-model with asset-level daily exposure CAGR: `0.1830`
- One-model with asset-level daily exposure Max Drawdown: `-0.2793`
- Interpretation: this does not beat the fixed/tactical Gold25 crash-protection result; it is included as a requested one-model experiment, not as the current final best.
- Summary file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_summary_thb.csv`
- Curve file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_curves_thb.csv`
- Latest weights file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_latest_weights_thb.csv`
- Period compare file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_period_compare_thb.csv`

One-model US/TH group cap sweep:

- Script: `scripts\run_us_th_tactical_one_model_group_caps.py`
- Tested group caps: no cap, `40%`, `50%`, `60%` applied to total US stock weight and total TH stock weight separately.
- Best in this sweep: `One-model No group cap + daily exposure`
  - CAGR: `0.1830`
  - Sharpe: `0.8323`
  - Max Drawdown: `-0.2793`
- Best capped variant: `One-model US/TH group cap 60% + daily exposure`
  - CAGR: `0.1689`
  - Sharpe: `0.8198`
  - Max Drawdown: `-0.3154`
- Lower group caps reduce concentration and drawdown in some cases, but they also force more cash and do not improve Sharpe versus no group cap in this run.
- Summary file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_summary_thb.csv`
- Curves file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_curves_thb.csv`
- Latest weights file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_latest_weights_thb.csv`
- Grouped weight history file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_grouped_weight_history_thb.csv`

One-model asymmetric US/TH group cap sweep:

- Script: `scripts\run_us_th_tactical_one_model_asym_group_caps.py`
- Tested: US group cap `70%`, `80%`, `90%`; TH group cap fixed at `50%`.
- Best in this sweep: `One-model US cap 70% / TH cap 50% + daily exposure`
  - CAGR: `0.1865`
  - Sharpe: `0.9225`
  - Max Drawdown: `-0.2321`
  - Average US stock weight: `0.6852`
  - Average TH stock weight: `0.1670`
  - Average Gold weight: `0.1276`
  - Average BTC weight: `0.0203`
  - Average cash/reduced exposure: `0.1571`
- This is the best one-model variant so far, but it still does not beat the fixed/tactical Gold25 crash-protection final strategy.
- Summary file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_summary_thb.csv`
- Curves file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_curves_thb.csv`
- Latest weights file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_latest_weights_thb.csv`
- Grouped weight history file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_grouped_weight_history_thb.csv`

One-model asymmetric grid US 70/80 and TH 30/40:

- Script: `scripts\run_us_th_tactical_one_model_asym_group_caps.py`
- Tested: US group cap `70%`, `80%`; TH group cap `30%`, `40%`.
- Best in this grid: `One-model US cap 70% / TH cap 30% + daily exposure`
  - CAGR: `0.1919`
  - Sharpe: `0.9447`
  - Max Drawdown: `-0.2163`
  - Average US stock weight: `0.7393`
  - Average TH stock weight: `0.1043`
  - Average Gold weight: `0.1361`
  - Average BTC weight: `0.0203`
  - Average cash/reduced exposure: `0.1477`
- This is the best one-model variant so far, but still below the fixed/tactical Gold25 crash-protection final strategy on Sharpe.
- Summary file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv`
- Curves file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv`
- Latest weights file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_latest_weights_thb.csv`
- Grouped weight history file: `result\us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv`

One-model US70/TH30 single-stock concentration sweep:

- Script: `scripts\run_us_th_one_model_us70_th30_concentration_sweep.py`
- Purpose: reduce single-stock concentration for the one-model `US cap 70% / TH cap 30% + daily exposure` path without changing the country group caps or proxy daily-exposure rule.
- Baseline in this sweep: `stockcap8 penalty0.02 assets30 + daily exposure`
- Tested concentration levers:
  - stock cap: `8%`, `7%`, `6%`, `5%`
  - concentration penalty: `0.02`, `0.05`, `0.10`, `0.20`
  - selected assets per side: `30`, `40`, `50`
  - combined candidates including `stockcap5 penalty0.02 assets50` and `stockcap6 penalty0.05 assets40`
- Selected low-concentration config to keep on hand:
  - Strategy: `One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure`
  - Stock cap: `5%`
  - Concentration penalty: `0.02`
  - US selected assets: `50`
  - TH selected assets when TH signal is on: `50`
  - US group cap: `70%`
  - TH group cap: `30%`
  - Gold cap: `30%`
  - BTC cap: `10%`
  - Daily exposure: current proxy exposure; US uses SPY, TH uses SET, Gold/BTC use existing rules
- Full-period metrics, `2007-12-31` to `2026-04-29`:
  - CAGR: `19.77%`
  - Annual Vol: `16.05%`
  - Sharpe: `1.0179`
  - Max Drawdown: `-18.01%`
- Concentration metrics:
  - Latest max single-stock weight: `5.00%`
  - Latest top-5 stock weight: `25.00%`
  - Latest top-10 stock weight: `50.00%`
  - Latest effective stock count: `23.33`
- Average effective weights:
  - US stocks: `60.13%`
  - TH stocks: `10.23%`
  - Gold: `12.90%`
  - BTC: `2.16%`
  - Cash / Reduced Exposure: `14.58%`
- Latest effective weights on `2026-04-29`:
  - US stocks: `60.00%`
  - TH stocks: `30.00%`
  - Gold: `5.00%`
  - BTC: `0.00%`
  - Cash / Reduced Exposure: `5.00%`
- Latest holdings on `2026-04-29`:

| Asset | Effective Weight |
|---|---:|
| GC=F | 5.00% |
| Cash / Reduced Exposure | 5.00% |
| LRCX | 5.00% |
| WMT | 5.00% |
| CVX | 5.00% |
| GEV | 5.00% |
| JNJ | 5.00% |
| INTC | 5.00% |
| AMAT | 5.00% |
| MU | 5.00% |
| TXN | 5.00% |
| COST | 5.00% |
| XOM | 5.00% |
| MRK | 5.00% |
| PTTEP.BK | 4.34% |
| BCP.BK | 4.32% |
| TOP.BK | 4.30% |
| HANA.BK | 4.27% |
| DELTA.BK | 4.27% |
| IVL.BK | 4.27% |
| PTTGC.BK | 4.24% |

- Interpretation:
  - This is the most practical conservative low-concentration candidate from the sweep.
  - It lowers latest max stock weight from `8%` to `5%`, lowers latest top-5 from `40%` to `25%`, and improves latest max drawdown versus the sweep baseline.
  - It gives up some full-period CAGR versus looser `stockcap8` variants, but keeps Sharpe slightly above the `stockcap8 penalty0.02 assets30` sweep baseline.
- Output files:
  - Summary: `result\us_th_one_model_us70_th30_concentration_sweep_summary_thb.csv`
  - Curves: `result\us_th_one_model_us70_th30_concentration_sweep_curves_thb.csv`
  - Latest weights: `result\us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv`
  - Effective weights: `result\us_th_one_model_us70_th30_concentration_sweep_effective_weights_thb.csv`
  - Concentration history: `result\us_th_one_model_us70_th30_concentration_sweep_concentration_history_thb.csv`
  - Period compare: `result\us_th_one_model_us70_th30_concentration_sweep_period_compare_thb.csv`

One-model US70/TH30 AI-tech theme-cap sweep:

- Script: `scripts\run_us_th_one_model_us70_th30_theme_cap_sweep.py`
- Purpose: add a manual strict AI-tech/theme guardrail on top of the selected `stockcap5 penalty0.02 assets50 + daily exposure` config.
- Strict AI-tech bucket:
  - `AAPL`, `AMD`, `GOOG`, `GOOGL`, `INTC`, `MU`, `NVDA`, `QCOM`, `TXN`
- Tested theme caps:
  - no theme cap baseline, `40%`, `35%`, `30%`, `25%`, `20%`
- Selected capped strategy to keep on hand:
  - Strategy: `One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 AI-tech cap 25% + daily exposure`
  - Stock cap: `5%`
  - Concentration penalty: `0.02`
  - US selected assets: `50`
  - TH selected assets when TH signal is on: `50`
  - Strict AI-tech cap: `25%`
  - US group cap: `70%`
  - TH group cap: `30%`
  - Gold cap: `30%`
  - BTC cap: `10%`
  - Daily exposure: current proxy exposure; US uses SPY, TH uses SET, Gold/BTC use existing rules
- Full-period metrics, `2007-12-31` to `2026-04-29`:
  - CAGR: `19.59%`
  - Annual Vol: `16.05%`
  - Sharpe: `1.0083`
  - Max Drawdown: `-18.01%`
  - Sharpe loss vs uncapped `stockcap5 penalty0.02 assets50` baseline: `0.0096`
- Theme exposure metrics:
  - Average strict AI-tech weight: `9.99%`
  - Max strict AI-tech weight: `25.00%`
  - Latest strict AI-tech weight on `2026-04-29`: `15.00%`
  - Latest strict AI-tech assets: `INTC 5.00%`, `MU 5.00%`, `TXN 5.00%`
- Concentration metrics:
  - Latest max single-stock weight: `5.00%`
  - Latest top-5 stock weight: `25.00%`
  - Latest top-10 stock weight: `50.00%`
  - Latest effective stock count: `23.33`
- Latest effective weights on `2026-04-29`:
  - US stocks: `60.00%`
  - TH stocks: `30.00%`
  - Gold: `5.00%`
  - BTC: `0.00%`
  - Cash / Reduced Exposure: `5.00%`
- Interpretation:
  - `AI-tech cap 30%` is the lowest no-cost guardrail in this repo's long-period backtest because the strict AI-tech bucket never exceeded `30%`.
  - `AI-tech cap 25%` is the more defensive capped strategy to keep on hand: it enforces a lower theme ceiling with only a small full-period Sharpe loss.
  - `AI-tech cap 20%` looks over-constrained in this test, with Sharpe falling to `0.9781` and Max Drawdown slightly worse at `-18.20%`.
  - This run evaluates this repo's data through `2026-04-29`; the separate `port_opt_advance` latest snapshot on `2026-06-18` showed strict AI-tech effective weight of about `40%`, so the latest-refresh path still needs the same theme cap if that live allocation is used.
- Output files:
  - Summary: `result\us_th_one_model_us70_th30_theme_cap_sweep_summary_thb.csv`
  - Curves: `result\us_th_one_model_us70_th30_theme_cap_sweep_curves_thb.csv`
  - Latest weights: `result\us_th_one_model_us70_th30_theme_cap_sweep_latest_weights_thb.csv`
  - Effective weights: `result\us_th_one_model_us70_th30_theme_cap_sweep_effective_weights_thb.csv`
  - Concentration history: `result\us_th_one_model_us70_th30_theme_cap_sweep_concentration_history_thb.csv`
  - Period compare: `result\us_th_one_model_us70_th30_theme_cap_sweep_period_compare_thb.csv`

One-model US70/TH30 combined segment-cap handoff:

- Script: `scripts\run_us_th_one_model_segment_cap_backtest.py`
- Purpose: replace the old hard-coded AI-tech bucket with local segment files and test segment caps on the selected `stockcap5 penalty0.02 assets50 + daily exposure` one-model US/TH strategy.
- Core strategy settings:
  - Strategy family: one combined mean/covariance optimizer across active PIT US stocks, active PIT Thailand stocks when TH tactical signal is on, Gold, and BTC.
  - Start date: `2016-01-01`; metric window from output: `2017-05-31` to `2026-06-26`.
  - US universe: active PIT S&P 500 members, liquidity-selected top `50` at each rebalance.
  - TH universe: active PIT SET100 members, liquidity-selected top `50` only when the TH tactical signal is on.
  - TH tactical rule: monthly SET-vs-SPY(THB) relative-return binary signal, shifted one month before use.
  - Optimizer signal: `mom_63`; concentration penalty: `0.02`; risk aversion: `8.0`.
  - Stock cap: `5%`; US group cap: `70%`; TH group cap: `30%`; Gold cap: `30%`; BTC cap: `10%`.
  - Daily exposure overlay: US uses SPY MA300 floor `50%`; TH uses SET MA200 floor `0%`; Gold uses existing crash-protection exposure; BTC uses MA50 floor `0%`; reduced exposure goes to cash.
- Segment source files:
  - US: `data\us_segment.csv`
    - Local CSV supplied for this repo run.
    - Required columns: `ticker`, `segment`; optional detail column currently present: `gics_sub_industry`.
    - Current file shape during this handoff: `503` rows.
    - Tickers are loaded uppercase exactly as US symbols, e.g. `AAPL`.
  - Thailand: `data\set100_segment.xls`
    - Local HTML-style `.xls` table supplied for this repo run.
    - Header in file: `List of Listed Companies & Contact Information`, `As of 22 Jun 2026`.
    - Loader reads the table header row where first column is `Symbol` and uses `Industry` as the raw TH segment. It appends `.BK` to symbols, e.g. `ADVANC` -> `ADVANC.BK`.
    - Raw TH columns of interest: `Symbol`, `Industry`, `Sector`; this strategy uses `Industry` for segment cap.
  - Segment classification is currently a static/latest local classification file, not historical PIT sector membership. The stock universe itself remains PIT, but segment labels are not time-varying in this run.
- Combined segment cap logic:
  - The final desired cap mode is `all_us_plus_th_segments`.
  - Segment caps are applied after normalizing TH industries into US-style segment buckets, then summing US + TH weights in the same bucket before applying the `25%` cap.
  - There is no separate `US::` / `TH::` cap bucket in the final combined mode.
  - Missing tickers in either segment file are left uncapped at the segment level, but still subject to stock, country-group, Gold, and BTC caps.
- TH-to-US segment normalization used before the combined cap:
  - `Technology` -> `Information Technology`
  - `Resources` -> `Energy`
  - `Property & Construction` -> `Real Estate`
  - `Industrial` / `Industrials` -> `Industrials`
  - `Financials` -> `Financials`
  - `Agro & Food Industry` -> `Consumer Staples`
  - `Consumer Products` -> `Consumer Discretionary`
  - `Services` -> `Consumer Discretionary`
- Tested variants in the latest run:
  - `it_only`: cap only US `Information Technology` at `25%`.
  - `all_segments`: cap all US segments at `25%`, without including TH in the cap bucket.
  - `it_plus_th_segments`: cap combined US `Information Technology` + TH `Technology` at `25%`; other combined segments can exceed `25%`.
  - `all_us_plus_th_segments`: cap every normalized combined US+TH segment at `25%`. This is the recommended combined segment-cap variant.
- Latest THB metrics from `result\us_th_one_model_us70_th30_segment_cap25_backtest_summary_thb.csv`:

| Variant | Cap mode | CAGR | Sharpe | Max Drawdown | Avg US Wt | Avg TH Wt | Avg Cash Wt | Max combined segment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `all_segments` | US-only all segment cap | `14.43%` | `0.8891` | `-20.11%` | `59.43%` | `29.64%` | `27.98%` | `34.88%` |
| `it_only` | US-only IT cap | `14.40%` | `0.8869` | `-19.93%` | `59.45%` | `29.64%` | `27.98%` | `34.88%` |
| `all_us_plus_th_segments` | combined US+TH all segment cap | `14.33%` | `0.8840` | `-20.05%` | `59.31%` | `29.76%` | `27.98%` | `25.00%` |
| `it_plus_th_segments` | combined IT/Technology cap only | `14.33%` | `0.8825` | `-19.93%` | `59.42%` | `29.76%` | `27.98%` | `36.44%` |

- Interpretation:
  - `all_us_plus_th_segments` is the clean handoff candidate if the portfolio should enforce a true cross-market segment cap. Its max normalized combined segment weight is exactly `25.00%`.
  - The Sharpe cost versus `all_segments` is small in this run: `0.8891` -> `0.8840`.
  - `it_plus_th_segments` only caps the combined IT/Technology bucket. It does not cap other combined buckets, so `Consumer Discretionary` reached `36.44%` on `2018-03-31`.
  - In the US-only cap variants, combined US+TH `Information Technology` reached `34.88%` on `2026-03-31`; this is why the combined cap matters.
- Validation checks performed after the run:
  - Raw optimizer weights summed to `1.0` for every rebalance and every variant.
  - Effective daily weights summed to `1.0` for every day and every variant.
  - `all_us_plus_th_segments` max combined segment weight was `25.00%`.
  - `it_plus_th_segments` max `Information Technology` bucket weight was `25.00%`.
- Output files:
  - Summary: `result\us_th_one_model_us70_th30_segment_cap25_backtest_summary_thb.csv`
  - Curves: `result\us_th_one_model_us70_th30_segment_cap25_backtest_curves_thb.csv`
  - Raw optimizer weights: `result\us_th_one_model_us70_th30_segment_cap25_backtest_raw_weight_history_thb.csv`
  - Effective daily weights: `result\us_th_one_model_us70_th30_segment_cap25_backtest_effective_weight_history_thb.csv`
  - Latest effective weights: `result\us_th_one_model_us70_th30_segment_cap25_backtest_latest_effective_weights_thb.csv`
  - Universe history: `result\us_th_one_model_us70_th30_segment_cap25_backtest_universe_history_thb.csv`
  - Daily exposure history: `result\us_th_one_model_us70_th30_segment_cap25_backtest_exposure_history_thb.csv`
  - Combined segment weight history: `result\us_th_one_model_us70_th30_segment_cap25_backtest_segment_weight_history_thb.csv`
- Rerun command:
  - `$env:PYTHONIOENCODING='utf-8'; $env:PYTHONWARNINGS='ignore'; python -B scripts\run_us_th_one_model_segment_cap_backtest.py`

Latest recheck note:

- `scripts\recheck_us_th_tactical_gold_btc_latest_weights.py` has been updated to recompute the selected `65/25/10 Gold crash protection` latest weights from fresh yfinance data.
- Latest full universe rerun on `2026-06-03` was blocked by yfinance rate limit, so `scripts\derive_gold25_latest_weights_from_prior_fresh.py` refreshed only overlay tickers from yfinance and rescaled the latest successful fresh US/TH PIT internal weights to the selected `65/25/10` mix.

- Date: `2026-06-03`
- Source file: `result\us_th_tactical_perf_momentum_final_best_latest_effective_security_weights_thb.csv`
- Sleeve file: `result\us_th_tactical_perf_momentum_final_best_latest_effective_sleeve_weights_thb.csv`
- Meta file: `result\us_th_tactical_perf_momentum_final_best_latest_meta.csv`
- Timing note: overlay tickers refreshed from yfinance; US/TH internal weights reused from the last successful fresh PIT rerun and rescaled to `65/25/10`.
- TH tactical weight inside equity sleeve: `0.3000`
- BTC latest price: `66886.4688`
- BTC MA50: `76941.0578`
- BTC daily exposure: `0.0000`
- Gold latest price: `4484.7998`
- Gold DD252: `-0.1567`
- Gold daily exposure: `0.5000`
- US sleeve internal weight date: `2026-05-31`
- TH sleeve internal weight date: `2026-05-31`

Latest effective sleeve weights:

| Sleeve | Effective Weight |
|---|---:|
| US Equity | 0.4550 |
| TH Equity | 0.1950 |
| Gold | 0.1250 |
| BTC | 0.0000 |
| Cash / Reduced Exposure | 0.2250 |

Latest effective security weights:

| Asset | Sleeve | Effective Weight | Internal Weight | Raw Sleeve Weight | Daily Exposure |
|---|---|---:|---:|---:|---:|
| Cash / Reduced Exposure | Cash / Reduced Exposure | 0.2250 | 1.0000 | 0.2250 | 1.0000 |
| GC=F | Gold | 0.1250 | 1.0000 | 0.2500 | 0.5000 |
| GS | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| MSFT | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| GOOG | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| AMZN | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| UNH | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| HOOD | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| INTC | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| MU | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| AVGO | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| APP | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| ORCL | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| AMD | US Equity | 0.0364 | 0.0800 | 0.4550 | 1.0000 |
| LLY | US Equity | 0.0182 | 0.0400 | 0.4550 | 1.0000 |
| KTB.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| KTC.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| DELTA.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| KBANK.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| CPALL.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| BBL.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| CRC.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| MINT.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| BH.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| GPSC.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| HANA.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| SCC.BK | TH Equity | 0.0156 | 0.0800 | 0.1950 | 1.0000 |
| TISCO.BK | TH Equity | 0.0056 | 0.0286 | 0.1950 | 1.0000 |
| AOT.BK | TH Equity | 0.0022 | 0.0114 | 0.1950 | 1.0000 |
| BTC-USD | BTC | 0.0000 | 1.0000 | 0.1000 | 0.0000 |

## Latest Recommended Effective Weights

- Strategy: `Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure`
- Date: `2026-05-29`
- Source file: `result\mean_covariance_gold30_asset_daily_latest_effective_weights.csv`
- Sleeve history file: `result\mean_covariance_gold30_asset_daily_sleeve_weight_history.csv`

| Asset | Sleeve | Effective Weight |
|---|---|---:|
| GOOGL | US Equity | 0.0800 |
| APP | US Equity | 0.0800 |
| UNH | US Equity | 0.0800 |
| AMZN | US Equity | 0.0800 |
| HOOD | US Equity | 0.0800 |
| ORCL | US Equity | 0.0800 |
| AMD | US Equity | 0.0800 |
| AVGO | US Equity | 0.0800 |
| INTC | US Equity | 0.0800 |
| SMCI | US Equity | 0.0800 |
| MU | US Equity | 0.0800 |
| GOOG | US Equity | 0.0610 |
| AAPL | US Equity | 0.0590 |

## Latest-Year US+TH 1Y Lookback Test

- Step: `2.3c-2 US+TH Mean Covariance Gold30, 1Y Lookback, Latest-Year Rebalance`
- Strategy: `US+TH Mean Covariance Gold30 stock cap 8 mom_63 1Y lookback latest-year rebalance + asset-level daily exposure`
- Note: this is a latest-year-only test, so compare directionally against the full-period final strategy.
- Lookback: `252` trading days
- Backtest period: `2025-06-02 to 2026-04-29` for daily exposure; raw rebalance curve starts `2025-05-30`
- Rebalance count: `11`
- US assets: `30`
- TH assets: `30`
- Stock cap: `8%` for both US and Thailand stocks
- Gold/BTC/BIL caps: `30%/5%/0%`
- Daily exposure: US uses SPY trend, TH uses SET trend, Gold/BTC use own trend
- Source files:
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_summary.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_latest_effective_weights.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_sleeve_weight_history.csv`

| Variant | CAGR | Annual Vol | Sharpe | Max Drawdown | Total Return |
|---|---:|---:|---:|---:|---:|
| Raw monthly rebalance | 52.99% | 21.53% | 2.0291 | -9.97% | 47.52% |
| Asset-level daily exposure | 44.62% | 20.68% | 1.8307 | -10.32% | 39.71% |

Latest daily effective weights on `2026-04-29`:

| Asset | Sleeve | Effective Weight |
|---|---|---:|
| ADVANC.BK | TH Equity | 0.0800 |
| DELTA.BK | TH Equity | 0.0800 |
| TRUE.BK | TH Equity | 0.0800 |
| GULF.BK | TH Equity | 0.0800 |
| GEV | US Equity | 0.0800 |
| PTTGC.BK | TH Equity | 0.0800 |
| IVL.BK | TH Equity | 0.0800 |
| HANA.BK | TH Equity | 0.0800 |
| AMAT | US Equity | 0.0800 |
| PTTEP.BK | TH Equity | 0.0800 |
| TOP.BK | TH Equity | 0.0800 |
| XOM | US Equity | 0.0800 |
| INTC | US Equity | 0.0400 |

## Latest-Year US+TH Side-Switch Test

- Step: `2.3c-3 US+TH Side-Switch Daily Exposure`
- Note: this is also a latest-year-only test, so compare directionally against full-period results.
- Base model: `US+TH Mean Covariance Gold30 stock cap 8 mom_63 1Y lookback latest-year rebalance`
- TH daily exposure trigger: `^SET.BK`
- Best TH trigger param from grid: `MA100`, below-trend exposure `0%`
- US trigger remains the existing SPY daily-exposure config from `best_param_step3b_best_signal_config_used.csv`
- Rule:
  - US off and TH on: move US stock weight into active TH stocks pro-rata
  - TH off and US on: move TH stock weight into active US stocks pro-rata
  - both on: keep model weights
  - both off: keep stock-side cash drag
- Source files:
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_th_set_param_sweep.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_comparison.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_side_switch_latest_weights.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_side_switch_sleeve_weight_history.csv`

| Variant | CAGR | Annual Vol | Sharpe | Max Drawdown | Total Return | Average Exposure |
|---|---:|---:|---:|---:|---:|---:|
| TH SET optimized cash drag | 54.75% | 20.91% | 2.1267 | -9.97% | 48.55% | 91.49% |
| US/TH side-switch | 64.57% | 22.80% | 2.2300 | -9.97% | 57.06% | 99.09% |

Side-switch trigger count:

| State | Days |
|---|---:|
| US off, TH on | 0 |
| TH off, US on | 61 |
| Both on | 173 |
| Both off | 3 |

Latest side-switch effective weights on `2026-04-29`:

| Asset | Sleeve | Effective Weight |
|---|---|---:|
| ADVANC.BK | TH Equity | 0.0800 |
| DELTA.BK | TH Equity | 0.0800 |
| TRUE.BK | TH Equity | 0.0800 |
| GULF.BK | TH Equity | 0.0800 |
| GEV | US Equity | 0.0800 |
| PTTGC.BK | TH Equity | 0.0800 |
| IVL.BK | TH Equity | 0.0800 |
| HANA.BK | TH Equity | 0.0800 |
| AMAT | US Equity | 0.0800 |
| PTTEP.BK | TH Equity | 0.0800 |
| TOP.BK | TH Equity | 0.0800 |
| XOM | US Equity | 0.0800 |
| INTC | US Equity | 0.0400 |

## Full-Period US+TH Side-Switch Timing-Aligned Test

- Step: `2.3c-4 US+TH Side-Switch, Same Timing As Final Strategy`
- Purpose: rerun the latest-year side-switch idea on the same evaluation window as the current final strategy.
- Evaluation period: `2018-01-02 to 2026-04-29`
- Lookback: `252` trading days
- Base model: US+TH mean covariance with PIT reselection, US top 30 + TH top 30, stock cap `8%`, Gold cap `30%`, BTC cap `5%`, BIL cap `0%`
- TH daily exposure trigger is re-optimized on the full period using `^SET.BK`
- Best full-period TH trigger param: `MA200`, below-trend exposure `0%`
- Source files:
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_th_set_param_sweep.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_comparison.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_side_switch_latest_weights.csv`
  - `result\mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_side_switch_sleeve_weight_history.csv`

| Variant | CAGR | Annual Vol | Sharpe | Max Drawdown | Total Return | Average Exposure |
|---|---:|---:|---:|---:|---:|---:|
| TH SET optimized cash drag | 17.99% | 17.35% | 1.0123 | -25.47% | 296.08% | 78.85% |
| US/TH side-switch | 14.54% | 20.77% | 0.7385 | -31.07% | 209.39% | 91.05% |

Interpretation:

- Latest-year side-switch looked strong, but it does not generalize over `2018-2026`.
- Full-period side-switch increases average exposure but worsens Sharpe and drawdown.
- Do not promote side-switch to the final full-period strategy without another constraint/filter.

## Full-Period Gated Thailand Sleeve Test

- Step: `2.3e Gated Thailand Sleeve`
- Purpose: test whether Thailand should enter as a conditional sleeve instead of joining the US optimizer full time.
- Core: final no-TH `Mean Covariance Gold30 stockcap8 mom_63 + asset-level daily exposure`
- TH sleeve: SET100 PIT top-30 Static Copula sleeve with momentum
- Currency: THB
- Evaluation period: `2018-01-03 to 2026-04-29`
- Gate sweep:
  - absolute SET trend
  - SET trend + SET/SPY_THB relative trend
  - SET/SPY_THB relative trend + relative momentum
  - all conditions
  - SET moving-average slope
  - SET moving-average slope + SET/SPY_THB ratio moving-average slope
  - slope variants with price or relative-momentum confirmation
- TH sleeve weight when gate is on: `5%`, `10%`, `15%`, `20%`
- Source files:
  - `result\mean_covariance_th_gated_sleeve_summary_thb.csv`
  - `result\mean_covariance_th_gated_sleeve_curves_thb.csv`
  - `result\mean_covariance_th_gated_sleeve_annual_returns_thb.csv`
  - `result\mean_covariance_th_gated_sleeve_best_weight_history_thb.csv`
  - `result\mean_covariance_th_gated_sleeve_best_gate_history_thb.csv`

Best gated config:

- Strategy: `TH gated sleeve ma_slope_relative TH20 SET_MA75 ratio_MA200 slope63`
- Gate mode: SET MA slope plus SET/SPY_THB ratio MA slope
- TH weight when on: `20%`
- SET MA period: `75`
- SET/SPY_THB ratio MA period: `200`
- Slope lookback: `63` trading days
- Average TH weight: `1.96%`
- TH on days: `212`

| Strategy | CAGR | Annual Vol | Sharpe | Max Drawdown | Total Return |
|---|---:|---:|---:|---:|---:|
| Core no-TH final | 28.36% | 20.37% | 1.1816 | -22.20% | 750.94% |
| Best gated TH sleeve | 28.62% | 20.19% | 1.2006 | -22.20% | 765.97% |

Annual return comparison:

| Year | Core no-TH final | Best gated TH sleeve |
|---|---:|---:|
| 2018 | 5.09% | 5.09% |
| 2019 | 26.26% | 25.49% |
| 2020 | 44.80% | 44.80% |
| 2021 | 60.12% | 60.12% |
| 2022 | -11.31% | -9.71% |
| 2023 | 33.84% | 33.87% |
| 2024 | 42.88% | 42.88% |
| 2025 | 34.00% | 34.00% |
| 2026 | 19.54% | 20.03% |

Interpretation:

- MA-slope relative gate improves Sharpe, CAGR, and total return slightly versus the THB core view.
- Best average TH allocation is still only `1.47%`, so Thailand works as a very selective tactical sleeve, not a large standing allocation.
- Slope-based detection is better than simple price-above-MA in this test because it avoids many noisy SET rebounds.

## Thailand MA200 Visual Regime Gate

- Step: `2.3f Thailand MA200 Regime Gate`
- Purpose: match the visual MA200 regime blocks in the SET chart.
- Source files:
  - `result\mean_covariance_th_ma200_regime_gate_summary_thb.csv`
  - `result\mean_covariance_th_ma200_regime_gate_curves_thb.csv`
  - `result\mean_covariance_th_ma200_regime_gate_annual_returns_thb.csv`
  - `result\mean_covariance_th_ma200_regime_gate_best_gate_history_thb.csv`

Best performance in the MA200-only regime sweep:

- Strategy: `TH MA200 regime TH10 slope63 buffer0% entry20 exit40`
- Regime spans: `2021-02-10 to 2022-07-27; 2024-11-22 to 2025-03-05; 2026-01-28 to 2026-04-29`
- CAGR: `27.94%`
- Sharpe: `1.1839`
- Max Drawdown: `-22.20%`

Visual two-block detector closest to the chart:

- Strategy: `TH MA200 regime TH5 slope63 buffer8% entry1 exit20`
- Rule: enter when SET is at least `8%` above MA200 and MA200 slope over `63` trading days is positive; exit after confirmed MA200/slope deterioration.
- Regime spans: `2021-01-14 to 2022-06-29; 2026-01-28 to 2026-04-29`
- CAGR: `28.13%`
- Sharpe: `1.1812`
- Max Drawdown: `-22.20%`

Breadth-filter test:

- Breadth history file: `result\set100_pit_breadth_history.csv`
- Best breadth-filtered MA200 regime strategy: `TH MA200 regime TH20 slope63 buffer0% mom63_gt50 entry20 exit20`
- Breadth rule: SET100 PIT `Positive Mom63 > 50%`
- Regime spans: `2021-02-10 to 2021-07-16; 2021-11-01 to 2022-05-26; 2024-11-22 to 2024-12-31; 2026-03-03 to 2026-04-29`
- CAGR: `28.06%`
- Sharpe: `1.1931`
- Max Drawdown: `-22.20%`

SET100 PIT breadth window averages:

| Window | Above MA200 | Above MA75 | Positive Mom63 | Median Mom63 |
|---|---:|---:|---:|---:|
| 2021-2022 visual | 63.22% | 52.90% | 54.35% | 2.22% |
| 2026 visual | 62.58% | 60.96% | 65.69% | 5.91% |
| 2024 false positive | 33.19% | 24.15% | 25.88% | -10.26% |
| Latest test window | 48.40% | 54.34% | 54.51% | 2.05% |

Interpretation:

- MA200 visual detector can match the two intended Thailand-on blocks.
- However, it is not the best performance gate; the earlier `SET_MA75 + SET/SPY_THB ratio_MA200 + slope63` gate still has better Sharpe (`1.2006`).
- Breadth confirms that the 2024 false positive was weak internally, but adding breadth directly can cut the 2021-2022 regime too early.
- Use MA200 visual regime when interpretability is more important; use slope-relative gate when selecting by backtest performance.

## Timing Audit

- Overall-best comparison window: `2018-01-02 to 2026-04-29`
- Rows marked false are not on the exact same start/end dates and should be compared directionally unless rerun on the common overlap.

| Step | Strategy | Period | Sharpe | Same Timing |
|---|---|---:|---:|---:|
| 1. Stock only | US stock only Dynamic HMM Copula [mean_variance] [with momentum] max10 PIT reselect | 2017-12-29 to 2026-04-29 | 1.1091 | False |
| 2.1 Equity + Gold/BTC/BIL allocation | Best stock sleeve + EQUITY/GOLD/BTC 55/40/5 | 2018-01-02 to 2026-04-29 | 1.2472 | True |
| 2.2 Stocks + Gold/BTC/BIL one model | Stocks+Gold+BTC+BIL one-model Static Copula [mean_variance] PIT reselect | 2017-12-29 to 2026-04-29 | 0.6746 | False |
| 2.3 Capped one model from 2.1 | Stocks+Gold+BTC+BIL one-model capped Static Copula [mean_variance] PIT reselect | 2017-12-29 to 2026-04-29 | 0.6622 | False |
| 2.3b No-TH mean covariance + asset daily exposure | Mean Covariance + Gold/BTC/BIL capped Gold 40 + asset-level daily exposure | 2018-01-02 to 2026-04-29 | 1.3875 | True |
| 2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63 | Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure | 2018-01-02 to 2026-04-29 | 1.4003 | True |
| 2.3c-2 US+TH 1Y lookback latest-year daily exposure | US+TH Mean Covariance Gold30 stock cap 8 mom_63 1Y lookback latest-year rebalance + asset-level daily exposure | 2025-06-02 to 2026-04-29 | 1.8307 | False |
| 2.3c-3 US+TH side-switch latest-year daily exposure | US+TH side-switch Mean Covariance Gold30 stock cap 8 1Y lookback THSET_MA100_below0% | 2025-06-02 to 2026-04-29 | 2.2300 | False |
| 2.3c-4 US+TH side-switch full-period timing-aligned | US+TH side-switch Mean Covariance Gold30 stock cap 8 1Y lookback full-period THSET_MA200_below0% | 2018-01-02 to 2026-04-29 | 0.7385 | True |
| 2.3e Gated TH sleeve full-period | TH gated sleeve ma_slope_relative TH20 SET_MA75 ratio_MA200 slope63 | 2018-01-03 to 2026-04-29 | 1.2006 | False |
| 2.4 Best stock assets + Gold/BTC/BIL/IEF reoptimized | Best stock assets + Gold/BTC/BIL/IEF reoptimized Dynamic HMM Copula [US stock only] [mean_variance] max10 PIT reselect | 2017-12-29 to 2026-04-29 | 1.1659 | False |
| 2.5 Daily exposure on best 2.4 | S&P trend | 2017-12-29 to 2026-04-29 | 1.2217 | False |





