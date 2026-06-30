# Best Param By Step - Port Opt Advance Handoff

This file lists only the best-Sharpe strategy from each final notebook family. Assets with zero allocation are intentionally omitted from strategy names and active-weight configs.

## 1. S&P buy hold: S&P 500 buy and hold

- Strategy name: `S&P 500 buy and hold`
- Winner type: `Benchmark Sharpe`
- Config: SPX benchmark buy-and-hold; no allocation sweep; no daily exposure.
- Metrics:
  - CAGR: 0.1492
  - Sharpe: 0.6999
  - Max Drawdown: -0.3372
- Precompute port growth file path: `result\best_param_step1_sp500_buy_hold_curve.csv`
- Precompute period: `2016-01-04 to 2026-04-29`

## 2. Allocation: Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15

- Strategy name: `Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15`
- Winner type: `Best Sharpe`
- Config: Monthly rebalanced buy-and-hold allocation; active weights: SPY=35%, Gold=40%, BTC=10%, BIL=15%; tested assets: SPY, Gold, BTC, BIL, IEF, VXUS, TIP; step=5%; max weights SPY=70%, Gold=40%, BTC=10%, BIL=50%, IEF=30%, VXUS=40%, TIP=30%.
- Metrics:
  - CAGR: 0.1347
  - Sharpe: 0.9725
  - Max Drawdown: -0.2157
- Precompute port growth file path: `result\best_param_step2_multi_asset_best_curves.csv`
- Precompute period: `2016-01-05 to 2026-04-28`

## 2B. Managed futures: Managed futures overlay Core/DBMF 85/15

- Strategy name: `Managed futures overlay Core/DBMF 85/15`
- Winner type: `Best Sharpe`
- Config: Monthly rebalanced blend of Step 2 best-Sharpe Core sleeve plus one managed-futures ETF; active weights: Core=85%, DBMF=15%; Core source: Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15; DBMF and KMLM are tested separately on their own overlap windows; managed-futures max weight=30%.
- Metrics:
  - CAGR: 0.1267
  - Sharpe: 0.9399
  - Max Drawdown: -0.1776
- Precompute port growth file path: `result\best_param_step2b_core_dbmf_best_curves.csv`
- Precompute period: `2019-05-09 to 2026-04-28`

## 3B. Daily exposure allocation: Daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25

- Strategy name: `Daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25`
- Winner type: `Best Sharpe`
- Config: Best-Sharpe daily exposure signal applied to SPY, Gold, and BTC, then monthly allocation sweep; daily exposure signals: BTC: MA50, below=0%; GOLD: MA50, below=100%; SPY: MA300, below=50%; active weights: SPY=35%, Gold=30%, BTC=10%, BIL=25%; daily exposure uses lagged close signal by one trading session to avoid lookahead.
- Metrics:
  - CAGR: 0.1238
  - Sharpe: 1.1659
  - Max Drawdown: -0.1531
- Precompute port growth file path: `result\best_param_step3b_daily_exposure_multi_asset_best_curves.csv`
- Precompute period: `2016-01-04 to 2026-04-28`

## 4. Enhanced daily exposure: Country ETF tactical with Gold DD boost

- Strategy name: `Country ETF tactical + Gold DD boost 16`
- Winner type: `Best Sharpe`
- Config: Enhanced daily-exposure allocation using core weights SPY=45%, Gold=30%, BTC=10%, BIL=15%; adds an 8% country ETF tactical satellite funded from SPY; country sleeve selects top 2 ETFs by monthly momentum rank from the full-history country ETF universe; country ETF candidates require at least roughly 10 years of usable price history. Daily exposure signals: SPY: MA300, below=50%; BTC: MA50, below=0%; Gold: drawdown rule `dd252_warn8_crash20_half`; additional Gold boost adds 16% Gold funded from SPY when SPY is below MA200 or SPY 252-day drawdown is at or below -8%. Signals are lagged to the next trading session to avoid lookahead.
- Metrics:
  - CAGR: 0.1359
  - Sharpe: 1.2166
  - Max Drawdown: -0.1377
- Latest selected country ETFs: `EWZ`, `NORW`
- Precompute port growth file path: `result\spy_gold_btc_bil_combined_etf_universe_current_best_curves.csv`
- Precompute summary file path: `result\spy_gold_btc_bil_combined_etf_universe_current_best_summary.csv`
- Precompute selection history file path: `result\spy_gold_btc_bil_combined_etf_universe_current_best_selection_history.csv`
- Precompute period: `2016-10-01 to 2026-04-29`
