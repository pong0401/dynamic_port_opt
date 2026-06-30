# Best Param By Step - Port Opt Advance Handoff

This file lists only the best-Sharpe strategy from each final notebook family. Assets with zero allocation are intentionally omitted from strategy names and active-weight configs.

## 1. S&P buy hold: S&P 500 buy and hold

- Strategy name: `S&P 500 buy and hold`
- Winner type: `Benchmark Sharpe`
- Config: SPX benchmark buy-and-hold; no allocation sweep; no daily exposure.
- Metrics:
  - Total Return: 3.1880
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
  - Total Return: 5.6164
  - CAGR: 0.1347
  - Sharpe: 0.9725
  - Max Drawdown: -0.2157
- Precompute port growth file path: `result\best_param_step2_multi_asset_best_curves.csv`
- Precompute period: `2016-01-05 to 2026-04-28`

## 2C. Extended allocation: Extended defensive allocation SPY/BIL/BTAL/SH/PSQ/XLP/XLU/XLV/GLD/BTC 28/3/13/1/2/2/1/5/35/10

- Strategy name: `Extended defensive allocation SPY/BIL/BTAL/SH/PSQ/XLP/XLU/XLV/GLD/BTC 28/3/13/1/2/2/1/5/35/10`
- Winner type: `Best Sharpe`
- Config: Memory-aware random capped search across SPY plus managed futures, T-Bills, anti-beta, inverse equity hedges, defensive sectors, GLD, volatility, BTC, and ETH; active weights: SPY=28%, BIL=3%, BTAL=13%, SH=1%, PSQ=2%, XLP=2%, XLU=1%, XLV=5%, GLD=35%, BTC=10%; candidates=60,000; batch size=1,000; top rows kept=250; seed=20260607.
- Metrics:
  - Total Return: 4.1694
  - CAGR: 0.1162
  - Sharpe: 0.9344
  - Max Drawdown: -0.1645
- Precompute port growth file path: `result\best_param_step2c_extended_defensive_allocation_best_curves.csv`
- Precompute period: `2016-01-05 to 2026-04-28`

## 2C. Extended >=3%: Extended defensive allocation >=3% SPY/BIL/BTAL/XLV/GLD/BTC 29/3/14/6/37/10

- Strategy name: `Extended defensive allocation >=3% SPY/BIL/BTAL/XLV/GLD/BTC 29/3/14/6/37/10`
- Winner type: `Pruned Weights`
- Config: Derived from the best Step 2C allocation by removing assets below 3% weight, renormalizing the remaining weights, and monthly rebalancing; active weights: SPY=29%, BIL=3%, BTAL=14%, XLV=6%, GLD=37%, BTC=10%.
- Metrics:
  - Total Return: 4.8588
  - CAGR: 0.1255
  - Sharpe: 0.9588
  - Max Drawdown: -0.1803
- Precompute port growth file path: `result\best_param_step2c_extended_defensive_pruned_3pct_curve.csv`
- Precompute period: `2016-01-05 to 2026-04-28`

## 2B. Managed futures: Managed futures overlay Core/DBMF 85/15

- Strategy name: `Managed futures overlay Core/DBMF 85/15`
- Winner type: `Best Sharpe`
- Config: Monthly rebalanced blend of Step 2 best-Sharpe Core sleeve plus one managed-futures ETF; active weights: Core=85%, DBMF=15%; Core source: Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15; DBMF and KMLM are tested separately on their own overlap windows; managed-futures max weight=30%.
- Metrics:
  - Total Return: 2.3395
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
  - Total Return: 4.7277
  - CAGR: 0.1238
  - Sharpe: 1.1659
  - Max Drawdown: -0.1531
- Precompute port growth file path: `result\best_param_step3b_daily_exposure_multi_asset_best_curves.csv`
- Precompute period: `2016-01-04 to 2026-04-28`

## 3C. Gold DD fixed: Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25

- Strategy name: `Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25`
- Winner type: `Best Sharpe`
- Config: Uses the same fixed allocation as Step 3B, but replaces Gold's no-op trend exposure with drawdown exposure rule: Gold DD warn-8%->50% crash-20%->25%.
- Metrics:
  - Total Return: 4.3687
  - CAGR: 0.1190
  - Sharpe: 1.1479
  - Max Drawdown: -0.1428
- Precompute port growth file path: `result\best_param_step3c_gold_drawdown_best_fixed_curve.csv`
- Precompute period: `2016-01-04 to 2026-04-28`
