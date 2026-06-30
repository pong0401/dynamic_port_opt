# Best Param S&P Port Opt Advance

This handoff keeps the fixed Gold drawdown variant and omits the optimized Gold-DD duplicate because the optimized sweep selected the same `35/30/10/25` weights and produced the same metrics.

## Selected Gold-DD Variant

- Selected strategy: `Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25`
- Gold rule: `Gold DD warn-8%->50% crash-20%->25%`
- Reason: fixed and optimized Gold-DD results are identical, so the fixed version is simpler and avoids duplicate reporting.
- Metrics:
  - Total Return: 4.3687
  - CAGR: 0.1190
  - Sharpe: 1.1479
  - Max Drawdown: -0.1428
- Precompute port growth file path: `result\best_param_step3c_gold_drawdown_best_fixed_curve.csv`
- Precompute period: `2016-01-04 to 2026-04-28`

## Final Snapshot

| Strategy | Total Return | Max Drawdown | Sharpe | CAGR | Period |
|---|---:|---:|---:|---:|---|
| S&P 500 buy and hold | 3.1880 | -0.3372 | 0.6999 | 0.1492 | 2016-01-04 to 2026-04-29 |
| Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15 | 5.6164 | -0.2157 | 0.9725 | 0.1347 | 2016-01-05 to 2026-04-28 |
| Extended defensive allocation SPY/BIL/BTAL/SH/PSQ/XLP/XLU/XLV/GLD/BTC 28/3/13/1/2/2/1/5/35/10 | 4.1694 | -0.1645 | 0.9344 | 0.1162 | 2016-01-05 to 2026-04-28 |
| Extended defensive allocation >=3% SPY/BIL/BTAL/XLV/GLD/BTC 29/3/14/6/37/10 | 4.8588 | -0.1803 | 0.9588 | 0.1255 | 2016-01-05 to 2026-04-28 |
| Managed futures overlay Core/DBMF 85/15 | 2.3395 | -0.1776 | 0.9399 | 0.1267 | 2019-05-09 to 2026-04-28 |
| Daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25 | 4.7277 | -0.1531 | 1.1659 | 0.1238 | 2016-01-04 to 2026-04-28 |
| Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25 | 4.3687 | -0.1428 | 1.1479 | 0.1190 | 2016-01-04 to 2026-04-28 |

## Output Files

- `result/best_param_by_step_final_snapshot.csv`
- `result/best_param_by_step_final_snapshot_port_growth_chart.html`
- `result/best_param_step3c_gold_drawdown_summary_chart.html`
- `result/best_param_step3c_gold_drawdown_fixed_mix_sweep.csv`