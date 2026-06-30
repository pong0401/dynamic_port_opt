# Best US/TH Side Trigger Config

Selected from fee/slippage-adjusted results only.

- Strategy: `Side trigger whipsaw confirm2_hold5 realloc, fee+slippage`
- Objective: `min_vol_mom_tilt`
- Fee + slippage: `17.0` bps
- US trigger: `SPY + ^VIX`
- Thailand trigger: `^SET.BK`
- Reallocate stock sleeve: `True`
- Whipsaw filter: `True`
- Confirm days: `2`
- Minimum hold days: `5`
- US assets: `30`
- Thailand assets: `30`
- Max stock weight: `6.00%` inside equity sleeve
- Strategic weights: `Equity 60% / Gold 30% / BTC 10%`

## Metrics

- CAGR: `20.7974%`
- Sharpe: `1.2348`
- Sortino: `1.6390`
- Max Drawdown: `-16.6182%`
- Hit Rate: `0.5611`
- Start: `2017-12-29`
- End: `2026-04-29`

## Files

- `result/us_th_best_config_side_trigger_fee_slippage.json`
- `result/us_th_best_config_side_trigger_fee_slippage.csv`
- `result/us_th_side_trigger_latest_asset_weights_thb.csv`
- `result/us_th_side_trigger_reallocation_summary_thb.csv`