```markdown
# Risk-Managed Equity Portfolio Simulator

A Python simulation of a multi-asset equity portfolio with rolling risk metrics, systematic pitch evaluation, and performance visualization. Built as part of a quantitative finance challenge.

## Overview

The simulator runs a historical backtest from 2022–2024 on a 20-stock universe. It starts with 5 initial positions and evaluates new stock "pitches" from a portfolio manager approximately every 3 months. Each pitch is accepted or rejected based on a quantitative risk framework, and the portfolio rebalances accordingly.

## Features

- Daily portfolio valuation and return tracking
- Rolling 30-day risk metrics at both the asset and portfolio level
- Systematic pitch evaluation using 4 risk criteria
- Equal-weight rebalancing on accepted pitches with a 1% cash buffer
- Full dashboard visualization with pitch annotations
- Long-only constraint enforcement throughout

## Project Structure

```
.
├── firstname_lastname.py   # Main simulation file
├── ticker_data.csv         # Adjusted close prices for 20 tickers (2022–2024)
├── benchmark_data.csv      # SPY prices for benchmark comparison
├── pitch_schedule.csv      # Dates and tickers of PM pitches
└── portfolio_dashboard.png # Output dashboard (generated on run)
```

## Installation

```bash
pip install numpy pandas matplotlib
```

## Usage

```bash
python firstname_lastname.py
```

The simulation will run automatically and output:
- A 4-panel dashboard saved as `portfolio_dashboard.png`
- A holdings table as of the last simulation day
- The full pitch log with decisions and rationale
- A portfolio risk metrics snapshot for the final 5 days

## Stock Universe

| Initial Holdings | Pitch Universe |
|---|---|
| AAPL, MSFT, JPM, JNJ, XOM | TSLA, PG, CVX, NFLX, HD, AVGO, NVDA, BAC, GS, CAKE, UPS, AMZN, PFE, WMT, DIS |

## Risk Metrics

All metrics are computed on a rolling 30-day window using only historical data.

| Metric | Description |
|---|---|
| **VaR (95%)** | 5th percentile of the empirical 30-day return distribution |
| **Volatility** | Standard deviation of daily returns |
| **Max Drawdown** | Largest peak-to-trough decline over the window |
| **Beta** | Covariance with SPY returns divided by SPY variance |
| **Sharpe Ratio** | Mean return divided by total volatility |
| **Sortino Ratio** | Mean return divided by downside volatility |
| **Downside Volatility** | Std dev of negative daily returns only (custom metric) |

## Pitch Decision Framework

A pitched stock is accepted if it passes **at least 3 of 4** risk checks:

| Check | Threshold | Rationale |
|---|---|---|
| Sharpe > 0 | Positive | Risk-adjusted return is favorable |
| Volatility < 0.02 | Daily | Keeps overall portfolio vol controlled |
| VaR > -0.03 | 95% confidence | Worst expected day loss under 3% |
| Downside Vol < 0.015 | Daily | Left-tail dispersion stays contained |

On acceptance, all active positions are rebalanced to **equal weights** over investable capital (portfolio value minus 1% cash buffer). Rejected pitches leave existing holdings unchanged.

## Dashboard

The output dashboard includes 4 panels:
1. **Portfolio Value vs SPY** — normalized to the same starting value
2. **Daily Returns** — portfolio return series over time
3. **Sharpe & Volatility** — rolling metrics with decision thresholds marked
4. **VaR & Downside Volatility** — rolling metrics with decision thresholds marked

Green dashed lines mark approved pitches, red dashed lines mark rejections.

## Results (2022–2024)

- **Final portfolio value:** ~$169,000 (started at $100,000)
- **Pitches accepted:** 11 of 15
- **Pitches rejected:** TSLA, NFLX, CAKE, DIS (all failed vol/VaR checks)
- **Final holdings:** 16 stocks at roughly equal weight with ~1.5% cash

## Dependencies

- Python 3.8+
- numpy
- pandas
- matplotlib
```
