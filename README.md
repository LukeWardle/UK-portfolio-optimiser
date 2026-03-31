# UK Portfolio Optimiser
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)
![FCA](https://img.shields.io/badge/FCA-compliant-blue)
A command-line tool for optimal FTSE 100 portfolio allocation using
Modern Portfolio Theory (Markowitz, 1952). Built for quantitative
analysts at FCA-regulated UK investment firms.
## The Problem It Solves
Given historical prices for a set of FTSE 100 stocks, find the
portfolio weights that **minimise risk (variance) while achieving a
target annual return**, subject to:
- All capital invested (weights sum to 1)
- No short selling (FCA retail compliance)
## Quick Start (Windows)
```
git clone https://github.com/LukeWardle/uk-portfolio-optimiser
cd uk-portfolio-optimiser
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/cli.py
```
## Example Output
```
==============================================================
UK PORTFOLIO OPTIMISER — RESULTS
==============================================================
Config: config.yaml
Period: 2023-01-01 to 2024-01-01
Target return: 12.00%
Risk-free rate: 4.50% (UK 10yr gilt)
Short selling: Prohibited (FCA)
--------------------------------------------------------------
METRIC VALUE
--------------------------------------------------------------
Expected return 12.00%
Portfolio volatility 15.34%
Sharpe ratio 0.493
Solver status SUCCESS
--------------------------------------------------------------
OPTIMAL WEIGHTS
Ticker Weight Allocation
--------------------------------------------------------------
SHEL.L 0.2530 25.30% ████████
TSCO.L 0.4820 48.20% ██████████████
BP.L 0.2650 26.50% ████████
--------------------------------------------------------------
TOTAL 1.0000 100.00%
==============================================================
FCA Compliance: PASS — all weights non-negative.
```
## Changing the Configuration
Edit `config.yaml` to change tickers, dates, or targets:
```yaml
tickers: # FTSE 100 stocks (.L = London Stock Exchange)
- BARC.L # Barclays
- SHEL.L # Shell
- AZN.L # AstraZeneca
- TSCO.L # Tesco
- BP.L # BP
start_date: '2023-01-01'
end_date: '2024-01-01'
target_return: 0.12 # 12% annual target
risk_free_rate: 0.045 # UK 10-year gilt rate
allow_short_selling: false
```
## CLI Reference
| Flag | Default | Description |
|-----------------------|-----------------|--------------------------------------|
| -c / --config | config.yaml | Path to configuration file |
| -o / --output-dir | outputs/ | Directory for JSON output |
| --target-return FLOAT | from config | Override target return (e.g. 0.10) |
| --plot | off | Generate allocation and frontier PNG |
| -q / --quiet | off | Suppress terminal output |
| --no-audit | off | Skip FCA audit log (dry runs only) |
## Project Structure
```
uk-portfolio-optimiser/
├── src/
│ ├── cli.py # Entry point, orchestrator, audit log
│ ├── data_loader.py # yfinance market data pipeline
│ ├── finance_utils.py # Returns, covariance, validation
│ ├── optimizer.py # SLSQP constrained optimisation
│ └── visualiser.py # Efficient frontier, allocation chart
├── tests/
│ ├── test_portfolio_maths.py # 11 unit tests, no internet
│ └── test_full_pipeline.py # Integration test
├── documentation/
│ └── audit_log.txt # FCA compliance audit trail (append-only)
├── outputs/ # JSON results and plots (git-ignored)
├── config.yaml # User configuration
├── requirements.txt
└── README.md
```
## Mathematical Background
This tool implements Markowitz Modern Portfolio Theory (1952). Given
expected returns vector **μ** and covariance matrix **Σ**, it solves:
```
minimise w^T Σ w (portfolio variance)
subject to Σwᵢ = 1 (all capital invested)
wᵢ >= 0 (FCA: no short selling)
w^T μ = R* (achieve target return R*)
```
The solver uses scipy.optimize.minimize with method='SLSQP'
(Sequential Least Squares Programming) — the same algorithm used
by Bloomberg portfolio analytics.
## Running Tests
```
pytest tests\test_portfolio_maths.py -v # No internet required
python tests\test_full_pipeline.py # Requires internet (yfinance)
```
## Limitations
- Historical data only — no live price streaming
- yfinance subject to rate limits; retry if download fails
- 252-day lookback is fixed; does not adapt to market regime changes
- No multi-currency support; all assets must trade in GBP
- Assumes returns are stationary; non-stationarity is not detected
## Licence
MIT — see LICENSE file.
## Author
Luke Wardle | Week 2 Sunday Capstone | Module 1: Linear Algebra
Built as part of the AI Engineering programme — UK cohort.