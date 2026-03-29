"""
test_full_pipeline.py

End-to-end integration test: data fetch -> calculations -> optimise -> visualise.

Requires an internet connection (calls yfinance).
Run from the project root: python tests/test_full_pipeline.py

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_config, fetch_price_data
from finance_utils import (calculate_returns,
                           calculate_expected_returns,
                           calculate_covariance_matrix,
                           validate_covariance_matrix)
from optimizer import optimise_portfolio, calculate_sharpe_ratio
from visualiser import plot_allocation, plot_efficient_frontier
import numpy as np

def test_full_pipeline():
  """
  Run the complete portfolio optimisation pipeline.
  
  """
  print('='*60)
  print('FTSE 100 PORTFOLIO OPTIMISER - INTEGRATION TEST')
  print('='*60)

  # --- Step 1: Load configuration ---
  config = load_config('config.yaml')
  print(f"Tickers:    {config['tickers']}")
  print(f"Period    {config['start_date']} to {config['end_date']}")
  print(f"Target return:    {config['target_return']:.1%}")

  # -- Step 2: Fetch market data ---
  prices = fetch_price_data(config['tickers'],
                            config['start_date'],
                            config['end_date'])
  
  # --- Step 3: Financial calculations ---
  returns = calculate_returns(prices, method='simple')
  exp_returns = calculate_expected_returns(returns, annualise=True)
  cov_matrix = calculate_covariance_matrix(returns, annualise=True)
  validate_covariance_matrix(cov_matrix)
  print('Covariance matrix validated (symmetric, PSD). OK')

  print('\nExpected Annual Returns:')
  for ticker, ret in zip(config['tickers'], exp_returns):
    print(f'  {ticker:8s} {ret:+.2%}')

  # --- Step 4: Optimise ---
  result = optimise_portfolio(
    exp_returns,
    cov_matrix,
    config['target_return'],
    config['allow_short_selling']
  )

  sharpe = calculate_sharpe_ratio(
    result['expected_return'],
    result['volatility'],
    config['risk_free_rate']
  )

  # --- Step 5: Print results ---
  print('\n' + '-'*60)
  print('OPTIMAL PORTFOLIO RESULTS')
  print('-'*60)
  print(f"Target return: {config['target_return']:.2%}")
  print(f"Achieved return: {result['expected_return']:.2%}")
  print(f"Portfolio vol: {result['volatility']:.2%}")
  print(f"Sharpe ratio: {sharpe:.3f}")
  print(f"Solver success: {result['success']}")
  print('\nOptimal Weights:')
  weight_sum = 0.0
  for ticker, w in zip(config['tickers'], result['weights']):
    print(f' {ticker:8s} {w:.4f} ({w:.2%})')
    weight_sum += w
    assert w >= -1e-6, f'Negative weight for {ticker}: {w:.4f} — FCA violation!'
  print(f' --------')
  print(f' Total {weight_sum:.6f} (should be 1.000000)')
  assert abs(weight_sum - 1.0) < 1e-5, f'Weights do not sum to 1: {weight_sum}'
  print('\nConstraint checks passed: weights >= 0 and sum = 1.0 OK')
  # --- Step 6: Generate plots ---
  os.makedirs('outputs/plots', exist_ok=True)
  plot_allocation(result['weights'], config['tickers'])
  plot_efficient_frontier(exp_returns, cov_matrix, result['weights'], config['risk_free_rate'])
  print('\n' + '='*60)
  print('INTEGRATION TEST PASSED')
  print('='*60)
if __name__ == '__main__':
  test_full_pipeline()