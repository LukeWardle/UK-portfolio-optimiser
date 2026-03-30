"""
test_portfolio_maths.py
Unit tests for financial calculations using synthetic data.
No internet connection required - all data is constructed in code.
Run with: pytest tests/test_portfolio_maths.py -v

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..','src'))

import numpy as np
import pytest
from finance_utils import (calculate_expected_returns,
                           calculate_covariance_matrix,
                           validate_covariance_matrix)
from optimizer import (portfolio_variance,
                       portfolio_return,
                       optimise_portfolio,
                       calculate_sharpe_ratio)

# -- Fixtures: synthetic data with known analytical answers
def make_two_asset_data():
  """Two uncorrelated assets: easy to verify by hand."""
  mu = np.array([0.10, 0.20])     # 10% and 20% annual return
  cov = np.array([[0.04, 0.00],   # σ₁ = 20%, uncorrelated
                  [0.00, 0.09]])  # σ₂ = 30%
  return mu, cov

# -- Test 1: portfolio_variance quadratic form
def test_portfolio_variance_equal_weights():
  """w^T Σ w for equal weights in a 2-asset uncorrelated portfolio."""
  mu, cov = make_two_asset_data()
  w = np.array([0.5, 0.5])
  # Expected: 0.5² × 0.04 + 0.5² × 0.09 = 0.01 + 0.0225 = 0.0325
  result = portfolio_variance(w, cov)
  assert abs(result - 0.0325) < 1e-10, f'Expected 0.0325, got {result}'

def test_portfolio_variance_100_percent_asset1():
  """100% in asset 1 should give exactly asset 1 variance."""
  mu, cov = make_two_asset_data()
  w = np.array([1.0, 0.0])
  result = portfolio_variance(w, cov)
  assert abs(result - 0.04) < 1e-10

# -- Test 2: portfolio_return dot product
def test_portfolio_return_midpoint():
  """Equal weights on 10%/20% assets should give 15%."""
  mu, cov = make_two_asset_data()
  w = np.array([0.5, 0.5])
  result = portfolio_return(w, mu)
  assert abs(result - 0.15) < 1e-10

# -- Test 3: optimise_portfolio constraint satisfaction
def test_optimise_weights_sum_to_one():
  """Optimal weights must always sum to exactly 1.0."""
  mu, cov = make_two_asset_data()
  result = optimise_portfolio(mu, cov, target_return=0.15)
  assert abs(np.sum(result['weights']) - 1.0) < 1e-5

def test_optimise_no_negative_weights():
  """FCA constraint: all weights must be >= 0."""
  mu, cov = make_two_asset_data()
  result = optimise_portfolio(mu, cov, target_return=0.15, allow_short_selling=False)
  assert np.all(result['weights'] >= -1e-6), (
  f'Negative weights found: {result["weights"]}'
)
  
def test_optimise_achieves_target_return():
  """Achieved return must match the target (within tolerance)."""
  mu, cov = make_two_asset_data()
  target = 0.15
  result = optimise_portfolio(mu, cov, target_return=target)
  assert abs(result['expected_return'] - target) < 1e-4,(f'Target {target:.2%} not achieved: {result["expected_return"]:.2%}'
)
  
def test_optimise_known_solution():
  """For target=0.15 with equal-var uncorrelated assets: expect [0.5, 0.5]."""
  mu, cov = make_two_asset_data()
  result = optimise_portfolio(mu, cov, target_return=0.15)
  expected = np.array([0.5, 0.5])
  assert np.allclose(result['weights'], expected, atol=1e-4), (f'Expected [0.5, 0.5], got {result["weights"]}'
)

# Test 4: infeasible target raises ValueError
def test_infeasible_target_raises():
  """Target return above max asset return should raise ValueError."""
  mu, cov = make_two_asset_data()
  with pytest.raises(ValueError, match='outside the achievable range'):
    optimise_portfolio(mu, cov, target_return=0.99) # 99% is impossible


# Test 5: covariance matrix validation
def test_valid_covariance_passes():
  """A valid PSD covariance matrix should pass validation without error."""
  _, cov = make_two_asset_data()
  assert validate_covariance_matrix(cov) is True

def test_non_psd_covariance_fails():
  """A matrix with a negative eigenvalue should raise ValueError."""
  bad_cov = np.array([[1.0, 2.0],  # off-diagonal larger than diagonal
                      [2.0, 1.0]]) # eigenvalues: 3 and -1
  with pytest.raises(ValueError, match='positive semi-definite'):
    validate_covariance_matrix(bad_cov)


# Test 6: Sharpe ratio
def test_sharpe_ratio_calculation():
  """Sharpe = (return - risk_free) / volatility."""
  sharpe = calculate_sharpe_ratio(0.15, 0.20, 0.045)
  # (0.15 - 0.045) / 0.20 = 0.105 / 0.20 = 0.525
  assert abs(sharpe - 0.525) < 1e-10