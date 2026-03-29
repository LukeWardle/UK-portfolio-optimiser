"""
optimizer.py
Portfolio optimisation using Modern Portfolio Theory (Markowitz)

Ssolves the constrained quadratic programme:
  minimise w^T Σ w (portfolio variance)
  subject to Σwᵢ = 1    (all capital invested)
             wᵢ >= 0    (FCA: no short selling)
             w^T μ = R* (achieve target return)

Uses scipy.optimize import minimize with method='SLSQP'.

"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict

def portfolio_variance(weights: np.ndarray,
                       cov_matrix: np.ndarray) -> float:
  """
  Calculate portfolio variance using the quadratic form w^T Σ w.

  This is the objective function we minimise. It captures:
  - Individual asset variances (diagonal terms of Σ)
  - All pairwise covariances (off-diagonal terms)
  - How weighting affects total risk

  Args:
    weights:    Portfolio weights, np.ndarray of shape (n,)
    cov_matrix: Annualised covariance matrix, shape (n, n)

  Returns:
    float - portfolio variance (always >= 0 if Σ is PSD)
  
  """
  # w^T @ Σ @ w - two matrix multiplications
  # Step 1: Σ @ w gives a vector of shape (n,): the covariance-weighted returns
  # Step 2: w^T @ result gives the scalar variance
  return float(weights.T @ cov_matrix @ weights)

def portfolio_return(weights: np.ndarray,
                     expected_returns: np.ndarray) -> float:
  """
  Calculate expected portfolio return: E[Rp] = w^Tμ

  Args:
    weights:          Portfolio weights, shape (n,)
    expected_returns: Mean annualised returns per asset, shape (n,)
  
  Returns:
    float - expected annualised portfolio return
  
  """
  return float(weights.T @ expected_returns)

def optimise_portfolio(expected_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       target_return: float,
                       allow_short_selling: bool = False) -> Dict:
  """
  Find the minimum-variance portfolio that achieves the target return.

  Solves the constrained quadratic programme using scipy SLSQP.

  Args:
    expected returns:     Annualised means returns, shape (n,)
    cov_matrix:           Annulaised covariance matrix, shape (n, n)
    target_return:        Desired annualised portfolio return (e.g. 0.12 = 12%)
    allow_short_selling:  If False, enforce wᵢ >= 0 (FCA compliance)

  Returns:
    dict with keys: weights, expected_return, volatility, variance, sharpe (without risk_free yet), success
  
  """
  n_assets = len(expected_returns)

  # --- Pre-check: is the target return achievable? ---
  # The minimum achievable return is the lowest individual asset return.
  # The maximum is the highest (put 100% in the best-returning asset).
  min_achievable = expected_returns.min()
  max_achievable = expected_returns.max()

  if target_return < min_achievable or target_return > max_achievable:
    raise ValueError(
      f'Target return {target_return:.2%} is outside the achievable range '
      f'[{min_achievable:.2%}, {max_achievable:.2%}]. '
      f'Choose a target return within this range.'
    )
  
  # --- Objective function: minimise portfolio variance ---
  def objective(w):
    return portfolio_variance(w, cov_matrix)
  
  # --- Constraints ---
  constraints = [
    # Constraint 1: weights must sum to exactly 1 (all capital invested)
    # Type 'eg' means the function must equal zero: sum(w) - 1 = 0
    {
      'type': 'eq',
      'fun': lambda w: np.sum(w) - 1.0
    },
    # Constraint 2: portfolio must achieve the target return
    # w^T μ - R* = 0
    {
      'type': 'eq',
      'fun': lambda w: portfolio_return(w, expected_returns) - target_return
    }
  ]
  # --- Bounds on individual weights ---
  if not allow_short_selling:
    #FCA compliance: no short selling (wᵢ >= 0)
    # Upper bound 1.0 prevents 100%+ allocation to one asset
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
  else:
    # Short selling allowed: weights between -1 and +1
    bounds = tuple((-1.0, 1.0) for _ in range(n_assets))

  # --- Initial guess: equal weights ---
  # Satisfies the sum=1 constraint from the start.
  # For convex problems (which this is), the initial guess
  # affects speed but not the final answer.
  initial_weights = np.full(n_assets, 1.0 / n_assets)

  # --- Run the optimisation ---
  result = minimize (
    objective,
    initial_weights,
    method='SLSQP',     # Sequential Least Squares Programming
    bounds=bounds,
    constraints=constraints,
    options={'ftol': 1e-9,    # Convergence tolerance
             'maxiter': 1000, # Max iterations
             'disp': False}   # Suppress solver output
  )

  if not result.success:
    print(f'Optimisation warning: {result.message}')

  # --- Compute final portfolio statistics ---
  optimal_w = result.x
  port_ret = portfolio_return(optimal_w, expected_returns)
  port_var = portfolio_variance(optimal_w, cov_matrix)
  port_vol = np.sqrt(port_var)  # Standard deviation = sqrt(variance)

  return {
    'weights': optimal_w,
    'expected_return': port_ret,
    'volatility': port_vol,
    'variance': port_var,
    'success': result.success,
    'solver_message': result.message
  }

def calculate_sharpe_ratio(port_return: float,
                           port_volatility: float,
                           risk_free_rate: float) -> float:
  """
  Calculate the sharpe ratio: risk-adjusted return.

  Sharpe = (E[Rp] - Rf) / σp

  Interpretation:
    Sharpe > 1: Good  - earning more than 1 unit of return per unit of risk
    Sharpe < 0: Poor  - Would be better off in risk-free gilts

  Args:
    port_return: Expected annualised portfolio return
    port_volatility: Portfolio standard deviation (annualised)
    risk_free_rate: UK 10-year gilt rate (e.g. 0.045 = 4.5%)

  Returns:
    float - Sharpe ratio
  
  """
  if port_volatility == 0:
    raise ValueError('Cannot compute Sharpe ratio for zero-volatility portfolio.')
  return (port_return - risk_free_rate) / port_volatility