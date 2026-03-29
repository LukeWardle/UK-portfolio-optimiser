"""
finance_utils.py
Financial calculations for portfolio optimisation.

Implements:
  - Return calculations for portfolio optimisation
  - Annualisation (252 UK trading days per year)
  - Covariance matrix computation and validation
  - Positive semi-definiteness checking
"""

import pandas as pd
import numpy as np


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Convert adjusted close prices to daily returns.

    Args:
        prices: DataFrame of adjusted close prices (rows = dates, cols = tickers)
        method: "simple" for percentage returns, "log" for log returns

    Returns:
        DataFrame of daily returns, with first NaN row removed.

    Raises:
        ValueError: If method is not "simple" or "log".
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")

    return returns.dropna()


def calculate_expected_returns(returns: pd.DataFrame, annualise: bool = True) -> np.ndarray:
    """
    Calculate expected (mean) returns per asset.

    Args:
        returns: DataFrame of daily returns
        annualise: If True, multiply by 252 trading days

    Returns:
        NumPy array of expected returns, shape (n_assets,)
    """
    mean_daily_returns = returns.mean()

    if annualise:
        mean_daily_returns = mean_daily_returns * 252

    return mean_daily_returns.values


def calculate_covariance_matrix(returns: pd.DataFrame, annualise: bool = True) -> np.ndarray:
    """
    Calculate the covariance matrix of asset returns.

    Args:
        returns: DataFrame of daily returns
        annualise: If True, multiply by 252 trading days

    Returns:
        NumPy array covariance matrix, shape (n_assets, n_assets)
    """
    cov_daily = returns.cov()

    if annualise:
        cov_matrix = cov_daily * 252
    else:
        cov_matrix = cov_daily

    return cov_matrix.values


def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    Validate that a covariance matrix is symmetric and positive semi-definite.

    Args:
        cov_matrix: NumPy array covariance matrix

    Returns:
        True if valid

    Raises:
        ValueError: If matrix is invalid
    """
    if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-10):
        raise ValueError(
            "Covariance matrix is not symmetric. "
            "Check for NaN or Inf values."
        )

    eigenvalues = np.linalg.eigvals(cov_matrix)
    min_eigenvalue = eigenvalues.min().real

    if min_eigenvalue < -1e-8:
        raise ValueError(
            f"Covariance matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {min_eigenvalue:.2e}. "
            "This usually means insufficient data or near-perfect correlation."
        )

    return True