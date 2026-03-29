"""
data_loader.py
Market data pipeline for UK stocks using yfinance.

Handles:
- FTSE 100 ticker fetching (.L suffix for London Stock Exchange)
- Missing data validation and forward-fill.
- Corporate action adjustments (Adj Close)
- Data sufficiency checks (minimum 50 trading days)
- Currencyu normalisation (GBP - ensured by .L tickers)

"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import yaml

def load_config(config_path: str = 'config.yaml') -> dict:
  """
  Load portfolio configuration from YAML file.

  Args:
    config_path: Path to the YAML configuration file.
                 Defaults to config.yaml in the working directory.

  Returns:
    Configuration dictionary with keys: tickers, start_date, end_date, target_return, risk_free_rate, allow_short_selling.

  Raises:
    FileNotFoundError: If config file does not exist.
  
  """
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  return config

def fetch_price_data(tickers: List[str],
                     start_date: str,
                     end_date: str) -> pd.DataFrame:
  """
  Fetch historical adjusted close prices for FTSE 100 stocks.

  Args:
    tickers: List of LSE ticker symbols (e.g ['BARC.L', 'SHEL.L'])
    start_date: Start date string in YYYY-MM-DD format
    end_date: End date string in YYYY-MM-DD format

  Returns:
    pd.DataFrame with adjusted close prices, indexed by date.
    Columns are ticker symbols. All values are float 64 GBP prices.

  Raises:
    ValueError: If no data returned, tickers invalid, or insufficient history (< 50 trading days).
  
  """
  print(f'Fetching data for {len(tickers)} stocks: {tickers}')
  print(f'Period: {start_date} to {end_date}')

  try:
    # yfinance download - 'Adj Close' corrects for splits and dividends
    raw_data = yf.download(tickers,
                       start=start_date,
                       end=end_date,
                       progress=False) 
    # Use adjusted close if available, otherwise fall back to close
    if 'Adj Close' in raw_data:
      data = raw_data['Adj Close']
    elif 'Close' in raw_data:
      print("WARNING: 'Adj Close' not found. Falling back to 'Close'.")
      data = raw_data['Close']
    else:
      raise ValueError(
        f"Downloaded data does not contain 'Adj Close' or 'Close'. "
        f"Available columns: {raw_data.columns}"
      )
    
    # yfinance returns a Series (not DataFrame) for a single ticker 
    if isinstance(data, pd.Series):
      data = data.to_frame(name=tickers[0])

    # --- Validation 1: Was any data returned at all? ---
    if data.empty:
      raise ValueError(
        'No data returned. Check tickers are valid LSE symbols'
        '(must end in .L) and dates are not in the future.'
      )
    
    # --- Validation 2: Were all requested tickers found? ---
    missing = set(tickers) - set(data.columns)
    if missing:
      print(f'WARNING: These tickers returned no data: {missing}')
      print('They will be excluded from the analysis.')

    # --- Data repair: forward-fill gaps up to 3 days ---
    # Handles bank holidays and short trading halts.
    # limit=3 is conservative: longer gaps suggest a real problem.
    data = data.ffill(limit=3)

    # Drop any rows that still contain NaN after forward-fill
    data = data.dropna()

    # --- Validation3: Is there enough data for stable estimates? ---
    # Financial rule: need at least 2x samples as assets.
    # With 5 assets, need >= 10 rows minimum.
    # We require 50 to ensure statistically stable covariance estimates.
    if len(data) < 50:
      raise ValueError(
        f'Only {len(data)} trading days of data available.'
        f'Need at least 50 for stable covariance estimates.'
        f'Try extending the date range.'
      )
    
    print(f'Downloaded {len(data)} trading days for {len(data.columns)} stocks.')
    return data
  
  except Exception as e:
    raise ValueError(f'Data fetch failed: {e}')