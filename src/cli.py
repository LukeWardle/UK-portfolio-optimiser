"""
cli.py
Command-line interface for the UK Portfolio Optimiser.

Entry point: python src/cli.py [options]

Orchestrates the full pipeline:
  1. Parse arguments
  2. Load config
  3. Fetch market data (data loader)
  4. Compute returns and covariance (finance_utils)
  5. Optimise portfolio (optimizer)
  6. Print formatted results
  7. Save JSON output
  8. Append FCA audit log entry
  9. Generate plots (if --plot flag set)

Exit codes:
  0 - Successful optimisation
  1 - Any failure (data error, infeasible target, solver failure)

"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src/ to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_config, fetch_price_data
from finance_utils import (calculate_returns,
                           calculate_expected_returns,
                           calculate_covariance_matrix,
                           validate_covariance_matrix)
from optimizer import optimise_portfolio, calculate_sharpe_ratio

def build_parser() -> argparse.ArgumentParser:
  """Build and return the argument parser."""
  parser = argparse.ArgumentParser(
    prog='uk-portfolio-optimizer',
    description='FTSE 100 Portfolio Optimiser - Modern Portfolio Theory',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
    Examples:
      # Run with default config (config.yaml in current directory):
      python src/cli.py
      # Run with a custom config file:
      python src/cli.py --config strategies/conservative.yaml
      # Save outputs to a named directory and generate plots:
      python src/cli.py --output-dir results/strategy1 --plot
      # Run silently (no terminal output) for use in scripts:
      python src/cli.py --quiet
      # Override the target return from the command line:
      python src/cli.py --target-return 0.08
    '''
  )
  parser.add_argument(
    '-c', '--config',
    default='config.yaml',
    help='Path to YAML configuration file (default: config.yaml)'
  )
  parser.add_argument(
    '-o', '--output-dir',
    default='outputs',
    help='Directory for JSON output and plots (default: outputs/)'
  )
  parser.add_argument(
    '--target-return',
    type=float,
    default=None,
    help='Override target return from config (e.g. 0.12 for 12%%)'
  )
  parser.add_argument(
    '--plot',
    action='store_true',
    help='Generate and save allocation chart and efficient frontier'
  )
  parser.add_argument(
    '-q', '--quiet',
    action='store_true',
    help='Suppress all terminal output (useful in automated pipelines)'
  )
  parser.add_argument(
    '--no-audit',
    action='store_true',
    help='Skip writing to FCA audit log (for dry runs only)'
  )
  return parser


def print_results(config: dict,
                  result: dict,
                  sharpe: float,
                  tickers: list) -> None:
  """Print formatted optimisation results to the terminal."""
  width = 62
  print()
  print('=' * width)
  print(' UK PORTFOLIO OPTIMISER - RESULTS')
  print('=' * width)
  print(f"  Config:     {config.get('_source', 'config.yaml')}")
  print(f"  Period:     {config['start_date']} to {config['end_date']}")
  print(f"  Target return:      {config['target_return']:.2%}")
  print(f"  Risk-free rate:     {config['risk_free_rate']:.2%}  (UK 10yr gilt)")
  print(f"  Short selling:      {'Allowed' if config['allow_short_selling'] else 'Prohibited (FCA)'}")
  print('-' * width)
  print(f"  {'METRIC':<28}  {'VALUE':>10}")
  print('-' * width)
  print(f"  {'Expected return':<28} {result['expected_return']:>9.2%}")
  print(f" {'Portfolio volatility':<28} {result['volatility']:>9.2%}")
  print(f" {'Portfolio variance':<28} {result['variance']:>9.4f}")
  print(f" {'Sharpe ratio':<28} {sharpe:>9.3f}")
  print(f" {'Solver status':<28} {'SUCCESS' if result['success'] else 'WARNING':>10}")
  print('-' * width)
  print(f" {'OPTIMAL WEIGHTS':}")
  print(f" {'Ticker':<12} {'Weight':>8} {'Allocation':>12}")
  print('-' * width)

  for ticker, w in zip(tickers, result['weights']):
    bar_len = int(w * 30) # ASCII bar, max 30 chars = 100%
    bar = '█' * bar_len
    print(f" {ticker:<12} {w:>7.4f} {w:>9.2%} {bar}")
  
  print('-' * width)
  weight_sum = sum(result['weights'])
  print(f" {'TOTAL':<12} {weight_sum:>7.4f} {weight_sum:>9.2%}")
  print('=' * width)
  print()

  # FCA compliance summary
  neg_weights = [t for t, w in zip(tickers, result['weights']) if w < -1e-6]

  if neg_weights:
    print(f'  WARNING: Negative weights detected: {neg_weights}')
    print(' This violates FCA no-short-selling constraint.')
  else:
    print(' FCA Compliance: PASS - all weights non-negative.')
  print()

def save_json_output(config: dict,
                     result: dict,
                     sharpe: float,
                     tickers: list,
                     output_dir: str) -> str:
  """
  Save optimisation results as a timestamped JSON file.

  Args:
    config:     Configuration dictionary
    result:     Optimisation result dictionary
    sharpe:     Sharpe ratio
    tickers:    List of ticker symbols
    output_dir: Directory to save into

  Returns:
    Path to the saved JSON file as a string.
  
  """
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f'portfolio_{timestamp}.json'
  filepath = Path(output_dir) / filename
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  output = {
    # Metadata
    'timestamp':      datetime.now().isoformat(),
    'tool_version':   '1.0.0',

    # Inputs
    'inputs': {
      'tickers':              tickers,
      'start_date':           config['start_date'],
      'end_date':             config['end_date'],
      'target_return':        config['target_return'],
      'risk_free_rate':       config['risk_free_rate'],
      'allow_short_selling':  config['allow_short_selling'],
    },

    # Results
    'results': {
      'weights':              {t: float(w) for t, w in zip(tickers, result['weights'])},
      'expected_return':      float(result['expected_return']),
      'portfolio_volatility': float(result['volatility']),
      'portfolio_variance':   float(result['variance']),
      'sharpe_ratio':         float(sharpe),
      'solver_success':       bool(result['success']),
      'solver_message':       result.get('solver_message', ''),
    },

    # Compliance
    'compliance': {
      'weights_sum': float(sum(result['weights'])),
      'negative_weights': [t for t, w in zip(tickers, result['weights']) if w < -1e-6],
      'fca_compliant': all(w >= -1e-6 for w in result['weights']),
    }
  }

  with open(filepath, 'w') as f:
    json.dump(output, f, indent=2)

  return str(filepath)

def append_audit_log(config: dict,
                     result: dict,
                     sharpe: float,
                     tickers: list,
                     json_filepath: str,
                     log_dir: str = 'documentation') -> None:
  """
  Append a structured entry to the FCA compliance audit log.

  The audit log is append-only-never overwritten.
  Each entry is self-contained and human readable.

  Args:
    config:         Configuration used for this run
    result:         Optimisation result
    sharpe:         Sharpe ratio
    tickers:        List of ticker symbols
    json_filepath:  Path to the JSON output file for cross-reference
    log_dir:        Directory containing the audit log
  
  """

  Path(log_dir).mkdir(parents=True, exist_ok=True)
  log_path = Path(log_dir) / 'audit_log.txt'

  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  separator = '=' * 72
  weight_lines = '\n'.join(
    f'    {t:<10} {w:.4f} ({w:.2%})'
    for t, w in zip(tickers, result['weights'])
  )

  fca_status = 'COMPLIANT' if all(w >= -1e-6 for w in result['weights']) else 'NON-COMPLIANT'

  entry = f'''
  {separator}
  AUDIT ENTRY - UK PORTFOLIO OPTMISER
  Timestamp:        {timestamp}
  JSON output:      {json_filepath}
  {separator}
  INPUTS
  Tickers:          {', '.join(tickers)}
  Period:           {config['start_date']} to {config['end_date']}
  Target return:    {config['target_return']:.2%}
  Risk-free rate:   {config['risk_free_rate']:.2%}
  Short selling:    {'Allowed' if config['allow_short_selling'] else 'Prohibited'}
  RESULTS
  Expected return:  {result['expected_return']:.4%}
  Volatility:       {result['volatility']:.4%}
  Sharpe ratio:     {sharpe:.4f}
  Solver status:    {'SUCCESS' if result['success'] else 'WARNING'}
  OPTIMAL WEIGHTS
  {weight_lines}
  Total:            {sum(result['weights']):.6f}
  FCA COMPLIANCE
  Status:           {fca_status}
  Negative weights: {[t for t, w in zip(tickers, result['weights']) if w < -1e-6] or 'None'}
  {separator}
  '''
  with open(log_path, 'a') as f:  # 'a' = append, never overwrite
    f.write(entry)

def main() -> None:
  """Main entry point: parse args, run pipeline, handle all errors."""

  parser = build_parser()
  args = parser.parse_args()

  # --- Step 1: Load configuration ---
  try:
    config = load_config(args.config)
    config['_source'] = args.config   # store for reporting
    
  except FileNotFoundError:
    print(f'ERROR: Configuration file not found: {args.config}')
    print(' Create a config.yaml file or specify --config <path>')
    sys.exit(1)

  # CLI flag overrides config value
  if args.target_return is not None:
    if not args.quiet:
      print(f'CLI override: target_return = {args.target_return:.2%}')
      print(f'  (config.yaml value {config["target_return"]:.2%} is ignored)')
    config['target_return'] = args.target_return

  # --- Step 2: Fetch market data ---
  if not args.quiet:
    print(f'Loading market data for {len(config["tickers"])} FTSE 100 stocks...')
  
  try:
    prices = fetch_price_data(config['tickers'],
                              config['start_date'],
                              config['end_date'])
  except ValueError as e:
    print(f'ERROR: Date loading failed: {e}')
    sys.exit(1)

  # --- Step 3: Financial calculations ---
  try:
    returns = calculate_returns(prices, method='simple')
    exp_returns = calculate_expected_returns(returns, annualise=True)
    cov_matrix = calculate_covariance_matrix(returns, annualise=True)
    validate_covariance_matrix(cov_matrix)
  except ValueError as e:
    print(f'ERROR: Financial calculation failed: {e}')
    sys.exit(1)

  # --- Step 4: Optimise ---
  try:
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
  except ValueError as e:
    print(f'ERROR: Optimisation failed: {e}')
    print(' Check that the target_return is within the achievable range.')
    sys.exit(1)
  
  tickers = list(prices.columns)

  # --- Step 5: Print formatted results ---
  if not args.quiet:
    print_results(config, result, sharpe, tickers)

  # --- Step 6: Save JSON output ---
  json_path = save_json_output(config, result, sharpe, tickers, args.output_dir)
  if not args.quiet:
    print(f'Results saved: {json_path}')

  # --- Step 7: Write FCA audit log ---
  if not args.no_audit:
    append_audit_log(config, result, sharpe, tickers, json_path)
    if not args.quiet:
      print('Audit log updated: documentation/audit_log.txt')

  # --- Step 8: Generate plots (optional) ---
  if args.plot:
    try:
      from visualiser import plot_allocation, plot_efficient_frontier
      plots_dir = str(Path(args.output_dir) / 'plots')
      Path(plots_dir).mkdir(parents=True, exist_ok=True)
      plot_allocation(result['weights'], tickers, str(Path(plots_dir) / 'allocation.png'))
      plot_efficient_frontier(exp_returns,
                              cov_matrix,
                              result['weights'],
                              config['risk_free_rate'],
                              str(Path(plots_dir) / 'efficient_frontier.png')
      )
      if not args.quiet:
        print(f'Plots saved to: {plots_dir}/'
      )
    except Exception as e:
      print(f'WARNING: Plot generation failed: {e}')
      # Plots are optional - do not exit with error code

  # --- Exit successfully ---
  sys.exit(0)

if __name__ == '__main__':
  main()

