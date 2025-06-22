import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import os
import shutil
import glob
import yfinance as yf
import pandas_ta as ta
import random
import argparse
from datetime import datetime, timedelta

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================

# --- Path to the C++ trading simulation executable ---
SIMULATION_EXECUTABLE = "./bin/autonomous_trading"

# --- Data Source Configuration ---
# A default list of tickers to use if none are provided via a file
DEFAULT_STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "PFE", "MRK", "PEP", "KO",
    "XOM", "CVX", "WMT"
]

# ==============================================================================
# HELPER FUNCTION: SETUP & CLEANUP
# ==============================================================================

def setup_session_directories(run_id):
    """Creates a full directory structure for the current training session."""
    session_root = Path(f"training_run_{run_id}")
    dirs = {
        "root": session_root,
        "data": session_root / "data",
        "logs": session_root / "logs",
        "reports": session_root / "reports",
        "best_model_state": session_root / "best_model_state",
        "temp_model_state": session_root / "temp_model_state",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"Session directories created at: {session_root.resolve()}")
    return dirs

def cleanup_generated_data(data_dir):
    """Removes the 'generated' subdirectory from previous runs to ensure fresh data."""
    generated_dir = Path(data_dir) / "generated"
    if generated_dir.exists():
        print(f"Cleaning up old data from: {generated_dir}")
        shutil.rmtree(generated_dir)

def promote_model(temp_dir, best_dir):
    """Copies and renames the new global best model files."""
    print("Promoting new global best model...")
    temp_prefix = temp_dir / "temp_agent_state"
    best_prefix = best_dir / "best_agent_state"

    temp_files = glob.glob(f"{temp_prefix}_*")
    if not temp_files:
        print("Warning: No temporary model files found to promote.")
        return

    best_dir.mkdir(parents=True, exist_ok=True)

    # Remove old best files before copying new ones
    for f_old in glob.glob(f"{best_prefix}_*"):
        os.remove(f_old)

    # Copy and rename files from temp to best
    for f_temp in temp_files:
        temp_filename = Path(f_temp).name
        best_filename = temp_filename.replace("temp_agent_state", "best_agent_state")
        destination_path = best_dir / best_filename
        shutil.copy(f_temp, destination_path)
        print(f"  Copied {temp_filename} to {destination_path}")

# ==============================================================================
# HELPER FUNCTION: DATA PREPARATION
# ==============================================================================

def fetch_and_process_single_stock(ticker, data_dir):
    """
    Fetches daily data for a random period (6mo to 3yr) from a random start
    date in the last 10 years. Processes data, adds technical indicators,
    and saves it to a CSV file. Returns the path to the CSV file, or None on failure.
    """
    print(f"Fetching varied historical daily data for {ticker}...")
    try:
        ten_years_ago = datetime.now() - timedelta(days=365 * 10)
        max_start_date = datetime.now() - timedelta(days=365 * 3)
        if max_start_date < ten_years_ago:
            max_start_date = ten_years_ago + timedelta(days=1)

        random_start_offset = random.randint(0, (max_start_date - ten_years_ago).days)
        fetch_start_date = ten_years_ago + timedelta(days=random_start_offset)

        random_duration_days = random.randint(180, 365 * 3)
        fetch_end_date = fetch_start_date + timedelta(days=random_duration_days)

        print(f"Fetching daily data for {ticker} from {fetch_start_date.strftime('%Y-%m-%d')} to {fetch_end_date.strftime('%Y-%m-%d')}")

        stock_data = yf.download(
            ticker,
            start=fetch_start_date,
            end=fetch_end_date,
            interval="1d",
            progress=False
        )

        if stock_data.empty:
            print(f"No data found for {ticker} in the selected date range."); return None

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        stock_data.ta.strategy(ta.CommonStrategy)
        stock_data.dropna(inplace=True)

        if stock_data.empty:
            print(f"Data for {ticker} became empty after TA calculation and cleanup."); return None

        stock_data.columns = stock_data.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

        start_str = fetch_start_date.strftime('%Y%m%d')
        end_str = fetch_end_date.strftime('%Y%m%d')
        filename = data_dir / f"{ticker}_1d_{start_str}_to_{end_str}.csv"
        stock_data.to_csv(filename)
        print(f"Successfully created data file: {filename}")
        return str(filename)

    except Exception as e:
        print(f"An error occurred while processing {ticker}: {e}")
        return None

# ==============================================================================
# HELPER FUNCTION: SIMULATION EXECUTION
# ==============================================================================

def run_single_simulation(csv_filepath, session_dirs, epoch, load_state_prefix=None):
    """
    Runs one instance of the C++ trading simulation and captures its output.
    Logs output to a file and returns a dictionary with parsed results.
    """
    ticker = Path(csv_filepath).stem.split('_')[0]
    # State prefixes are now global, not per-ticker
    save_state_prefix = session_dirs["temp_model_state"] / "temp_agent_state"

    command = [
        SIMULATION_EXECUTABLE,
        "--csv", csv_filepath,
        "--save", str(save_state_prefix)
    ]
    if load_state_prefix and os.path.exists(f"{load_state_prefix}_agent_state.json"):
        command.extend(["--load", str(load_state_prefix)])

    print(f"\nExecuting: {' '.join(command)}")

    run_data = { "ticker": ticker, "initial_value": 10000.0, "final_value": 0.0 }
    log_path = session_dirs["logs"] / f"epoch_{epoch}_{ticker}_sim.log"

    try:
        with open(log_path, 'w') as log_file:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                for line in proc.stdout:
                    print(line, end='')
                    log_file.write(line)

                    final_val_match = re.search(r"Total Value:\s+\$([\d\.]+)", line)
                    if final_val_match:
                        run_data["final_value"] = float(final_val_match.group(1))

        if proc.returncode != 0:
            print(f"Warning: Simulation exited with non-zero code {proc.returncode}.")

        print(f"Simulation log saved to: {log_path}")
        return run_data

    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")
        return None

# ==============================================================================
# HELPER FUNCTION: REPORTING
# ==============================================================================

def analyze_and_report(results, current_epoch, report_dir):
    """Analyzes aggregated results and generates a markdown report with charts."""
    if not results: return
    print(f"\n--- Analyzing {len(results)} runs and generating report for epoch {current_epoch} ---")
    df = pd.DataFrame(results)

    df["pnl"] = df["final_value"] - df["initial_value"]
    df["pnl_pct"] = (df["pnl"] / df["initial_value"].replace(0, np.nan)) * 100
    df.dropna(subset=['pnl_pct'], inplace=True)

    if df.empty:
        print("No valid results with P/L percentage to report."); return

    plt.style.use('seaborn-v0_8-darkgrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(df['pnl_pct'], kde=True, bins=15)
    plt.title(f'Distribution of Simulation Returns (Epochs 1-{current_epoch})', fontsize=16)
    plt.xlabel('Return (P/L %)')
    plt.ylabel('Frequency')
    dist_chart_path = report_dir / f"epoch_{current_epoch}_returns_distribution.png"
    plt.savefig(dist_chart_path)
    plt.close()

    plt.figure(figsize=(12, 7))
    ticker_perf = df.groupby('ticker')['pnl_pct'].mean().sort_values(ascending=False)
    sns.barplot(x=ticker_perf.index, y=ticker_perf.values)
    plt.title('Average Return (P/L %) by Stock Ticker', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ticker_chart_path = report_dir / f"epoch_{current_epoch}_performance_by_ticker.png"
    plt.savefig(ticker_chart_path)
    plt.close()

    report_path = report_dir / f"performance_report_epoch_{current_epoch}.md"
    with open(report_path, 'w') as f:
        f.write(f"# Trading Agent Performance Report (Epoch {current_epoch})\n\n")
        f.write(f"This report summarizes agent performance over {len(df)} simulation runs.\n\n")
        f.write("## Overall Performance\n")
        f.write(f"- **Average Return (P/L %):** `{df['pnl_pct'].mean():.2f}%`\n")
        f.write(f"- **Win Rate (profitable runs):** `{((df['pnl_pct'] > 0).sum() / len(df)) * 100:.1f}%`\n\n")
        f.write(f"![Returns Distribution]({dist_chart_path.name})\n\n")
        f.write(f"![Performance by Ticker]({ticker_chart_path.name})\n\n")

    print(f"Generated interim report: {report_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(args):
    """Main function to run the training and evaluation loop."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dirs = setup_session_directories(run_id)

    cleanup_generated_data(args.data_dir)

    all_results = []
    # Use a single float to track the best performance across all tickers
    best_performance = -999.0

    stock_tickers = DEFAULT_STOCK_TICKERS
    if args.ticker_file:
        try:
            with open(args.ticker_file, 'r') as f:
                stock_tickers = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(stock_tickers)} tickers from {args.ticker_file}.")
        except FileNotFoundError:
            print(f"Warning: Ticker file '{args.ticker_file}' not found. Using default list.")

    print("\nStarting evaluation harness...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*25} EPOCH {epoch}/{args.epochs} {'='*25}")

        target_ticker = random.choice(stock_tickers)
        csv_file = fetch_and_process_single_stock(target_ticker, session_dirs["data"])
        if not csv_file:
            print(f"Failed to get data for {target_ticker}, skipping epoch."); continue

        # The model state prefix is now global and not ticker-specific
        best_model_prefix = session_dirs["best_model_state"] / "best_agent_state"

        result = run_single_simulation(csv_file, session_dirs, epoch, load_state_prefix=best_model_prefix)
        if not result or result.get("initial_value", 0) <= 0:
            print("Epoch failed or produced invalid results."); continue
        all_results.append(result)

        pnl_pct = ((result["final_value"] - result["initial_value"]) / result["initial_value"]) * 100

        print(f"\nEpoch {epoch} complete. Ticker: {target_ticker}, P/L: {pnl_pct:.2f}% (Best overall: {best_performance:.2f}%)")

        if pnl_pct > best_performance:
            best_performance = pnl_pct
            print(f"New best overall performance! Promoting model.")
            # Promote the single global model
            promote_model(session_dirs["temp_model_state"], session_dirs["best_model_state"])
        else:
            print("Performance did not exceed the best overall. Discarding temp model.")

        if epoch % args.report_interval == 0 or epoch == args.epochs:
            analyze_and_report(all_results, epoch, session_dirs["reports"])

    print(f"\n{'='*20} Evaluation harness finished. {'='*20}")
    print(f"All artifacts for this run are located in: {session_dirs['root'].resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A continuous training and evaluation harness for the NeuroGen autonomous trading agent.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("stock_data_csv"),
        help="Base directory to store generated stock data CSVs.\n(default: %(default)s)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Total number of simulation cycles to run.\n(default: %(default)s)"
    )
    parser.add_argument(
        '--report-interval',
        type=int,
        default=10,
        help="Generate a full performance report every N epochs.\n(default: %(default)s)"
    )
    parser.add_argument(
        '--ticker-file',
        type=str,
        default=None,
        help="Path to a text file containing stock tickers to use, one per line.\n(default: Use internal hardcoded list)"
    )

    parsed_args = parser.parse_args()
    main(parsed_args)