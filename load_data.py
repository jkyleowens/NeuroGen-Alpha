# load_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time
import os
import traceback # For more detailed error messages

def download_stock_data_for_simulation(ticker, date_str, start_time_str, end_time_str, interval='5m', output_dir='stock_data_for_sim'):
    """
    Downloads yfinance stock time-series data for a specific ticker and time window on a given date,
    at a specified interval, and saves it to a CSV file with a clean, single-line header.
    """
    try:
        start_datetime = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M:%S")

        days_diff = (datetime.now() - start_datetime).days
        if interval == '1m' and days_diff > 6:
            print(f"INFO: Data for interval '1m' for {ticker} may not be available for dates older than 7 days (requested: {days_diff} days old).")
        elif interval != '1m' and days_diff > 59:
            print(f"INFO: Data for interval '{interval}' for {ticker} may not be available for dates older than 60 days (requested: {days_diff} days old).")

        print(f"Attempting to download: {ticker} from {start_datetime} to {end_datetime} (Interval: {interval})")

        data = yf.download(
            tickers=ticker,
            start=start_datetime,
            end=end_datetime,
            interval=interval,
            progress=False,
            auto_adjust=True # Provides OHLCV adjusted for splits/dividends
        )

        if data.empty:
            print(f"WARNING: No data downloaded for {ticker} for the window {start_datetime} to {end_datetime}.")
            print("  Possible reasons: Market closed (weekend/holiday), time outside trading hours,")
            print("  date too far past for interval, invalid ticker, or no trades.")
            return False # Indicate failure

        # --- Sanity check and diagnostic prints before saving ---
        print(f"  Data downloaded for {ticker}. Shape: {data.shape}")
        if not data.empty:
            print(f"  DataFrame columns: {data.columns.tolist()}") # Should be ['Open', 'High', 'Low', 'Close', 'Volume']
            print(f"  DataFrame index name: {data.index.name}") # Should be 'Datetime' or None then set to Datetime
            # Ensure 'Datetime' is the index name, yfinance usually sets this.
            if data.index.name is None or data.index.name.lower() != 'datetime':
                # If yfinance changes behavior or if index is reset, ensure 'Datetime' is the name for CSV output.
                # This script relies on yfinance setting 'Datetime' as the index.
                # If it's a column (e.g. after reset_index()), we'd save differently.
                # For now, assume yfinance default where Datetime is the index.
                pass # Keep it simple, rely on yf.download default behavior.


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")

        start_time_fn = start_time_str.replace(":", "")
        end_time_fn = end_time_str.replace(":", "")
        filename = f"{output_dir}/{ticker}_{date_str}__{start_time_fn}_{end_time_fn}_{interval}.csv"

        # Critical part: Saving to CSV
        # header=True: Writes the column names as the first line.
        # index=True: Writes the DataFrame index (which is 'Datetime' from yfinance) as the first column.
        # This should result in a CSV starting with:
        # Datetime,Open,High,Low,Close,Volume (if index is named 'Datetime')
        # or ,Open,High,Low,Close,Volume (if index is unnamed, first col is index vals)
        # The C++ parser expects the first field to be the datetime string.
        data.to_csv(filename, header=True, index=True)
        print(f"  SUCCESS: Data for {ticker} saved to {filename} ({len(data)} rows)")
        return True # Indicate success

    except Exception as e:
        print(f"ERROR: An exception occurred while processing {ticker} for date {date_str}:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        traceback.print_exc()
        return False # Indicate failure

def generate_simulation_data_over_period(tickers_list, start_date_str, end_date_str, hours_per_simulation_step=3, interval='5m', output_dir='stock_data_for_sim', market_open_time='09:30:00', market_close_time='16:00:00'):
    """
    Generates multiple CSV files for simulation for a list of tickers.
    """
    current_date_loop_start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_loop = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    m_open = time.fromisoformat(market_open_time)
    m_close = time.fromisoformat(market_close_time)

    print(f"\nStarting data generation for {len(tickers_list)} tickers...")
    print(f"Period: {start_date_str} to {end_date_str}")
    print(f"Each simulation step file will cover approx. {hours_per_simulation_step} hours.")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    successful_files = 0
    failed_attempts = 0

    for ticker in tickers_list:
        print(f"\nProcessing Ticker: {ticker}")
        current_date = current_date_loop_start
        while current_date <= end_date_loop:
            date_str = current_date.strftime("%Y-%m-%d")

            if current_date.weekday() >= 5: # Skip weekends
                current_date += timedelta(days=1)
                continue
            
            # print(f"  Date: {date_str}") # Can be verbose
            
            current_chunk_start_dt = datetime.combine(current_date, m_open)
            market_close_dt_for_day = datetime.combine(current_date, m_close)
            
            chunks_this_day = 0
            while current_chunk_start_dt < market_close_dt_for_day:
                chunk_end_dt = current_chunk_start_dt + timedelta(hours=hours_per_simulation_step)
                if chunk_end_dt > market_close_dt_for_day:
                    chunk_end_dt = market_close_dt_for_day
                if chunk_end_dt <= current_chunk_start_dt:
                     break

                start_time_str = current_chunk_start_dt.strftime("%H:%M:%S")
                end_time_str = chunk_end_dt.strftime("%H:%M:%S")
                
                if download_stock_data_for_simulation(
                    ticker=ticker, date_str=date_str, start_time_str=start_time_str,
                    end_time_str=end_time_str, interval=interval, output_dir=output_dir
                ):
                    successful_files += 1
                else:
                    failed_attempts += 1
                
                current_chunk_start_dt = chunk_end_dt
                chunks_this_day +=1
            
            # if chunks_this_day == 0 and current_date.weekday() < 5: # Optional: Log if no chunks on a weekday
            #     print(f"  No trading chunks generated for {ticker} on {date_str}.")
            current_date += timedelta(days=1)
        print(f"Finished Ticker: {ticker}")
    
    print(f"\nData generation summary: {successful_files} CSV files created, {failed_attempts} download attempts failed/yielded no data.")

# --- Main Execution Block ---
if __name__ == "__main__":
    highly_diverse_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'JNJ', 'PFE',
        'PG', 'MCD', 'XOM', 'F', 'AMC', 'PLTR', 'U', 'NEE', 'SPY', 'QQQ', 'IWM',
        'GLD', 'SLV', 'USO', 'EEM', 'ARKK', 'BND'
    ]
    # Current date: May 31, 2025 (Saturday)
    # yfinance 5m data is available for the last 60 days (approx. since April 2, 2025)
    # US Memorial Day: Monday, May 26, 2025 (Market Closed)
    
    start_date_for_run = "2025-05-27"  # Tuesday
    end_date_for_run = "2025-05-30"    # Friday (Day before current date)

    # It's CRITICAL that the C++ program reads from this EXACT directory.
    output_directory_name = 'highly_diverse_stock_data' 
    # If old files exist, clear them first for a clean test:
    # import shutil
    # if os.path.exists(output_directory_name):
    #     shutil.rmtree(output_directory_name)
    # print(f"Cleared directory: {output_directory_name} (if it existed)")


    print(f"--- Python Data Generation Script: load_data.py ---")
    print(f"Tickers: {len(highly_diverse_tickers)} (e.g., {', '.join(highly_diverse_tickers[:3])}...)")
    print(f"Period: {start_date_for_run} to {end_date_for_run}")
    print(f"Outputting to: ./{output_directory_name}/") # Relative path
    print("----------------------------------------------------")

    generate_simulation_data_over_period(
        tickers_list=highly_diverse_tickers,
        start_date_str=start_date_for_run,
        end_date_str=end_date_for_run,
        hours_per_simulation_step=3, # Approx. duration of data in each file
        interval='5m',
        output_dir=output_directory_name,
        market_open_time='09:30:00',
        market_close_time='16:00:00'
    )

    print("\n--- Python Data Generation Script Finished ---")
    final_output_path = os.path.abspath(output_directory_name)
    print(f"Verify CSV files in: {final_output_path}")
    print("Expected CSV format: Single header line 'Datetime,Open,High,Low,Close,Volume', then data rows.")