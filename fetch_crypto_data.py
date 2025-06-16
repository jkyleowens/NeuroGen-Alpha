'''
Python script to fetch cryptocurrency OHLCV data using ccxt and save it to CSV files.
This data can be used by a C++ trading simulation.
'''
import ccxt
import csv
import datetime
import random
import time
import os

# --- Configuration ---
EXCHANGE_ID = 'kraken'  # Exchange to use (e.g., 'binance', 'coinbasepro', 'kraken')
NUM_COINS_TO_FETCH = 25    # Number of different cryptocurrencies to fetch data for
# Prefer pairs trading against these quote currencies
PREFERRED_QUOTE_CURRENCIES = ['USDT', 'USD', 'BUSD'] 
TIMEFRAME = '1d'          # Timeframe for OHLCV data (e.g., '1m', '5m', '1h', '1d')
# Options for the number of days of historical data to fetch
HISTORICAL_DAYS_OPTIONS = [90, 180, 270, 365]
OUTPUT_DIR = "crypto_data_csv" # Directory to save CSV files
API_CALL_DELAY_SECONDS = 10 # Delay between fetching data for different coins to respect rate limits

def initialize_exchange(exchange_id):
    '''Initializes and returns the ccxt exchange instance.'''
    try:
        exchange = getattr(ccxt, exchange_id)()
        print(f"Successfully initialized exchange: {exchange_id}")
        return exchange
    except AttributeError:
        print(f"Error: Exchange '{exchange_id}' not found in ccxt.")
        return None
    except Exception as e:
        print(f"Error initializing exchange '{exchange_id}': {e}")
        return None

def get_random_trading_pairs(exchange, num_pairs, quote_currencies):
    '''Fetches available markets and randomly selects trading pairs.'''
    if not exchange:
        return []
    try:
        print("Fetching available markets...")
        markets = exchange.load_markets()
        if not markets:
            print("Warning: Could not load markets.")
            return []

        # Filter for active spot markets and preferred quote currencies
        valid_pairs = []
        for symbol, market_info in markets.items():
            if market_info.get('active', True) and market_info.get('spot', True):
                if '/' in symbol:
                    base, quote = symbol.split('/')
                    if quote in quote_currencies:
                        valid_pairs.append(symbol)
        
        if not valid_pairs:
            print(f"Warning: No active spot trading pairs found for quote currencies: {quote_currencies}")
            return []

        if num_pairs > len(valid_pairs):
            print(f"Warning: Requested {num_pairs} pairs, but only {len(valid_pairs)} suitable pairs are available.")
            num_pairs = len(valid_pairs)
        
        return random.sample(valid_pairs, num_pairs)

    except ccxt.NetworkError as e:
        print(f"Network error fetching markets: {e}")
        return []
    except ccxt.ExchangeError as e:
        print(f"Exchange error fetching markets: {e}")
        return []
    except Exception as e:
        print(f"An error occurred while fetching trading pairs: {e}")
        return []

def fetch_ohlcv_data(exchange, symbol, timeframe, days_to_fetch):
    '''Fetches OHLCV data for a given symbol and timeframe.'''
    if not exchange or not exchange.has['fetchOHLCV']:
        print(f"Exchange '{exchange.id}' does not support fetching OHLCV data.")
        return []

    print(f"Fetching {timeframe} OHLCV data for '{symbol}' for the last {days_to_fetch} days...")
    try:
        # Calculate `since` timestamp (milliseconds ago)
        since = exchange.milliseconds() - (days_to_fetch * 24 * 60 * 60 * 1000)
        
        # Limit parameter can be used if needed, but fetching by `since` is often sufficient
        # Some exchanges might have a limit on the number of candles per request.
        # For simplicity, we fetch all data since the calculated start time in one go if possible.
        # If data is too large, pagination logic would be needed here.
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        
        if not ohlcv:
            print(f"No OHLCV data returned for '{symbol}'.")
            return []
        
        # Convert timestamp from ms to s for each entry
        # Data format: [timestamp_ms, open, high, low, close, volume]
        processed_ohlcv = []
        for candle in ohlcv:
            processed_ohlcv.append([
                candle[0] // 1000, # timestamp in seconds
                candle[1],         # open
                candle[2],         # high
                candle[3],         # low
                candle[4],         # close
                candle[5]          # volume
            ])
        return processed_ohlcv

    except ccxt.NetworkError as e:
        print(f"Network error fetching OHLCV for '{symbol}': {e}")
    except ccxt.ExchangeError as e:
        print(f"Exchange error fetching OHLCV for '{symbol}': {e}")
    except Exception as e:
        print(f"An error occurred fetching OHLCV for '{symbol}': {e}")
    return []

def save_to_csv(data, symbol, timeframe, output_dir):
    '''Saves the OHLCV data to a CSV file.'''
    if not data:
        print(f"No data to save for '{symbol}'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize symbol for filename (replace '/' with '_')
    filename_symbol = symbol.replace('/', '_')
    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_symbol}_{timeframe}_{now_str}.csv"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Header
            writer.writerows(data)
        print(f"Successfully saved data for '{symbol}' to '{filepath}'")
    except IOError as e:
        print(f"Error writing CSV file '{filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving CSV for '{symbol}': {e}")

def main():
    '''Main function to orchestrate fetching and saving data.'''
    print(f"Starting crypto data fetching process using ccxt (Exchange: {EXCHANGE_ID}).")
    
    exchange = initialize_exchange(EXCHANGE_ID)
    if not exchange:
        print("Exiting due to exchange initialization failure.")
        return

    selected_pairs = get_random_trading_pairs(exchange, NUM_COINS_TO_FETCH, PREFERRED_QUOTE_CURRENCIES)

    if not selected_pairs:
        print("Could not select any trading pairs. Exiting.")
        return

    print(f"Selected trading pairs for data fetching: {selected_pairs}")

    for pair_symbol in selected_pairs:
        days_to_fetch = random.choice(HISTORICAL_DAYS_OPTIONS)
        ohlcv_data = fetch_ohlcv_data(exchange, pair_symbol, TIMEFRAME, days_to_fetch)
        
        if ohlcv_data:
            save_to_csv(ohlcv_data, pair_symbol, TIMEFRAME, OUTPUT_DIR)
        else:
            print(f"Skipping CSV save for '{pair_symbol}' due to no data fetched.")
        
        print("---")
        # Respect API rate limits
        if len(selected_pairs) > 1 and selected_pairs.index(pair_symbol) < len(selected_pairs) - 1: # No need to sleep if only one pair or it's the last one
             print(f"Waiting for {API_CALL_DELAY_SECONDS} seconds before next API call...")
             time.sleep(API_CALL_DELAY_SECONDS)

    print("Crypto data fetching process completed.")

if __name__ == "__main__":
    main()
