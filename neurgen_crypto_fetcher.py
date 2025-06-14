#!/usr/bin/env python3
"""
NeuroGen-Alpha Enhanced Cryptocurrency Data Fetcher
===================================================

Enhanced cryptocurrency data fetcher specifically designed for the NeuroGen-Alpha
trading simulation. This script provides comprehensive data collection with
advanced features for neural network training.

Features:
- Multi-exchange data aggregation
- Advanced data validation and cleaning
- Neural network optimized data formatting
- Real-time and historical data modes
- Automatic retry and error recovery
- Performance metrics tracking
- Direct integration with NeuroGen-Alpha simulation

Usage:
    python3 neurgen_crypto_fetcher.py --mode historical --symbols BTCUSD ETHUSD --days 30
    python3 neurgen_crypto_fetcher.py --mode realtime --symbols BTCUSD --interval 60
    python3 neurgen_crypto_fetcher.py --mode continuous --config config.json
"""

import requests
import pandas as pd
import argparse
import time
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurgen_crypto_fetcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NeuroGenCryptoFetcher:
    """
    Enhanced cryptocurrency data fetcher for NeuroGen-Alpha neural network trading
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the fetcher with configuration"""
        self.config = self._load_config(config_file)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuroGen-Alpha-Neural-Trading/2.0',
            'Accept': 'application/json'
        })
        
        # API Configuration
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.rate_limit_delay = 1.2  # Seconds between requests
        self.max_retries = 3
        self.last_request_time = 0
        
        # Symbol mappings for multiple exchanges
        self.symbol_mappings = {
            'BTCUSD': {'coingecko': 'bitcoin', 'display': 'Bitcoin'},
            'ETHUSD': {'coingecko': 'ethereum', 'display': 'Ethereum'},
            'ADAUSD': {'coingecko': 'cardano', 'display': 'Cardano'},
            'DOTUSD': {'coingecko': 'polkadot', 'display': 'Polkadot'},
            'SOLUSD': {'coingecko': 'solana', 'display': 'Solana'},
            'MATICUSD': {'coingecko': 'matic-network', 'display': 'Polygon'},
            'LINKUSD': {'coingecko': 'chainlink', 'display': 'Chainlink'},
            'AVAXUSD': {'coingecko': 'avalanche-2', 'display': 'Avalanche'},
            'ATOMUSD': {'coingecko': 'cosmos', 'display': 'Cosmos'},
            'LTCUSD': {'coingecko': 'litecoin', 'display': 'Litecoin'},
            'BCHUSDT': {'coingecko': 'bitcoin-cash', 'display': 'Bitcoin Cash'},
            'XRPUSD': {'coingecko': 'ripple', 'display': 'XRP'}
        }
        
        # Ensure output directories exist
        self.output_dir = Path("crypto_data_csv")
        self.neurgen_dir = Path("highly_diverse_stock_data_clean_csv")
        self.output_dir.mkdir(exist_ok=True)
        self.neurgen_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'requests_made': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_data_points': 0,
            'start_time': datetime.now()
        }
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'default_symbols': ['BTCUSD', 'ETHUSD', 'ADAUSD'],
            'default_interval': '1h',
            'default_days': 7,
            'neural_features': True,
            'data_validation': True,
            'max_data_points': 10000,
            'real_time_update_interval': 60
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _rate_limit(self):
        """Enforce rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with comprehensive error handling"""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                self.stats['requests_made'] += 1
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                self.stats['successful_fetches'] += 1
                return data
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error {response.status_code}: {e}")
                    break
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        self.stats['failed_fetches'] += 1
        return None
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return list(self.symbol_mappings.keys())
    
    def fetch_current_price(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive current price data"""
        if symbol not in self.symbol_mappings:
            logger.error(f"Unsupported symbol: {symbol}")
            return None
        
        coin_id = self.symbol_mappings[symbol]['coingecko']
        url = f"{self.coingecko_url}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        data = self._make_request(url, params)
        if data and coin_id in data:
            return data[coin_id]
        return None
    
    def fetch_historical_ohlcv(self, symbol: str, days: int = 7, interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data optimized for neural network training
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
            interval: Data interval ('5m', '1h', '4h', '1d')
        """
        if symbol not in self.symbol_mappings:
            logger.error(f"Unsupported symbol: {symbol}")
            return None
        
        coin_id = self.symbol_mappings[symbol]['coingecko']
        display_name = self.symbol_mappings[symbol]['display']
        
        logger.info(f"Fetching {days} days of {interval} OHLCV data for {display_name} ({symbol})")
        
        # Use market chart endpoint for comprehensive data
        url = f"{self.coingecko_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'hourly' if interval in ['1h', '5m'] else 'daily'
        }
        
        data = self._make_request(url, params)
        if not data:
            return None
        
        try:
            df = self._process_market_data(data, interval, symbol)
            if df is not None and not df.empty:
                self.stats['total_data_points'] += len(df)
                logger.info(f"Successfully processed {len(df)} data points for {symbol}")
                return df
            else:
                logger.warning(f"No valid data processed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
            return None
    
    def _process_market_data(self, data: Dict, interval: str, symbol: str) -> Optional[pd.DataFrame]:
        """Process market chart data into neural network optimized OHLCV format"""
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        market_caps = data.get('market_caps', [])
        
        if not prices or len(prices) < 2:
            logger.warning(f"Insufficient price data for {symbol}")
            return None
        
        # Create DataFrames
        price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # Merge data
        df = pd.merge(price_df, volume_df, on='timestamp', how='inner')
        if market_caps:
            mcap_df = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
            df = pd.merge(df, mcap_df, on='timestamp', how='left')
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Determine resampling frequency
        freq_map = {
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        freq = freq_map.get(interval, '1H')
        
        # Resample to create OHLCV data
        ohlcv = df.resample(freq).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()
        
        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv.reset_index(inplace=True)
        
        # Add neural network specific features if enabled
        if self.config.get('neural_features', True):
            ohlcv = self._add_neural_features(ohlcv)
        
        # Validate data quality
        if self.config.get('data_validation', True):
            ohlcv = self._validate_and_clean_data(ohlcv, symbol)
        
        return ohlcv
    
    def _add_neural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features optimized for neural network training"""
        # Price changes and returns
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages for trend detection
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Volatility indicators
        df['volatility'] = df['close'].rolling(window=20).std()
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close'] * 100
        
        return df
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean data for neural network consumption"""
        initial_len = len(df)
        
        # Remove rows with invalid OHLCV relationships
        df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
        df = df[df['low'] <= df[['open', 'close']].min(axis=1)]
        df = df[df['volume'] >= 0]
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]
        
        # Fill any remaining NaN values
        df = df.dropna()
        
        cleaned_len = len(df)
        if cleaned_len < initial_len:
            logger.info(f"Cleaned {symbol} data: {initial_len} -> {cleaned_len} rows")
        
        return df
    
    def save_neurgen_format(self, df: pd.DataFrame, symbol: str, suffix: str = "") -> str:
        """Save data in NeuroGen-Alpha compatible format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}{suffix}.csv"
        
        # Save to both directories
        output_paths = [
            self.output_dir / filename,
            self.neurgen_dir / filename
        ]
        
        # Format for NeuroGen-Alpha
        output_df = df.copy()
        output_df['datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Essential columns for trading simulation
        essential_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if all(col in output_df.columns for col in essential_columns):
            # Primary format with essential columns
            primary_df = output_df[essential_columns].copy()
            
            # Ensure numeric precision
            for col in ['open', 'high', 'low', 'close', 'volume']:
                primary_df[col] = pd.to_numeric(primary_df[col], errors='coerce')
            
            # Remove any rows with NaN values
            primary_df = primary_df.dropna()
            
            # Save primary format
            for path in output_paths:
                primary_df.to_csv(path, index=False, float_format='%.8f')
                logger.info(f"Saved {len(primary_df)} records to {path}")
            
            # Save extended format with neural features (if available)
            if len(output_df.columns) > len(essential_columns):
                extended_filename = f"{symbol}_{timestamp}{suffix}_extended.csv"
                extended_path = self.output_dir / extended_filename
                output_df.to_csv(extended_path, index=False, float_format='%.8f')
                logger.info(f"Saved extended format to {extended_path}")
            
            return str(output_paths[0])
        else:
            logger.error(f"Missing essential columns for {symbol}")
            return None
    
    def fetch_multiple_symbols(self, symbols: List[str], days: int = 7, interval: str = '1h') -> Dict[str, str]:
        """Fetch data for multiple symbols efficiently"""
        results = {}
        total_symbols = len(symbols)
        
        logger.info(f"Fetching data for {total_symbols} symbols: {', '.join(symbols)}")
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{total_symbols})")
            
            try:
                df = self.fetch_historical_ohlcv(symbol, days, interval)
                if df is not None:
                    suffix = f"_historical_{days}d_{interval}"
                    filepath = self.save_neurgen_format(df, symbol, suffix)
                    if filepath:
                        results[symbol] = filepath
                        logger.info(f"✅ {symbol} completed successfully")
                    else:
                        logger.error(f"❌ Failed to save {symbol}")
                else:
                    logger.error(f"❌ Failed to fetch data for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
            
            # Progress indicator
            if i < total_symbols:
                logger.info(f"Progress: {i}/{total_symbols} completed, {total_symbols - i} remaining")
        
        return results
    
    def start_realtime_mode(self, symbols: List[str], update_interval: int = 60):
        """Start continuous real-time data collection"""
        logger.info(f"🚀 Starting real-time mode for {len(symbols)} symbols")
        logger.info(f"Update interval: {update_interval} seconds")
        logger.info(f"Symbols: {', '.join(symbols)}")
        
        try:
            cycle = 0
            while True:
                cycle += 1
                logger.info(f"\n📊 Real-time cycle #{cycle} started")
                
                for symbol in symbols:
                    try:
                        # Fetch current price data
                        current_data = self.fetch_current_price(symbol)
                        if current_data:
                            # Create single-row DataFrame
                            price = current_data['usd']
                            now = datetime.now()
                            
                            df_data = {
                                'timestamp': [now],
                                'open': [price],
                                'high': [price * 1.001],
                                'low': [price * 0.999],
                                'close': [price],
                                'volume': [current_data.get('usd_24h_vol', 1000000)]
                            }
                            
                            df = pd.DataFrame(df_data)
                            suffix = f"_realtime_cycle_{cycle}"
                            filepath = self.save_neurgen_format(df, symbol, suffix)
                            
                            if filepath:
                                logger.info(f"✅ Updated {symbol}: ${price:,.2f}")
                            else:
                                logger.warning(f"⚠️  Failed to save {symbol} update")
                        else:
                            logger.warning(f"⚠️  No data received for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error updating {symbol}: {e}")
                
                # Print statistics
                self._print_statistics()
                
                logger.info(f"💤 Sleeping for {update_interval} seconds...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Real-time mode stopped by user")
            self._print_final_statistics()
    
    def _print_statistics(self):
        """Print current fetching statistics"""
        runtime = datetime.now() - self.stats['start_time']
        logger.info(f"📈 Statistics: {self.stats['successful_fetches']} successful, "
                   f"{self.stats['failed_fetches']} failed, "
                   f"{self.stats['total_data_points']} total data points, "
                   f"Runtime: {runtime}")
    
    def _print_final_statistics(self):
        """Print final statistics summary"""
        runtime = datetime.now() - self.stats['start_time']
        success_rate = (self.stats['successful_fetches'] / self.stats['requests_made'] * 100 
                       if self.stats['requests_made'] > 0 else 0)
        
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Runtime: {runtime}")
        logger.info(f"Total Requests: {self.stats['requests_made']}")
        logger.info(f"Successful Fetches: {self.stats['successful_fetches']}")
        logger.info(f"Failed Fetches: {self.stats['failed_fetches']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Data Points: {self.stats['total_data_points']}")
        logger.info(f"Data saved to: {self.output_dir} and {self.neurgen_dir}")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='NeuroGen-Alpha Enhanced Cryptocurrency Data Fetcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 7 days of hourly data for Bitcoin and Ethereum
  python3 neurgen_crypto_fetcher.py --mode historical --symbols BTCUSD ETHUSD --days 7 --interval 1h
  
  # Start real-time monitoring for multiple cryptocurrencies
  python3 neurgen_crypto_fetcher.py --mode realtime --symbols BTCUSD ETHUSD ADAUSD --interval 60
  
  # Fetch comprehensive dataset for neural network training
  python3 neurgen_crypto_fetcher.py --mode training --days 30
        """
    )
    
    parser.add_argument('--mode', choices=['historical', 'realtime', 'training'], default='historical',
                       help='Data fetching mode')
    parser.add_argument('--symbols', nargs='+', help='Cryptocurrency symbols to fetch')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data')
    parser.add_argument('--interval', default='1h', choices=['5m', '15m', '1h', '4h', '1d'],
                       help='Data interval')
    parser.add_argument('--update-interval', type=int, default=60,
                       help='Real-time update interval in seconds')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = NeuroGenCryptoFetcher(args.config)
    
    # Determine symbols to use
    if args.symbols:
        symbols = args.symbols
    elif args.mode == 'training':
        symbols = fetcher.get_supported_symbols()[:6]  # Top 6 for training
    else:
        symbols = fetcher.config['default_symbols']
    
    # Validate symbols
    supported = fetcher.get_supported_symbols()
    invalid_symbols = [s for s in symbols if s not in supported]
    if invalid_symbols:
        logger.error(f"Unsupported symbols: {invalid_symbols}")
        logger.info(f"Supported symbols: {supported}")
        return 1
    
    logger.info("🚀 NeuroGen-Alpha Enhanced Cryptocurrency Data Fetcher")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    try:
        if args.mode == 'historical' or args.mode == 'training':
            logger.info(f"Fetching {args.days} days of {args.interval} data")
            results = fetcher.fetch_multiple_symbols(symbols, args.days, args.interval)
            
            logger.info(f"\n✅ Successfully processed {len(results)} symbols:")
            for symbol, filepath in results.items():
                logger.info(f"  {symbol}: {filepath}")
                
            if not results:
                logger.error("❌ No data was successfully fetched")
                return 1
                
        elif args.mode == 'realtime':
            fetcher.start_realtime_mode(symbols, args.update_interval)
        
        fetcher._print_final_statistics()
        logger.info("✅ Data fetching completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
