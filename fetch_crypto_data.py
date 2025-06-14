#!/usr/bin/env python3
"""
NeuroGen-Alpha Cryptocurrency Data Fetcher
===========================================

This script fetches real-time and historical cryptocurrency price data from multiple
exchanges and converts it to CSV format for the NeuroGen-Alpha trading simulation.

Features:
- Fetches data from CoinGecko (free, no API key required)
- Supports multiple cryptocurrency pairs
- Historical data with configurable intervals
- Real-time price updates
- CSV export compatible with trading simulation
- Error handling and retry logic
- Rate limiting to respect API limits

Usage:
    python3 fetch_crypto_data.py --symbol BTCUSD --days 30 --interval 1h
    python3 fetch_crypto_data.py --symbol ETHUSD --days 7 --interval 5m --realtime
"""

import requests
import pandas as pd
import argparse
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """
    Fetches cryptocurrency data from CoinGecko API and exports to CSV
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuroGen-Alpha-Trading-Bot/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.1  # Seconds between requests (respecting API limits)
        
        # Symbol mappings (symbol -> CoinGecko ID)
        self.symbol_mappings = {
            'BTCUSD': 'bitcoin',
            'ETHUSD': 'ethereum',
            'ADAUSD': 'cardano',
            'DOTUSD': 'polkadot',
            'SOLUSD': 'solana',
            'MATICUSD': 'matic-network',
            'LINKUSD': 'chainlink',
            'AVAXUSD': 'avalanche-2',
            'ATOMUSD': 'cosmos',
            'ALGOUSD': 'algorand'
        }
        
        # Ensure output directory exists
        self.output_dir = "crypto_data_csv"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic"""
        for attempt in range(retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data after {retries} attempts")
                    return None
    
    def get_coin_id(self, symbol: str) -> Optional[str]:
        """Convert trading symbol to CoinGecko coin ID"""
        # Remove USD suffix for lookup
        clean_symbol = symbol.replace('USD', '').upper() + 'USD'
        return self.symbol_mappings.get(clean_symbol)
    
    def fetch_current_price(self, symbol: str) -> Optional[Dict]:
        """Fetch current price data for a cryptocurrency"""
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            logger.error(f"Unsupported symbol: {symbol}")
            return None
        
        url = f"{self.base_url}/simple/price"
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
    
    def fetch_historical_data(self, symbol: str, days: int = 30, interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSD')
            days: Number of days of historical data
            interval: Data interval ('5m', '1h', '1d')
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            logger.error(f"Unsupported symbol: {symbol}")
            return None
        
        logger.info(f"Fetching {days} days of {interval} data for {symbol}")
        
        # For detailed OHLCV data, we need to use the OHLC endpoint
        if days <= 1:
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': days}
        else:
            # For longer periods, use market chart data and simulate OHLCV
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if interval in ['1h', '5m'] else 'daily'
            }
        
        data = self._make_request(url, params)
        if not data:
            return None
        
        try:
            if 'prices' in data:
                # Market chart data - convert to OHLCV format
                df = self._process_market_chart_data(data, interval)
            else:
                # OHLC data
                df = self._process_ohlc_data(data)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {len(df)} data points")
                return df
            else:
                logger.warning("No data received or data is empty")
                return None
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _process_market_chart_data(self, data: Dict, interval: str) -> pd.DataFrame:
        """Process market chart data into OHLCV format"""
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            return None
        
        # Convert to DataFrame
        price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # Merge price and volume data
        df = pd.merge(price_df, volume_df, on='timestamp', how='left')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Resample to create OHLCV data based on interval
        if interval == '5m':
            freq = '5T'
        elif interval == '1h':
            freq = '1H'
        elif interval == '1d':
            freq = '1D'
        else:
            freq = '1H'  # Default
        
        df.set_index('timestamp', inplace=True)
        
        # Create OHLCV data
        ohlcv = df.resample(freq).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()
        
        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv.reset_index(inplace=True)
        
        return ohlcv
    
    def _process_ohlc_data(self, data: List) -> pd.DataFrame:
        """Process OHLC data from CoinGecko"""
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add synthetic volume data (since OHLC endpoint doesn't provide volume)
        df['volume'] = df['close'] * 1000  # Synthetic volume based on price
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, suffix: str = "") -> str:
        """Save DataFrame to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}{suffix}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Format DataFrame for trading simulation compatibility
        output_df = df.copy()
        output_df['datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns to match expected format
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        output_df = output_df[columns]
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
        
        # Remove any rows with NaN values
        output_df = output_df.dropna()
        
        # Save to CSV
        output_df.to_csv(filepath, index=False, float_format='%.8f')
        logger.info(f"Saved {len(output_df)} records to {filepath}")
        
        return filepath
    
    def fetch_and_save_historical(self, symbol: str, days: int = 30, interval: str = '1h') -> Optional[str]:
        """Fetch historical data and save to CSV"""
        df = self.fetch_historical_data(symbol, days, interval)
        if df is not None:
            suffix = f"_historical_{days}d_{interval}"
            return self.save_to_csv(df, symbol, suffix)
        return None
    
    def fetch_and_save_realtime(self, symbol: str) -> Optional[str]:
        """Fetch current price and save as single-row CSV"""
        current_data = self.fetch_current_price(symbol)
        if not current_data:
            return None
        
        # Create a single-row DataFrame with current data
        now = datetime.now()
        price = current_data['usd']
        
        # Since we only have current price, we'll use it for all OHLC values
        df = pd.DataFrame([{
            'timestamp': now,
            'open': price,
            'high': price * 1.001,  # Slight variation for high
            'low': price * 0.999,   # Slight variation for low
            'close': price,
            'volume': current_data.get('usd_24h_vol', 1000000)  # Use 24h volume or default
        }])
        
        suffix = "_realtime"
        return self.save_to_csv(df, symbol, suffix)
    
    def start_realtime_monitoring(self, symbols: List[str], update_interval: int = 60):
        """Start real-time monitoring and CSV updates"""
        logger.info(f"Starting real-time monitoring for {symbols}")
        logger.info(f"Update interval: {update_interval} seconds")
        
        try:
            while True:
                for symbol in symbols:
                    try:
                        filepath = self.fetch_and_save_realtime(symbol)
                        if filepath:
                            logger.info(f"Updated {symbol} data: {filepath}")
                        else:
                            logger.warning(f"Failed to update {symbol}")
                    except Exception as e:
                        logger.error(f"Error updating {symbol}: {e}")
                
                logger.info(f"Sleeping for {update_interval} seconds...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("Real-time monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency data for NeuroGen-Alpha trading')
    parser.add_argument('--symbol', default='BTCUSD', help='Trading pair symbol (default: BTCUSD)')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data (default: 7)')
    parser.add_argument('--interval', default='1h', choices=['5m', '1h', '1d'], 
                       help='Data interval (default: 1h)')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time monitoring')
    parser.add_argument('--update-interval', type=int, default=60, 
                       help='Real-time update interval in seconds (default: 60)')
    parser.add_argument('--multiple-symbols', nargs='+', 
                       help='Fetch data for multiple symbols')
    
    args = parser.parse_args()
    
    fetcher = CryptoDataFetcher()
    
    # Determine which symbols to process
    symbols = args.multiple_symbols if args.multiple_symbols else [args.symbol]
    
    logger.info("🚀 NeuroGen-Alpha Cryptocurrency Data Fetcher")
    logger.info("=" * 50)
    
    if args.realtime:
        # Real-time monitoring mode
        logger.info("Starting real-time monitoring mode...")
        fetcher.start_realtime_monitoring(symbols, args.update_interval)
    else:
        # Historical data mode
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}")
            logger.info(f"Period: {args.days} days, Interval: {args.interval}")
            
            filepath = fetcher.fetch_and_save_historical(symbol, args.days, args.interval)
            if filepath:
                logger.info(f"✅ Successfully saved data to {filepath}")
                
                # Also fetch current price for immediate use
                realtime_file = fetcher.fetch_and_save_realtime(symbol)
                if realtime_file:
                    logger.info(f"✅ Current price saved to {realtime_file}")
            else:
                logger.error(f"❌ Failed to fetch data for {symbol}")
    
    logger.info("Data fetching completed!")
    logger.info(f"CSV files saved in: {fetcher.output_dir}/")

if __name__ == "__main__":
    main()
