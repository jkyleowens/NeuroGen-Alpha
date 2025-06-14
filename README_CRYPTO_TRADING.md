# NeuroGen-Alpha Cryptocurrency Trading System

## 🚀 Overview

NeuroGen-Alpha is an autonomous cryptocurrency trading system that combines:
- **CUDA-accelerated neural networks** for decision making
- **Real-time Kraken API integration** for live market data
- **Advanced technical analysis** with 20+ indicators
- **Comprehensive risk management** and portfolio optimization
- **Real-time monitoring dashboard** with live visualizations

## 📋 Prerequisites

### System Requirements
- **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
- **Linux** (Ubuntu 20.04+ recommended)
- **CUDA Toolkit** 11.0 or higher
- **C++17** compatible compiler (GCC 9+)

### Dependencies
Run the installation script to install all required dependencies:
```bash
./install_dependencies.sh
```

Or install manually:
```bash
# Essential libraries
sudo apt-get install -y libcurl4-openssl-dev libjsoncpp-dev build-essential cmake

# CUDA Toolkit (if not installed)
# Download from: https://developer.nvidia.com/cuda-downloads
```

## 🔧 Building the System

1. **Clone and setup:**
```bash
git clone <repository-url>
cd NeuroGen-Alpha
```

2. **Build the project:**
```bash
make clean
make
```

3. **Verify build:**
```bash
ls -la bin/NeuroGen-Alpha
```

## 🚀 Running the Trading System

### Basic Usage
```bash
# Run with default settings (BTCUSD, 60-second intervals)
./bin/NeuroGen-Alpha

# Specify trading pair
./bin/NeuroGen-Alpha ETHUSD

# Specify trading pair and update interval (seconds)
./bin/NeuroGen-Alpha BTCUSD 30
```

### Supported Trading Pairs
- **BTCUSD** - Bitcoin to USD
- **ETHUSD** - Ethereum to USD  
- **LTCUSD** - Litecoin to USD
- **ADAUSD** - Cardano to USD
- **DOTUSD** - Polkadot to USD
- **SOLUSD** - Solana to USD
- **MATICUSD** - Polygon to USD

### Example Commands
```bash
# Bitcoin trading with 30-second updates
./bin/NeuroGen-Alpha BTCUSD 30

# Ethereum trading with 2-minute updates  
./bin/NeuroGen-Alpha ETHUSD 120

# Solana trading with default 1-minute updates
./bin/NeuroGen-Alpha SOLUSD
```

## 📊 System Architecture

### Core Components

1. **KrakenDataFetcher**
   - Real-time API integration with Kraken exchange
   - Rate limiting and error handling
   - OHLCV data parsing and validation

2. **TradingAgent**
   - Neural network decision engine
   - Advanced feature engineering (20+ technical indicators)
   - Portfolio management and risk control
   - Performance tracking and analytics

3. **CryptoTradingSimulator**
   - Real-time trading loop coordination
   - Market data processing pipeline
   - Autonomous decision execution
   - Performance monitoring

4. **RealTimeMonitor**
   - Live system status display
   - Performance metrics tracking
   - Real-time dashboard generation

### Neural Network Features
- **CUDA-accelerated** spiking neural network
- **Reward-based learning** from trading performance
- **Synaptic plasticity** for adaptive behavior
- **Neuromodulator systems** (dopamine, acetylcholine)

### Technical Analysis
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** position analysis
- **Moving Averages** (5, 10, 20, 50 periods)
- **Volatility** calculations
- **Volume indicators** (OBV, VWAP)
- **Momentum** analysis (multiple timeframes)
- **Stochastic oscillator**

## 📈 Monitoring and Analytics

### Real-Time Dashboard
The system generates live HTML dashboards at:
- `NeuroGen_Crypto_Dashboard_Live.html` (updated every 5 minutes)
- `NeuroGen_Crypto_Dashboard_Final.html` (generated on exit)

### Log Files
- **Trading Log**: `crypto_trading_log_<PAIR>.csv`
- **Performance Report**: `crypto_performance_report_<PAIR>.txt`
- **Metrics**: `crypto_trading_metrics_<PAIR>.csv`

### Dashboard Features
- 🕒 **Real-time price tracking**
- 📊 **Portfolio performance graphs**
- 📈 **Volume analysis**
- 🧠 **Neural network confidence levels**
- 💰 **Profit/Loss visualization**
- ⚡ **Auto-refresh every 30 seconds**

## ⚙️ Configuration

### Risk Management
Default settings (can be modified in code):
- **Max position size**: 50% of capital
- **Max risk per trade**: 2% of capital
- **Minimum confidence**: 60% for trade execution
- **Volatility threshold**: 50% maximum for trading

### Neural Network
- **Learning rate**: Adaptive based on performance
- **Reward decay**: 95% factor for temporal credit assignment
- **Confidence threshold**: 60% minimum for decisions
- **Feature dimensions**: 64 engineered features per decision

## 🛡️ Safety Features

### Error Handling
- **API failure recovery** with exponential backoff
- **Network disconnection** handling
- **Invalid data** filtering and validation
- **Memory leak prevention** with RAII patterns

### Risk Controls
- **Position size limits** to prevent overexposure
- **Volatility-based** position sizing
- **Confidence-based** trade filtering
- **Maximum drawdown** monitoring

### Graceful Shutdown
- **Ctrl+C handling** for clean exit
- **Automatic report generation** on shutdown
- **Resource cleanup** (CUDA, network connections)
- **Data persistence** for session recovery

## 📋 System Output

### Console Output
```
🚀 NeuroGen-Alpha Cryptocurrency Trading System 🚀
======================================================
Neural Network + Real-Time Crypto Trading
======================================================
[CONFIG] Trading pair set to: BTCUSD
[SYSTEM] Initializing cryptocurrency trading simulator...
[SIMULATOR] Fetching initial market data...
[SIMULATOR] Loaded 50 historical data points
[SYSTEM] Starting real-time trading simulation...

[CYCLE 1] Fetching market data...
[CYCLE 1] BTCUSD Price: $43,250.50 (Volume: 125.34)
[DECISION] Action: BUY (Confidence: 0.742)
[TRADE] BUY 0.5 @ $43,250.50 (Confidence: 0.742)
```

### Performance Metrics
```
=== Trading Agent Performance Report ===
Symbol: BTCUSD
Total Return: 2.34%
Sharpe Ratio: 1.42
Maximum Drawdown: 0.85%
Win Rate: 65.2%
Total Trades: 23
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Install CUDA Toolkit
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   ```

2. **Missing dependencies**
   ```bash
   ./install_dependencies.sh
   ```

3. **API connection errors**
   - Check internet connection
   - Verify Kraken API status
   - Reduce update frequency if rate limited

4. **Build errors**
   ```bash
   make clean
   make setup-headers
   make
   ```

## 📊 Performance Optimization

### For High-Frequency Trading
- Reduce update interval to 30 seconds or less
- Increase neural network learning rate
- Use lower confidence thresholds for faster execution

### For Conservative Trading
- Increase update interval to 5+ minutes
- Raise confidence thresholds to 70%+
- Reduce maximum position sizes

## 🤝 Contributing

### Development Setup
```bash
# Enable debug mode
make clean
CXXFLAGS="-DDEBUG -g" make

# Run with verbose output
./bin/NeuroGen-Alpha BTCUSD 60 --verbose
```

### Code Structure
- `src/main.cpp` - Main trading simulation
- `src/TradingAgent.cpp` - Trading agent implementation
- `include/NeuroGen/TradingAgent.h` - Trading agent interface
- `src/cuda/` - CUDA neural network kernels

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.**

---

*Built with ❤️ by the NeuroGen-Alpha team*
