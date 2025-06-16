# Autonomous Trading Agent

This implementation provides an autonomous trading agent that interfaces with a biologically inspired neural network to perform cryptocurrency trading within a simulation environment. The agent is designed to learn from market data, make trading decisions, and adapt its strategy based on the outcomes of those decisions.

## Architecture

The system follows the architecture outlined in the design document, with the following key components:

### Core Components

1. **PriceTick**: A structure that holds market data for a single point in time (OHLCV data).

2. **KrakenApi**: Interfaces with the Kraken cryptocurrency exchange API to fetch historical market data.

3. **TechnicalAnalysis**: Processes raw price data and computes technical indicators that serve as features for the neural network.

4. **NeuralNetworkInterface**: Acts as the communication bridge to the biological neural network, sending features and receiving predictions.

5. **Portfolio**: Manages the agent's assets, tracks value, and executes trades.

6. **AutonomousTradingAgent**: The "brain" of the operation, orchestrating the decision-making process based on neural network predictions.

7. **Simulation**: Manages the overall simulation, data flow, and the main event loop.

### Data Flow

1. The Simulation fetches historical market data using the KrakenApi.
2. For each time step, the TradingAgent uses TechnicalAnalysis to generate features.
3. These features are sent to the NeuralNetworkInterface, which queries the BNN for a prediction.
4. Based on the prediction, the TradingAgent makes a decision (BUY, SELL, or HOLD).
5. The Simulation evaluates the outcome and sends a reward signal back through the agent to the BNN.
6. The BNN adapts its weights based on the reward, improving future predictions.

## Building the Project

To build the autonomous trading agent:

```bash
make -f Makefile.autonomous_trading
```

This will compile all the necessary components and create the `autonomous_trading` executable.

## Running the Simulation

The autonomous trading agent can be run with various command-line options:

```bash
./autonomous_trading [OPTIONS]
```

### Options

- `--pair SYMBOL`: Trading pair (default: BTCUSD)
- `--interval INTERVAL`: Time interval (default: 1h)
- `--ticks NUM`: Number of ticks to process (default: all)
- `--load FILE`: Load simulation state from file
- `--save FILE`: Save simulation state to file
- `--help`: Display help message

### Examples

Run with default settings:
```bash
./autonomous_trading
```

Run with specific trading pair and interval:
```bash
./autonomous_trading --pair ETHUSD --interval 15m
```

Run for a specific number of ticks:
```bash
./autonomous_trading --ticks 100
```

Save simulation state:
```bash
./autonomous_trading --save my_simulation
```

Load simulation state:
```bash
./autonomous_trading --load my_simulation
```

### Makefile Targets

The Makefile provides several convenient targets:

- `make -f Makefile.autonomous_trading`: Build the project
- `make -f Makefile.autonomous_trading run`: Run with default settings
- `make -f Makefile.autonomous_trading run_btc`: Run with BTCUSD pair for 100 ticks
- `make -f Makefile.autonomous_trading run_save`: Run for 50 ticks and save state
- `make -f Makefile.autonomous_trading run_load`: Load saved state and run for 50 more ticks
- `make -f Makefile.autonomous_trading clean`: Clean build files

## Neural Network Interface

The autonomous trading agent interfaces with the biological neural network through the NeuralNetworkInterface class. This interface:

1. Converts market features into a format suitable for the neural network
2. Normalizes features to the range [-1, 1]
3. Sends the feature vector to the neural network
4. Receives predictions from the neural network
5. Sends reward signals back to the neural network based on trading outcomes

The neural network learns to predict price movements based on the features provided and the reward signals received.

## Reward Mechanism

The reward mechanism is implemented in the Simulation class and is based on the profitability of trading decisions:

- BUY followed by price increase: Large positive reward
- SELL followed by price decrease: Large positive reward
- BUY followed by price decrease: Large negative reward
- SELL followed by price increase: Large negative reward
- HOLD during stable prices: Small positive reward
- HOLD during significant price movements: Small negative reward (opportunity cost)

This reward structure encourages the neural network to make accurate price predictions that lead to profitable trading decisions.

## Logging and Analysis

The system generates several log files that can be used for analysis:

- `autonomous_trading_agent.log`: Detailed log of agent decisions and rewards
- `simulation_log.csv`: CSV file with simulation steps, prices, portfolio values, and rewards

These logs can be analyzed to evaluate the performance of the trading agent and the neural network.

## Dependencies

- C++17 compatible compiler
- libcurl
- OpenSSL
- nlohmann/json (header-only)
- pthread

## Future Improvements

1. Implement real-time trading capabilities
2. Add more sophisticated technical indicators
3. Enhance the reward function with risk-adjusted metrics
4. Implement portfolio optimization strategies
5. Add support for multiple trading pairs
6. Implement backtesting with different time periods
