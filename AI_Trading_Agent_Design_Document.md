# **AI Trading Agent Design Document (C++ Only)**

## **1\. Introduction & Philosophy**

This document outlines the software architecture for an autonomous AI agent designed to operate within a cryptocurrency trading simulation. The primary goal is to train a novel Biological Neural Network (BNN) by rewarding it for accurate price predictions that lead to profitable trading strategies.  
The design has been refactored to be a **pure C++ application**. This unified approach maximizes performance, eliminates potential language-bridging failures, and ensures seamless integration with the existing C++ BNN and a C++ library for Binance API for data acquisition. The architecture is rooted in **Object-Oriented Principles (OOP)** to ensure **modularity**, **testability**, and **maintainability**.

## **2\. High-Level System Architecture**

The system operates as a self-contained, high-performance loop. The Simulation class drives the process, using the BinanceApi class to fetch market data. At each time step, the TradingAgent uses the TechnicalAnalysis class to generate features for the NeuralNetworkInterface, which queries the BNN for a prediction. The Simulation evaluates the outcome and sends a reward signal back through the agent to the BNN.  
\+-----------------------------------------------------------------+  
|                      C++ Application Core                       |  
|                                                                 |  
|  \+--------------------------+      \+--------------------------+ |
|  |       BinanceApi         |-----\>|     Simulation Class     | |
|  | \- Fetches OHLCV data     |      | \- Manages time series    | |
|  \+--------------------------+      | \- Main event loop        | |
|                                    | \- Calculates reward      |\<--+
|                                    \+-------------^------------+   |
|                                                  |                |
|  \+--------------------------+      \+-------------+------------+   |
|  |      Portfolio           |      |      TradingAgent        |   |
|  | \- Manages assets         |\<-----| \- Orchestrates decisions |   |
|  | \- Tracks P\&L             |      | \- Holds other components |   |
|  \+--------------------------+      \+-------------+------------+   |
|                                                  |                |
|          \+---------------------------+           |                |
|          |                           |           |                |
|          v                           v           v                |
|  \+-----------------+      \+---------------------+  \+--------------------------+
|  | \[External BNN\]  |\<----\>| NeuralNetworkIFace  |  |   TechnicalAnalysis      |
|  | \- Prediction    |      | \- Formats data      |  | \- Calculates indicators  |
|  | \- Learning      |      | \- Sends/gets data   |  |   (SMA, RSI, etc.)       |
|  \+-----------------+      \+---------------------+  \+--------------------------+
|                                                                 |
\+-----------------------------------------------------------------+

## **3\. Core C++ Class Design**

This unified design consolidates all logic within the C++ environment for optimal performance.

### **Class: PriceTick (Struct)**

* **Responsibility**: To hold the complete market data for a single point in time.  
* **Attributes**:  
  * long timestamp: UNIX timestamp of the data point.  
  * double open: Opening price.  
  * double high: Highest price.  
  * double low: Lowest price.  
  * double close: Closing price.  
  * double volume: Trading volume.  
* **Methods**: None (plain data object).

### **Class: BinanceApi**

* **Responsibility**: To interface with a C++ Binance API library to fetch historical market data.  
* **Attributes**:  
  * // (Specific attributes will depend on the chosen Binance C++ library)
* **Methods**:  
  * fetchHistoricalData(const std::string& pair, long since, const std::string& interval) \-\> std::vector\<PriceTick\>: Connects to Binance, requests OHLCV (kline) data for a given currency pair, and parses the response into a vector of PriceTick objects.

### **Class: TechnicalAnalysis**

* **Responsibility**: To process raw price data and compute technical indicators. It provides the feature set that the BNN will use to make predictions. This class replaces the Python DataFeeder.  
* **Attributes**:  
  * const std::vector\<PriceTick\>& price\_series: A reference to the full time series data to avoid copying.  
* **Methods**:  
  * getFeaturesForTick(int index) \-\> std::map\<std::string, double\>: Given an index in the time series, calculates various indicators using data up to that point. Returns a map of feature names to their values (e.g., {"RSI\_14": 45.5, "SMA\_50": 29500.0}).  
  * \_calculateSMA(int period, int index): Private method to calculate Simple Moving Average.  
  * \_calculateRSI(int period, int index): Private method to calculate Relative Strength Index.  
  * \_calculateMACD(int index): Private method to calculate Moving Average Convergence Divergence.  
  * *(Note: C++ libraries like TA-Lib can be wrapped for these calculations, or they can be implemented manually).*

### **Class: NeuralNetworkInterface**

* **Responsibility**: To act as the sole communication bridge to the external Biological Neural Network (BNN).  
* **Attributes**:  
  * BNN\_Connection\_Handle bnn\_connection: A handle or pointer representing the connection to the BNN's C++ API.  
* **Methods**:  
  * getPrediction(const std::map\<std::string, double\>& features) \-\> double: Takes the feature map, formats it as required by the BNN, sends the data, and returns the BNN's price prediction.  
  * sendRewardSignal(double reward): Sends the calculated reward value back to the BNN to trigger its learning/adaptation mechanism.

### **Class: Portfolio**

* **Responsibility**: To manage the agent's assets, track value, and execute trades.  
* **Attributes**:  
  * double cashBalance: The amount of quote currency (e.g., USD).  
  * double coinBalance: The amount of base currency (e.g., BTC).  
  * double initialValue: The starting value of the portfolio for P\&L calculation.  
* **Methods**:  
  * executeBuy(double quantity, double price): Decreases cash, increases coins.  
  * executeSell(double quantity, double price): Increases cash, decreases coins.  
  * getCurrentValue(double currentPrice) const: Calculates and returns cashBalance \+ (coinBalance \* currentPrice).  
  * getProfitAndLoss(double currentPrice) const: Returns the P\&L based on initialValue.

### **Class: TradingAgent**

* **Responsibility**: The "brain" of the operation. It orchestrates the logic, making the final trading decision based on the BNN's output.  
* **Attributes**:  
  * Portfolio portfolio: The agent's wallet.  
  * TechnicalAnalysis tech\_analyzer: The indicator calculation engine.  
  * NeuralNetworkInterface nn\_interface: The interface to the BNN.  
* **Methods**:  
  * makeDecision(int tick\_index, double current\_price): The core decision method.  
    1. Calls tech\_analyzer.getFeaturesForTick(tick\_index) to get the feature set.  
    2. Passes these features to nn\_interface.getPrediction().  
    3. Receives the predicted\_price.  
    4. **Decision Logic**: Compares predicted\_price to current\_price to decide BUY, SELL, or HOLD.  
    5. **Quantity Logic**: Determines *how much* to trade.  
    6. Calls portfolio.executeBuy() or portfolio.executeSell().  
  * receiveReward(double reward): Passes the reward from the simulation to nn\_interface.sendRewardSignal(reward).

### **Class: Simulation**

* **Responsibility**: To manage the overall simulation, data, and the main event loop.  
* **Attributes**:  
  * std::vector\<PriceTick\> timeSeriesData: The entire historical price data for the simulation.  
  * int currentTickIndex: An index pointing to the current position in timeSeriesData.  
  * TradingAgent agent: The trading agent instance.  
  * BinanceApi data\_source: The data provider.  
* **Methods**:  
  * initialize(const std::string& pair, ...): Uses the data\_source to populate timeSeriesData.  
  * run(): The main loop. Iterates through timeSeriesData from start to finish.  
  * advanceTick(): Moves to the next PriceTick. In the loop, this method will:  
    1. Get current data (timeSeriesData\[currentTickIndex\]).  
    2. Call agent.makeDecision().  
    3. Increment currentTickIndex.  
    4. Get the *actual* next price (timeSeriesData\[currentTickIndex\].close).  
    5. Calculate the reward based on the trade's outcome.  
    6. Call agent.receiveReward() to propagate the signal to the BNN.

## **4\. Training and Reward Mechanism**

* **Prediction vs. Actual**: The core of the reward is based on the profitability of the action taken, which is informed by the BNN's prediction.  
* **Reward Function**: A sophisticated reward function will be implemented within the Simulation::advanceTick() method:  
  * Give a large positive reward if the agent BOUGHT and the price went UP.  
  * Give a large positive reward if the agent SOLD and the price went DOWN.  
  * Give a large negative reward (punishment) if the agent BOUGHT and the price went DOWN.  
  * Give a large negative reward if the agent SOLD and the price went UP.  
  * Give a small positive reward for HOLDING correctly (e.g., avoiding a volatile drop).  
  * Give a small negative reward for HOLDING when a profitable move could have been made (opportunity cost).

This pure C++ design provides a high-performance, tightly integrated, and robust framework for building and training the trading agent.