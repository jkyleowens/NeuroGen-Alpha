#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include <NeuroGen/Simulation.h>
#include <NeuroGen/AutonomousTradingAgent.h>
#include <NeuroGen/Portfolio.h>
#include <NeuroGen/PriceTick.h>
#include <NeuroGen/NeuralNetworkInterface.h>

// --- Helper Functions ---

/**
 * @brief Splits a string by a given delimiter.
 * @param s The string to split.
 * @param delimiter The character to split by.
 * @return A vector of strings.
 */
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief Extracts the asset symbol from a standard CSV filename.
 * e.g., "stock_data_csv/AAPL_1d_... .csv" -> "AAPL"
 * @param filename The full path to the CSV file.
 * @return The extracted symbol string.
 */
std::string getSymbolFromFilename(const std::string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    std::string basename = (last_slash == std::string::npos) ? filename : filename.substr(last_slash + 1);
    
    size_t first_underscore = basename.find('_');
    if (first_underscore == std::string::npos) return "UNKNOWN";

    return basename.substr(0, first_underscore);
}


/**
 * @brief Parses a single line from a CSV file into a PriceTick object.
 * @param line The string line from the CSV.
 * @param header The vector of header columns from the CSV.
 * @return A populated PriceTick object.
 */
PriceTick parseCSVLine(const std::string& line, const std::vector<std::string>& header) {
    auto values = split(line, ',');
    if (values.size() < 6 || values.size() > header.size()) {
         throw std::runtime_error("CSV line has mismatched column count. Header has " + std::to_string(header.size()) + ", line has " + std::to_string(values.size()));
    }

    PriceTick tick;
    tick.timestamp = std::stol(values[0]);
    tick.open = std::stod(values[1]);
    tick.high = std::stod(values[2]);
    tick.low = std::stod(values[3]);
    tick.close = std::stod(values[4]);
    tick.volume = std::stod(values[5]);

    // Dynamically parse additional columns as indicators
    for (size_t i = 6; i < values.size(); ++i) {
        if (!values[i].empty()) {
            double val = std::stod(values[i]);
            tick.indicators[header[i]] = val;
        }
    }
    return tick;
}

/**
 * @brief Loads a time series of PriceTick data from a CSV file.
 * @param filepath The full path to the CSV file.
 * @return A vector of PriceTick objects.
 */
std::vector<PriceTick> loadHistoricalData(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open data file: " + filepath);
    }

    std::vector<PriceTick> data;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Cannot read header from empty file: " + filepath);
    }
    std::vector<std::string> header = split(line, ',');

    while (std::getline(file, line)) {
        try {
            data.push_back(parseCSVLine(line, header));
        } catch (const std::exception& e) {
            std::cerr << "Warning: Skipping bad CSV line. Reason: " << e.what() << std::endl;
        }
    }
    return data;
}

void printUsage() {
    std::cout << "Usage: autonomous_trading_main [OPTIONS]\n"
              << "Options:\n"
              << "  --csv FILENAME        REQUIRED: Path to the CSV file containing historical price data.\n"
              << "  --cash INITIAL_CASH   Initial cash for the portfolio (double, default: 10000.0).\n"
              << "  --load FILE_PREFIX    Load agent state from files with this prefix.\n"
              << "  --save FILE_PREFIX    Save agent state to files with this prefix.\n"
              << "  --help                Display this help message.\n";
}


// --- Main Simulation Entry Point ---

int main(int argc, char* argv[]) {
    // --- Default parameters ---
    std::string csv_filename = "";
    double initial_cash = 10000.0;
    std::string load_file_prefix = "";
    std::string save_file_prefix = "";

    // --- Parse command-line arguments ---
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--csv" && i + 1 < args.size()) {
            csv_filename = args[++i];
        } else if (arg == "--cash" && i + 1 < args.size()) {
            initial_cash = std::stod(args[++i]);
        } else if (arg == "--load" && i + 1 < args.size()) {
            load_file_prefix = args[++i];
        } else if (arg == "--save" && i + 1 < args.size()) {
            save_file_prefix = args[++i];
        } else if (arg == "--help") {
            printUsage();
            return 0;
        }
    }

    if (csv_filename.empty()) {
        std::cerr << "Error: --csv FILENAME is a required argument." << std::endl;
        printUsage();
        return 1;
    }

    try {
        // 1. Load Data
        std::cout << "Loading data from " << csv_filename << "..." << std::endl;
        std::vector<PriceTick> historical_data = loadHistoricalData(csv_filename);
        if (historical_data.empty()) {
            std::cerr << "Error: No data loaded from file. Exiting." << std::endl;
            return 1;
        }
        std::cout << "Loaded " << historical_data.size() << " data points." << std::endl;

        // 2. Initialize Components
        std::string symbol = getSymbolFromFilename(csv_filename);
        NeuroGen::Portfolio portfolio(symbol, initial_cash);
        
        NeuralNetworkInterface::Config nn_config; 
        
        NeuroGen::AutonomousTradingAgent agent(symbol, nn_config, portfolio, nullptr);

        // 3. Load agent state if specified
        if (!load_file_prefix.empty()) {
            std::cout << "Attempting to load state from prefix: " << load_file_prefix << std::endl;
            if (agent.loadState(load_file_prefix)) {
                std::cout << "Agent state loaded successfully." << std::endl;
            } else {
                std::cerr << "Warning: Failed to load agent state. Starting fresh." << std::endl;
            }
        }

        // 4. Setup and Run Simulation
        NeuroGen::Simulation simulation(agent, portfolio, historical_data, nullptr);

        std::cout << "\nStarting Simulation for " << symbol << "..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        simulation.run();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // 5. Report Results
        std::cout << "\nSimulation Finished in " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds." << std::endl;
        
        const auto& final_portfolio = simulation.getPortfolio();
        double final_value = final_portfolio.getCurrentValue(historical_data.back().close);

        std::cout << "--------------------" << std::endl;
        std::cout << "  Final Portfolio  " << std::endl;
        std::cout << "--------------------" << std::endl;
        std::cout << "Cash:      $" << std::fixed << std::setprecision(2) << final_portfolio.getCashBalance() << std::endl;
        std::cout << "Asset:     " << std::fixed << std::setprecision(4) << final_portfolio.getCoinBalance() << " " << symbol << std::endl;
        std::cout << "Total Value: $" << std::fixed << std::setprecision(2) << final_value << std::endl;
        std::cout << "--------------------" << std::endl;

        // 6. Save agent state if specified
        if (!save_file_prefix.empty()) {
            std::cout << "Saving agent state to prefix: " << save_file_prefix << std::endl;
            if (agent.saveState(save_file_prefix)) {
                std::cout << "Agent state saved successfully." << std::endl;
            } else {
                std::cerr << "Error: Failed to save agent state." << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}