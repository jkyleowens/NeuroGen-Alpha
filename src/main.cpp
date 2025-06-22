#include <NeuroGen/AutonomousTradingAgent.h>
#include <NeuroGen/Simulation.h>
#include <NeuroGen/Portfolio.h>
#include <NeuroGen/PriceTick.h>
#include <NeuroGen/NeuralNetworkInterface.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>

// Helper function to load PriceTick data from a CSV file
std::vector<PriceTick> loadPriceTicksFromCSV(const std::string& csv_file_path) {
    std::vector<PriceTick> price_ticks;
    std::ifstream file(csv_file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csv_file_path << std::endl;
        return price_ticks;
    }

    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: CSV file is empty or header could not be read: " << csv_file_path << std::endl;
        return price_ticks;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;

        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        if (fields.size() >= 6) { // timestamp,open,high,low,close,volume
            try {
                long timestamp = std::stol(fields[0]);
                double open = std::stod(fields[1]);
                double high = std::stod(fields[2]);
                double low = std::stod(fields[3]);
                double close = std::stod(fields[4]);
                double volume = std::stod(fields[5]);
                
                PriceTick tick(timestamp, open, high, low, close, volume);
                if (tick.validate()) {
                    price_ticks.push_back(tick);
                } else {
                    std::cerr << "Warning: Invalid PriceTick data in CSV: " << line << std::endl;
                }
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Warning: Invalid data format in CSV line: " << line << " - " << ia.what() << std::endl;
            } catch (const std::out_of_range& oor) {
                std::cerr << "Warning: Data out of range in CSV line: " << line << " - " << oor.what() << std::endl;
            }
        } else {
            std::cerr << "Warning: Incorrect number of fields in CSV line: " << line << " (Expected >=6, Got " << fields.size() << ")" << std::endl;
        }
    }

    file.close();
    if (price_ticks.empty() && !csv_file_path.empty()) {
        std::cerr << "Warning: No valid PriceTicks loaded from " << csv_file_path << std::endl;
    }
    return price_ticks;
}

// Helper function to extract symbol from CSV filename
std::string getSymbolFromFilename(const std::string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    std::string basename = (last_slash == std::string::npos) ? filename : filename.substr(last_slash + 1);
    
    size_t first_underscore = basename.find('_');
    if (first_underscore == std::string::npos) return "UNKNOWN_SYMBOL";

    size_t second_underscore = basename.find('_', first_underscore + 1);
    if (second_underscore == std::string::npos) return basename.substr(0, first_underscore);

    return basename.substr(0, second_underscore);
}

void printUsage() {
    std::cout << "Usage: autonomous_trading_main [OPTIONS]\n"
              << "Options:\n"
              << "  --csv FILENAME        REQUIRED: Path to the CSV file containing historical price data.\n"
              << "  --cash INITIAL_CASH   Initial cash for the portfolio (double, default: 10000.0).\n"
              << "  --ticks NUM           Maximum number of ticks to process in this run (0 for all).\n"
              << "  --load FILE_PREFIX    Load agent and simulation state from files with this prefix.\n"
              << "  --save FILE_PREFIX    Save agent and simulation state to files with this prefix.\n"
              << "  --help                Display this help message.\n";
}

int main(int argc, char* argv[]) {
    // Seed the random number generator for agent's stochastic behaviors
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // --- Default parameters ---
    std::string csv_filename = "";
    double initial_cash = 10000.0;
    int max_ticks = 0;
    std::string load_file_prefix = "";
    std::string save_file_prefix = "";

    // --- Parse command-line arguments ---
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--csv" && i + 1 < args.size()) {
            csv_filename = args[++i];
        } else if (arg == "--cash" && i + 1 < args.size()) {
            try {
                initial_cash = std::stod(args[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --cash. Using default: " << initial_cash << std::endl;
            }
        } else if (arg == "--ticks" && i + 1 < args.size()) {
            try {
                max_ticks = std::stoi(args[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --ticks. Using default (all)." << std::endl;
            }
        } else if (arg == "--load" && i + 1 < args.size()) {
            load_file_prefix = args[++i];
        } else if (arg == "--save" && i + 1 < args.size()) {
            save_file_prefix = args[++i];
        } else if (arg == "--help") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            printUsage();
            return 1;
        }
    }

    if (csv_filename.empty()) {
        std::cerr << "Error: --csv FILENAME is a required argument." << std::endl;
        printUsage();
        return 1;
    }

    try {
        // --- 1. Load Data ---
        std::cout << "Loading historical data from: " << csv_filename << "..." << std::endl;
        std::vector<PriceTick> historical_data = loadPriceTicksFromCSV(csv_filename);
        if (historical_data.empty()) {
            std::cerr << "Error: No data loaded from CSV. Exiting." << std::endl;
            return 1;
        }
        std::cout << "Loaded " << historical_data.size() << " data points." << std::endl;

        // --- 2. Initialize Core Components ---
        std::string symbol = getSymbolFromFilename(csv_filename);
        std::cout << "Determined symbol: " << symbol << std::endl;

        NeuroGen::Portfolio portfolio(symbol, initial_cash);
        
        // Define NN config. In a real application, this would be loaded from a file.
        // The class is in the global namespace, so no "NeuroGen::" prefix is needed.
        NeuralNetworkInterface::Config nn_config; 
        
        // Pass the config to the agent
        NeuroGen::AutonomousTradingAgent agent(symbol, nn_config, portfolio, nullptr);

        // --- 3. Handle State Loading ---
        if (!load_file_prefix.empty()) {
            std::cout << "Attempting to load agent state from prefix: " << load_file_prefix << "..." << std::endl;
            if (agent.loadState(load_file_prefix)) {
                std::cout << "Agent state loaded successfully." << std::endl;
            } else {
                std::cerr << "Warning: Could not load agent state. Starting with a fresh agent." << std::endl;
            }
        }

        // --- 4. Initialize Simulation ---
        NeuroGen::Simulation simulation(agent, portfolio, historical_data, nullptr);

        if (!load_file_prefix.empty()) {
            std::cout << "Attempting to load simulation state from prefix: " << load_file_prefix << "..." << std::endl;
            if (simulation.loadState(load_file_prefix)) {
                std::cout << "Simulation state loaded successfully." << std::endl;
            } else {
                std::cerr << "Warning: Could not load simulation state. Starting from beginning." << std::endl;
            }
        }

        // --- 5. Run Simulation ---
        std::cout << "Running simulation for symbol '" << symbol << "'..." << std::endl;
        auto start_sim_time = std::chrono::high_resolution_clock::now();
        
        simulation.run(max_ticks);
        
        auto end_sim_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_sim_time - start_sim_time);
        
        std::cout << "Simulation run completed in " << duration.count() / 1000.0 << " seconds." << std::endl;

        // --- 6. Save State ---
        if (!save_file_prefix.empty()) {
            std::cout << "Saving state to prefix: " << save_file_prefix << "..." << std::endl;
            bool agent_saved = agent.saveState(save_file_prefix);
            bool sim_saved = simulation.saveState(save_file_prefix);
            if (agent_saved && sim_saved) {
                std::cout << "Agent and Simulation state saved successfully." << std::endl;
            } else {
                std::cerr << "Error: Failed to save one or more state components." << std::endl;
            }
        }
        
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }
}