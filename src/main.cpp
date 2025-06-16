#include <NeuroGen/AutonomousTradingAgent.h>
#include <NeuroGen/Simulation.h>
#include <NeuroGen/PriceTick.h> // Required for PriceTick struct
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
#include <sstream> // Required for parsing CSV
#include <algorithm> // Required for std::remove if used for cleaning filenames

// Helper function to load PriceTick data from a CSV file
std::vector<PriceTick> loadPriceTicksFromCSV(const std::string& csv_file_path) {
    std::vector<PriceTick> price_ticks;
    std::ifstream file(csv_file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csv_file_path << std::endl;
        return price_ticks; // Return empty vector
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

        if (fields.size() == 6) { // timestamp,open,high,low,close,volume
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
                std::cerr << "Warning: Invalid data format in CSV line (stod/stol failed): " << line << " - " << ia.what() << std::endl;
            } catch (const std::out_of_range& oor) {
                std::cerr << "Warning: Data out of range in CSV line (stod/stol failed): " << line << " - " << oor.what() << std::endl;
            }
        } else {
            std::cerr << "Warning: Incorrect number of fields in CSV line: " << line << " (Expected 6, Got " << fields.size() << ")" << std::endl;
        }
    }

    file.close();
    if (price_ticks.empty() && !csv_file_path.empty()) {
        std::cerr << "Warning: No valid PriceTicks loaded from " << csv_file_path << std::endl;
    }
    return price_ticks;
}

// Helper function to extract symbol from CSV filename (e.g., BTC_USD_1d_timestamp.csv -> BTC_USD)
std::string getSymbolFromFilename(const std::string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    std::string basename = (last_slash == std::string::npos) ? filename : filename.substr(last_slash + 1);
    
    size_t first_underscore = basename.find('_');
    if (first_underscore == std::string::npos) return "UNKNOWN_SYMBOL"; // Or return basename if no underscores
    
    size_t second_underscore = basename.find('_', first_underscore + 1);
    if (second_underscore == std::string::npos) return basename.substr(0, first_underscore); // e.g. BTC_1d... -> BTC

    return basename.substr(0, second_underscore); // e.g. BTC_USD_1d... -> BTC_USD
}


void printUsage() {
    std::cout << "Usage: autonomous_trading_main [OPTIONS]\n"
              << "Options:\n"
              << "  --csv FILENAME        REQUIRED: CSV file containing historical price data (from crypto_data_csv/ directory)\n"
              << "  --cash INITIAL_CASH   Initial cash for portfolio (double, default: 10000.0)\n"
              << "  --ticks NUM           Maximum number of ticks to process in this run (default: all available from CSV)\n"
              << "  --load FILE_PREFIX    Load simulation and agent state from files with this prefix\n"
              << "  --save FILE_PREFIX    Save simulation and agent state to files with this prefix\n"
              << "  --help                Display this help message\n";
}

int main(int argc, char* argv[]) {
    // --- Configuration Loading (kept for potential non-API agent configs) ---
    std::string config_file_path = "config.json";
    nlohmann::json config;
    std::ifstream config_fs(config_file_path);
    if (config_fs.is_open()) {
        try {
            config_fs >> config;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Warning: Failed to parse config file " << config_file_path << ": " << e.what() << std::endl;
            // Not fatal if config is only for API keys which are not used now
        }
    } else {
        std::cerr << "Warning: Could not open config file: " << config_file_path << ". Proceeding with defaults." << std::endl;
    }

    // API Config (placeholders, as API client might not be used directly for CSV simulation)
    // std::string coinbase_api_key = "YOUR_COINBASE_API_KEY"; 
    // std::string coinbase_api_secret = "YOUR_COINBASE_API_SECRET";
    // std::string coinbase_base_url = "https://api.coinbase.com";
    // ... (rest of API config loading removed or commented out as it's not primary for CSV) ...

    // Default parameters for command-line override
    std::string csv_filename = "";
    // std::string pair = default_product_id; // Removed, will derive from CSV filename
    // std::string interval_str = default_granularity_str; // Removed
    // std::string start_time_cli_str = ""; // Removed
    // std::string end_time_cli_str = "";   // Removed
    double initial_cash = 10000.0;
    int max_ticks = 0; 
    std::string load_file_prefix = "";
    std::string save_file_prefix = "";
    
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--csv" && i + 1 < args.size()) {
            csv_filename = args[++i];
        } else if (arg == "--cash" && i + 1 < args.size()) {
            try {
                initial_cash = std::stod(args[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --cash: " << args[i] << ". Using default: " << initial_cash << std::endl;
            }
        } else if (arg == "--ticks" && i + 1 < args.size()) {
            try {
                max_ticks = std::stoi(args[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --ticks: " << args[i] << ". Using default: " << max_ticks << std::endl;
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
    
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // --- API Client Initialization (No longer primary, pass nullptr to Simulation) ---
    // CoinbaseAdvancedTradeApi coinbase_api; // Keep declaration if Simulation constructor needs it, but don't initialize fully
    // if (!coinbase_api.initialize(coinbase_api_key, coinbase_api_secret, coinbase_base_url)) { ... }
    // if (!coinbase_api.test_connectivity()) { ... }
    // For CSV based simulation, we will pass nullptr for the API client to Simulation and Agent.
    // This assumes Agent/Simulation can handle a null API client if not performing live actions.

    std::string full_csv_path = "crypto_data_csv/" + csv_filename;
    std::string symbol_from_csv = getSymbolFromFilename(csv_filename);
    if (symbol_from_csv == "UNKNOWN_SYMBOL" || symbol_from_csv.find("_") == std::string::npos) {
        std::cerr << "Warning: Could not reliably determine trading symbol from CSV filename: " << csv_filename 
                  << ". Using default symbol 'SIM_PAIR'." << std::endl;
        symbol_from_csv = "SIM_PAIR"; // Default symbol if parsing fails
    }

    try {
        Simulation simulation(nullptr); // Pass nullptr for API client
        
        bool initialized_successfully = false;
        if (!load_file_prefix.empty()) {
            std::cout << "Attempting to load simulation state from prefix: " << load_file_prefix << "..." << std::endl;
            if (simulation.loadState(load_file_prefix)) {
                std::cout << "Simulation state loaded successfully." << std::endl;
                // If state is loaded, it includes historical data. 
                // The symbol might be part of the loaded agent state or simulation state.
                // For now, assume loaded state is self-contained regarding data and symbol.
                // If --csv is also provided with --load, it could imply using new data with a loaded agent.
                // Current logic: loaded state takes precedence for data if sim state file is found.
                // If only agent is loaded, new data from CSV will be used.
                initialized_successfully = true; 
            } else {
                std::cerr << "Failed to load simulation state from prefix: " << load_file_prefix 
                          << ". Proceeding with fresh initialization from CSV: " << full_csv_path << std::endl;
            }
        }

        if (!initialized_successfully) {
            std::cout << "Initializing new simulation for symbol '" << symbol_from_csv << "' from CSV file: " << full_csv_path 
                      << ", initial cash: $" << std::fixed << std::setprecision(2) << initial_cash << "..." << std::endl;
            
            std::vector<PriceTick> historical_data = loadPriceTicksFromCSV(full_csv_path);

            if (historical_data.empty()) {
                std::cerr << "Error: Failed to load historical data from CSV or CSV was empty: " << full_csv_path << std::endl;
                return 1; 
            } else {
                 std::cout << "Loaded " << historical_data.size() << " data points from CSV." << std::endl;
            }

            // Pass nullptr for API client, symbol from CSV, initial cash, and loaded data to simulation
            if (!simulation.initialize(symbol_from_csv, initial_cash, historical_data, nullptr)) {
                std::cerr << "Error: Simulation failed to initialize with data from CSV. Exiting." << std::endl;
                return 1;
            }
            initialized_successfully = true;
        }
        
        if (!initialized_successfully) {
             std::cerr << "Critical Error: Simulation could not be initialized. Exiting." << std::endl;
             return 1;
        }

        std::cout << "Running simulation..." << std::endl;
        auto start_sim_time = std::chrono::high_resolution_clock::now();
        
        simulation.run(max_ticks); // Pass max_ticks (0 means run all available)
        
        auto end_sim_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_sim_time - start_sim_time);
        
        std::cout << "Simulation run completed in " << duration.count() / 1000.0 << " seconds." << std::endl;
        
        if (!save_file_prefix.empty()) {
            std::cout << "Saving simulation state to prefix: " << save_file_prefix << "..." << std::endl;
            if (!simulation.saveState(save_file_prefix)) {
                std::cerr << "Error: Failed to save simulation state." << std::endl;
            } else {
                std::cout << "Simulation state saved successfully with prefix: " << save_file_prefix << std::endl;
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
