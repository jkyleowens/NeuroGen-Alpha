#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <iomanip> // For std::get_time
#include <filesystem>
#include <algorithm> // For std::max_element, std::shuffle
#include <iterator>  // For std::distance
#include "src/cuda/NetworkCUDA.cuh" // Interface to the CUDA-powered network

// --- Portfolio Structure (Unchanged from your version) ---
struct Portfolio {
    float cash = 100000.0f;
    float shares = 0.0f;
    float last_price = 0.0f;
    float previous_portfolio_value = 100000.0f;

    void update(const std::string& action, float current_price) {
        if (action == "buy" && cash >= current_price && current_price > 0) {
            shares += 1;
            cash -= current_price;
            last_price = current_price;
        } else if (action == "sell" && shares >= 1 && current_price > 0) {
            shares -= 1;
            cash += current_price;
            last_price = current_price;
        }
        // Update last_price with the current market price for accurate valuation,
        // even if it's a "hold" or a failed buy/sell.
        if (current_price > 0) {
             last_price = current_price;
        }
    }

    float total_value() const {
        return cash + shares * last_price;
    }

    float compute_reward() {
        float current_portfolio_value = total_value();
        float reward = current_portfolio_value - previous_portfolio_value;
        previous_portfolio_value = current_portfolio_value;
        return reward;
    }
};

// --- CSV Reading and Basic Feature Engineering ---
std::vector<std::string> get_available_ticker_files(const std::string& dir) {
    std::cout << "Searching for ticker CSV files in directory: " << dir << std::endl;
    std::vector<std::string> ticker_files;
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "[ERROR] Data directory '" << dir << "' does not exist or is not a directory." << std::endl;
        return ticker_files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            ticker_files.push_back(entry.path().string());
        }
    }
    if (ticker_files.empty()) {
        std::cout << "No CSV files found in " << dir << std::endl;
    } else {
        std::cout << "Found " << ticker_files.size() << " ticker CSV files." << std::endl;
    }
    return ticker_files;
}

// Reads raw CSVs from Python yfinance script (Expected: Datetime,Open,High,Low,Close,Volume[, Adj Close - if not auto_adjusted])
// And engineers a basic 60-feature vector.
std::vector<std::vector<float>> read_and_process_raw_csv_rows(
    const std::string& filepath,
    std::vector<float>& executable_prices // Output: 'Close' price from CSV
) {
    std::ifstream file(filepath);
    std::string line;
    std::vector<std::vector<float>> all_feature_data;
    executable_prices.clear();

    if (!file.is_open()) {
        std::cerr << "[WARNING] Could not open CSV file: " << filepath << std::endl;
        return all_feature_data;
    }

    std::getline(file, line); // Skip header row

    int line_number = 1;
    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> raw_row_values;

        while (std::getline(ss, token, ',')) {
            raw_row_values.push_back(token);
        }

        // Expecting at least 6 columns: Datetime,Open,High,Low,Close,Volume
        // yfinance might sometimes include 'Adj Close' as a 7th if auto_adjust=False
        if (raw_row_values.size() < 6) {
            std::cerr << "[WARNING] Skipping row " << line_number << " in " << filepath << ": not enough columns. Found "
                      << raw_row_values.size() << ", expected at least 6 (Datetime,O,H,L,C,V)." << std::endl;
            continue;
        }

        try {
            // raw_row_values[0] is Datetime (string)
            // raw_row_values[1] is Open
            // raw_row_values[2] is High
            // raw_row_values[3] is Low
            // raw_row_values[4] is Close
            // raw_row_values[5] is Volume
            // Note: If auto_adjust=True was used in yfinance, 'Close' is already adjusted.
            // If auto_adjust=False, an 'Adj Close' might be at index 5 and 'Volume' at index 6.
            // Assuming auto_adjust=True or that 'Close' at index 4 is the price to use.

            float open_price = std::stof(raw_row_values[1]);
            float high_price = std::stof(raw_row_values[2]);
            float low_price = std::stof(raw_row_values[3]);
            float close_price = std::stof(raw_row_values[4]); // This is our executable price
            float volume = std::stof(raw_row_values[5]);

            std::vector<float> current_features(60, 0.0f); // Initialize 60 features with 0.0f

            // Basic feature engineering:
            current_features[0] = open_price;
            current_features[1] = high_price;
            current_features[2] = low_price;
            current_features[3] = close_price; // Using close price as a feature
            current_features[4] = volume;

            // --- Optional: Derive time-based features from raw_row_values[0] (Datetime string) ---
            // This adds a few more features beyond the basic OHLCV.
            std::tm t{};
            // yfinance datetime format is often "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS-HH:MM" (with UTC offset)
            // We'll try to parse the common part.
            std::string datetime_str = raw_row_values[0];
            std::istringstream ss_dt(datetime_str.substr(0, 19)); // Take first 19 chars "YYYY-MM-DD HH:MM:SS"

            if (ss_dt >> std::get_time(&t, "%Y-%m-%d %H:%M:%S")) {
                current_features[5] = static_cast<float>(t.tm_hour);   // Hour of the day
                current_features[6] = static_cast<float>(t.tm_min);    // Minute of the hour
                current_features[7] = static_cast<float>(t.tm_wday);   // Day of the week (0=Sunday, 6=Saturday)
                current_features[8] = static_cast<float>(t.tm_mday);   // Day of the month
                current_features[9] = static_cast<float>(t.tm_mon);    // Month of the year (0-11)
            } else {
                // std::cerr << "[WARNING] Could not parse datetime string: " << raw_row_values[0] << " at line " << line_number << std::endl;
                // Silently fail on datetime parsing for now, features 5-9 will remain 0.0f
            }
            // --- End Optional Time Features ---
            // Features current_features[10] through current_features[59] will remain 0.0f (padding).

            all_feature_data.push_back(current_features);
            executable_prices.push_back(close_price);

        } catch (const std::invalid_argument& ia) {
            std::cerr << "[WARNING] Invalid numeric argument during parsing at " << filepath << " line " << line_number << ". Error: " << ia.what() << ". Skipping row." << std::endl;
            continue;
        } catch (const std::out_of_range& oor) {
             std::cerr << "[WARNING] Numeric value out of range during parsing at " << filepath << " line " << line_number << ". Error: " << oor.what() << ". Skipping row." << std::endl;
            continue;
        }
    }
    file.close();
    return all_feature_data;
}

// --- Neural Network Output Interpretation (Unchanged) ---
std::string interpret_output(const std::vector<float>& output) {
    if (output.empty()) {
        // std::cerr << "[WARNING] Neural network output is empty. Defaulting to 'hold'." << std::endl; // Can be verbose
        return "hold";
    }
    auto max_it = std::max_element(output.begin(), output.end());
    int max_idx = std::distance(output.begin(), max_it);

    if (max_idx == 0) return "buy";
    if (max_idx == 1) return "sell";
    return "hold";
}

// --- Main Simulation Logic ---
int main(int argc, char* argv[]) {
    // Configuration
    std::string data_dir = "highly_diverse_stock_data"; // Directory with RAW CSVs from Python
    int num_epochs = 5;
    bool shuffle_files_per_epoch = true;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) {
        try { num_epochs = std::stoi(argv[2]); }
        catch (const std::exception& e) { std::cerr << "[WARNING] Invalid num_epochs, using default. " << e.what() << std::endl; }
    }

    std::cout << "--- Trading Simulation Initializing ---" << std::endl;
    std::cout << "Data Directory (expecting raw yfinance CSVs): " << data_dir << std::endl;
    std::cout << "Number of Epochs: " << num_epochs << std::endl;
    std::cout << "CSV Reader will perform basic feature engineering (OHLCV + Time) and pad to 60 features." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    std::vector<std::string> ticker_files = get_available_ticker_files(data_dir);

    if (ticker_files.empty()) {
        std::cerr << "[FATAL ERROR] No ticker CSV files found in '" << data_dir << "'." << std::endl;
        return 1;
    }

    Portfolio portfolio;
    std::random_device rd;
    std::mt19937 rng_engine(rd());

    initializeNetwork();
    std::cout << "CUDA Neural Network Initialized." << std::endl;

    long long total_decisions_made = 0;
    long long total_rows_processed = 0;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\n--- Starting Epoch " << epoch + 1 << "/" << num_epochs << " ---" << std::endl;
        std::cout << "Current Portfolio Value: $" << portfolio.total_value() << std::endl;

        if (shuffle_files_per_epoch) {
            std::shuffle(ticker_files.begin(), ticker_files.end(), rng_engine);
        }

        for (size_t file_idx = 0; file_idx < ticker_files.size(); ++file_idx) {
            const std::string& current_file_path = ticker_files[file_idx];
            std::filesystem::path p(current_file_path);
            // std::cout << "Processing File " << (file_idx + 1) << "/" << ticker_files.size() << ": " << p.filename().string() << std::endl; // Can be verbose

            std::vector<float> executable_prices_from_csv;
            std::vector<std::vector<float>> feature_rows = read_and_process_raw_csv_rows(current_file_path, executable_prices_from_csv);

            if (feature_rows.empty()) {
                // std::cerr << "[INFO] No valid data rows processed from " << current_file_path << ". Skipping this file for epoch." << std::endl; // Verbose
                continue;
            }
            // std::cout << "Loaded and processed " << feature_rows.size() << " data points from " << p.filename().string() << std::endl; // Verbose

            for (size_t i = 0; i < feature_rows.size(); ++i) {
                total_rows_processed++;
                if (executable_prices_from_csv[i] <= 0) {
                    // std::cerr << "[WARNING] Invalid or zero price ($" << executable_prices_from_csv[i] << ") at row " << i << " in " << p.filename().string() << ". Holding." << std::endl; // Verbose
                    portfolio.update("hold", portfolio.last_price); // Update with last known good price for valuation
                    continue;
                }

                float reward_signal = portfolio.compute_reward();
                std::vector<float> network_output = forwardCUDA(feature_rows[i], reward_signal); // feature_rows[i] is now 60 elements
                std::string agent_decision = interpret_output(network_output);
                portfolio.update(agent_decision, executable_prices_from_csv[i]);
                updateSynapticWeightsCUDA(reward_signal);
                total_decisions_made++;

                if (total_decisions_made % 5000 == 0) { // Log less frequently
                    std::cout << "Epoch " << epoch + 1 << ", File " << (file_idx+1) <<"/"<<ticker_files.size()<<" ("<< p.filename().string()<<")"
                              << ", Row: " << i + 1 << "/" << feature_rows.size()
                              << ", Action: " << agent_decision
                              << ", Price: $" << executable_prices_from_csv[i]
                              << ", Shares: " << portfolio.shares
                              << ", Cash: $" << portfolio.cash
                              << ", Port. Value: $" << portfolio.total_value()
                              << ", Reward: " << reward_signal << "\n";
                }
            }
            // std::cout << "Finished processing file " << p.filename().string() << ". Port. Value: $" << portfolio.total_value() << std::endl; // Verbose
        }
        
        std::cout << "--- Epoch " << epoch + 1 << " Completed ---" << std::endl;
        std::cout << "End of Epoch Portfolio Value: $" << portfolio.total_value() << std::endl;
        std::cout << "Total Decisions Made So Far: " << total_decisions_made << std::endl;
    }

    std::cout << "\n--- Trading Simulation Finished ---" << std::endl;
    std::cout << "Final Portfolio Value after " << num_epochs << " epochs: $" << portfolio.total_value() << std::endl;
    std::cout << "Total data rows processed across all files & epochs: " << total_rows_processed << std::endl;
    std::cout << "Total decisions made throughout simulation: " << total_decisions_made << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    // Cleanup GPU resources
    cleanupNetwork();
    std::cout << "CUDA resources cleaned up." << std::endl;

    return 0;
}