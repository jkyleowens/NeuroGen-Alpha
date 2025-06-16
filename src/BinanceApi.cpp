#include "BinanceApi.h"
#include <iostream>
#include <stdexcept>
#include <curl/curl.h>
#include "nlohmann/json.hpp" // Assuming nlohmann/json.hpp is available and in the include path

// Callback function for libcurl to write received data into a string
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

BinanceApi::BinanceApi() /*: apiClient(nullptr)*/ {
    // Initialize libcurl globally, if not already done
    // curl_global_init(CURL_GLOBAL_DEFAULT); // Usually done once per application
    // curlHandle = curl_easy_init();
}

BinanceApi::~BinanceApi() {
    // if (curlHandle) {
    //     curl_easy_cleanup(curlHandle);
    // }
    // curl_global_cleanup(); // Usually done once per application
}

bool BinanceApi::initialize() {
    // For libcurl, initialization of a handle might happen in constructor or here
    // if (!curlHandle) {
    //     curlHandle = curl_easy_init();
    //     if (!curlHandle) {
    //         std::cerr << "Failed to initialize libcurl easy handle." << std::endl;
    //         return false;
    //     }
    // }
    // std::cout << "BinanceApi initialized (libcurl ready)." << std::endl;
    return true; // Placeholder
}

std::vector<PriceTick> BinanceApi::fetchHistoricalData(const std::string& pair, long long since, const std::string& interval) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    std::vector<PriceTick> priceTicks;

    curl = curl_easy_init();
    if (curl) {
        // Binance API endpoint for klines (candlesticks)
        // Example: https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=1500000000000&limit=500
        // Note: Binance uses milliseconds for timestamps.
        std::string url = "https://api.binance.com/api/v3/klines?";
        url += "symbol=" + pair;
        url += "&interval=" + interval;
        if (since > 0) {
            url += "&startTime=" + std::to_string(since);
        }
        url += "&limit=1000"; // Max limit is 1000, adjust as needed or implement pagination

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        // Binance API is public for klines, so no API key needed for this specific request
        // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // Uncomment for debugging

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            curl_easy_cleanup(curl);
            throw std::runtime_error("Failed to fetch data from Binance API: " + std::string(curl_easy_strerror(res)));
        }

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code == 200) {
            // std::cout << "Received data: " << readBuffer << std::endl; // For debugging
            priceTicks = _parseKlineData(readBuffer);
        } else {
            std::cerr << "Binance API request failed with HTTP code " << http_code << std::endl;
            std::cerr << "Response: " << readBuffer << std::endl;
            curl_easy_cleanup(curl);
            throw std::runtime_error("Binance API request failed with HTTP code " + std::to_string(http_code));
        }

        curl_easy_cleanup(curl);
    } else {
        throw std::runtime_error("Failed to initialize libcurl handle for fetching data.");
    }

    return priceTicks;
}

std::vector<PriceTick> BinanceApi::_parseKlineData(const std::string& jsonResponse) {
    std::vector<PriceTick> ticks;
    try {
        auto jsonData = nlohmann::json::parse(jsonResponse);

        if (jsonData.is_array()) {
            for (const auto& kline : jsonData) {
                if (kline.is_array() && kline.size() >= 7) { // Ensure there are enough elements
                    PriceTick tick;
                    // Binance kline data format:
                    // [
                    //   [
                    //     1499040000000,      // Kline open time (timestamp in milliseconds)
                    //     "0.01634790",       // Open price
                    //     "0.80000000",       // High price
                    //     "0.01575800",       // Low price
                    //     "0.01577100",       // Close price
                    //     "148976.11427815",  // Volume
                    //     1499644799999,      // Kline close time
                    //     "2434.19055334",    // Quote asset volume
                    //     308,                // Number of trades
                    //     "1756.87402397",    // Taker buy base asset volume
                    //     "28.46694368",      // Taker buy quote asset volume
                    //     "0"                 // Ignore
                    //   ]
                    // ]
                    tick.timestamp = kline[0].get<long long>() / 1000; // Convert ms to s
                    tick.open = std::stod(kline[1].get<std::string>());
                    tick.high = std::stod(kline[2].get<std::string>());
                    tick.low = std::stod(kline[3].get<std::string>());
                    tick.close = std::stod(kline[4].get<std::string>());
                    tick.volume = std::stod(kline[5].get<std::string>());
                    // PriceTick might not have close_time, quote_asset_volume etc.
                    // Adjust according to your PriceTick struct definition.
                    ticks.push_back(tick);
                }
            }
        } else {
            std::cerr << "Binance API response is not a JSON array as expected." << std::endl;
            // Potentially parse error message from Binance if not an array
            // e.g. {"code":-1121,"msg":"Invalid symbol."}
             if (jsonData.contains("msg")) {
                throw std::runtime_error("Binance API error: " + jsonData["msg"].get<std::string>());
            } else {
                throw std::runtime_error("Binance API returned unexpected JSON structure.");
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        std::cerr << "Original response: " << jsonResponse << std::endl;
        throw std::runtime_error("Failed to parse JSON response from Binance API.");
    } catch (const std::exception& e) {
        std::cerr << "Error processing kline data: " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
    return ticks;
}

// Note: If you are using a class member for curlHandle (e.g. `CURL* curlHandle;` in BinanceApi.h)
// then initialize it in the constructor `curlHandle = curl_easy_init();`
// and clean it up in the destructor `curl_easy_cleanup(curlHandle);`.
// Also, `curl_global_init` and `curl_global_cleanup` should ideally be called
// once when your application starts and once when it exits, respectively, not per-instance.
// For simplicity in this example, they are commented out but consider their proper placement.
