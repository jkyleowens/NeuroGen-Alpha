#ifndef BINANCEAPI_H
#define BINANCEAPI_H

#include <string>
#include <vector>
#include "PriceTick.h" // Assuming PriceTick.h is in the same directory or include paths are set up

// Forward declaration for any third-party library client if necessary
// For example, if using a library like Boost.Beast or a dedicated Binance C++ SDK
// namespace ThirdPartyBinance { class Client; }

class BinanceApi {
public:
    BinanceApi();
    ~BinanceApi();

    // Initializes the API client, potentially with API keys if needed for certain endpoints,
    // though public data like klines might not require them.
    bool initialize();

    // Fetches historical kline (OHLCV) data for a given trading pair.
    // pair: e.g., "BTCUSDT"
    // since: UNIX timestamp (milliseconds) to start fetching data from. Binance uses milliseconds.
    // interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
    std::vector<PriceTick> fetchHistoricalData(const std::string& pair, long long since, const std::string& interval);

private:
    // Helper method to parse JSON response from Binance into PriceTick objects
    std::vector<PriceTick> _parseKlineData(const std::string& jsonResponse);

    // Placeholder for a potential third-party library client
    // ThirdPartyBinance::Client* apiClient; 

    // Or, if implementing HTTP requests directly (e.g., with libcurl)
    // void* curlHandle; // Example: CURL* curl;
};

#endif // BINANCEAPI_H
