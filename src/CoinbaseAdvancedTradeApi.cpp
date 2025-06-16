#include "CoinbaseAdvancedTradeApi.h"
#include <iostream> // For std::cout, std::cerr
#include <sstream>  // For std::ostringstream
#include <iomanip>  // For std::hex, std::setw, std::setfill
#include <openssl/hmac.h> // For HMAC_CTX_new, HMAC_Init_ex, HMAC_Update, HMAC_Final, HMAC_CTX_free
#include <openssl/sha.h>  // For SHA256_DIGEST_LENGTH
#include <algorithm> // for std::transform

// Helper to convert granularity enum to string for API calls
std::string CoinbaseAdvancedTradeApi::_granularity_to_string(CandleGranularity granularity) {
    switch (granularity) {
        case CandleGranularity::ONE_MINUTE: return "ONE_MINUTE";
        case CandleGranularity::FIVE_MINUTE: return "FIVE_MINUTE";
        case CandleGranularity::FIFTEEN_MINUTE: return "FIFTEEN_MINUTE";
        case CandleGranularity::THIRTY_MINUTE: return "THIRTY_MINUTE"; // Coinbase might use "HALF_HOUR" or similar, check docs
        case CandleGranularity::ONE_HOUR: return "ONE_HOUR";
        case CandleGranularity::TWO_HOUR: return "TWO_HOUR";
        case CandleGranularity::SIX_HOUR: return "SIX_HOUR";
        case CandleGranularity::ONE_DAY: return "ONE_DAY";
        default: return "UNKNOWN_GRANULARITY";
    }
}

// libcurl write callback function
size_t CoinbaseAdvancedTradeApi::_curl_write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

CoinbaseAdvancedTradeApi::CoinbaseAdvancedTradeApi() : curl_handle_(nullptr), initialized_(false) {
    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        std::cerr << "[CoinbaseAdvancedTradeApi] Failed to initialize cURL handle." << std::endl;
        // No throw here, initialize will handle the state
    }
}

CoinbaseAdvancedTradeApi::~CoinbaseAdvancedTradeApi() {
    if (curl_handle_) {
        curl_easy_cleanup(curl_handle_);
    }
    curl_global_cleanup();
}

bool CoinbaseAdvancedTradeApi::initialize(const std::string& api_key, const std::string& api_secret, const std::string& base_url) {
    if (!curl_handle_) {
        std::cerr << "[CoinbaseAdvancedTradeApi] Cannot initialize, cURL handle is null." << std::endl;
        initialized_ = false;
        return false;
    }
    api_key_ = api_key;
    api_secret_ = api_secret;
    base_url_ = base_url;
    initialized_ = true;
    std::cout << "[CoinbaseAdvancedTradeApi] Initialized with base URL: " << base_url_ << std::endl;
    return true;
}

std::string CoinbaseAdvancedTradeApi::_generate_signature(const std::string& timestamp_str, const std::string& method, const std::string& request_path, const std::string& body) {
    std::string message = timestamp_str + method + request_path + body;
    unsigned char* digest = HMAC(EVP_sha256(),
                                 api_secret_.c_str(), api_secret_.length(),
                                 (unsigned char*)message.c_str(), message.length(),
                                 NULL, NULL);

    if (!digest) {
        std::cerr << "[CoinbaseAdvancedTradeApi] HMAC digest generation failed." << std::endl;
        return "";
    }
    
    std::ostringstream oss;
    for (unsigned int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    // Note: OpenSSL's HMAC doesn't require explicit freeing of `digest` if the last two args are NULL.
    // If a non-NULL pointer was passed for the digest output, it would need to be freed.
    return oss.str();
}

long CoinbaseAdvancedTradeApi::_make_request(const std::string& method, const std::string& endpoint, const std::string& params_or_body, bool needs_auth, std::string& response_string) {
    if (!initialized_ || !curl_handle_) {
        std::cerr << "[CoinbaseAdvancedTradeApi] API not initialized or cURL handle invalid." << std::endl;
        return -1; // Or some other error code
    }

    std::string url = base_url_ + endpoint;
    if (method == "GET" && !params_or_body.empty()) {
        url += "?" + params_or_body;
    }

    curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, _curl_write_callback);
    curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl_handle_, CURLOPT_TIMEOUT, 10L); // 10 seconds timeout

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    if (needs_auth) {
        std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        std::string signature = _generate_signature(timestamp, method, endpoint, (method == "POST" || method == "PUT") ? params_or_body : "");
        
        if (signature.empty()) {
            std::cerr << "[CoinbaseAdvancedTradeApi] Failed to generate signature for request." << std::endl;
            if (headers) curl_slist_free_all(headers);
            return -1; // Signature generation failure
        }

        headers = curl_slist_append(headers, ("CB-ACCESS-KEY: " + api_key_).c_str());
        headers = curl_slist_append(headers, ("CB-ACCESS-SIGN: " + signature).c_str());
        headers = curl_slist_append(headers, ("CB-ACCESS-TIMESTAMP: " + timestamp).c_str());
        // Coinbase Advanced Trade API might use a different versioning header, e.g., CB-VERSION
        // headers = curl_slist_append(headers, "CB-VERSION: 2023-02-01"); // Example, check current required version
    }

    if (method == "POST") {
        curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, params_or_body.c_str());
    } else if (method == "GET") {
        // Default is GET, no specific option needed unless changing from POST
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPGET, 1L);
    }
    // Add other methods like PUT, DELETE if needed

    curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);

    response_string.clear();
    CURLcode res = curl_easy_perform(curl_handle_);
    long http_code = 0;

    if (res != CURLE_OK) {
        std::cerr << "[CoinbaseAdvancedTradeApi] curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        http_code = -1; // cURL error
    } else {
        curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &http_code);
    }

    if (headers) {
        curl_slist_free_all(headers);
    }
    return http_code;
}

std::vector<PriceTick> CoinbaseAdvancedTradeApi::_parse_candles_response(const nlohmann::json& json_response) {
    std::vector<PriceTick> ticks;
    try {
        // Coinbase Advanced Trade API /api/v3/brokerage/products/{product_id}/candles
        // Response structure: { "candles": [ { "start": "1672531200", "low": "100.0", "high": "102.0", "open": "101.0", "close": "101.5", "volume": "1000" }, ... ] }
        if (json_response.contains("candles") && json_response["candles"].is_array()) {
            for (const auto& candle_json : json_response["candles"]) {
                PriceTick tick;
                tick.timestamp = std::stoll(candle_json.at("start").get<std::string>()); // Unix timestamp in seconds
                tick.open = std::stod(candle_json.at("open").get<std::string>());
                tick.high = std::stod(candle_json.at("high").get<std::string>());
                tick.low = std::stod(candle_json.at("low").get<std::string>());
                tick.close = std::stod(candle_json.at("close").get<std::string>());
                tick.volume = std::stod(candle_json.at("volume").get<std::string>());
                ticks.push_back(tick);
            }
        } else {
            std::cerr << "[CoinbaseAdvancedTradeApi] JSON response does not contain 'candles' array or is malformed." << std::endl;
            if (json_response.contains("message")) {
                 std::cerr << "[CoinbaseAdvancedTradeApi] Error from API: " << json_response["message"].get<std::string>() << std::endl;
            }
        }
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[CoinbaseAdvancedTradeApi] JSON parsing error in _parse_candles_response: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "[CoinbaseAdvancedTradeApi] Out of range error (likely missing field) in _parse_candles_response: " << e.what() << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "[CoinbaseAdvancedTradeApi] Invalid argument (conversion error) in _parse_candles_response: " << e.what() << std::endl;
    }
    return ticks;
}


std::vector<PriceTick> CoinbaseAdvancedTradeApi::get_product_candles(
    const std::string& product_id,
    long long start_timestamp_s,
    long long end_timestamp_s,
    CandleGranularity granularity) {
    
    if (!initialized_) {
        std::cerr << "[CoinbaseAdvancedTradeApi] API not initialized. Call initialize() first." << std::endl;
        return {};
    }

    std::string granularity_str = _granularity_to_string(granularity);
    if (granularity_str == "UNKNOWN_GRANULARITY") {
        std::cerr << "[CoinbaseAdvancedTradeApi] Invalid candle granularity specified." << std::endl;
        return {};
    }

    // Endpoint: /api/v3/brokerage/products/{product_id}/candles
    std::string endpoint = "/api/v3/brokerage/products/" + product_id + "/candles";
    
    std::ostringstream params;
    params << "start=" << start_timestamp_s
           << "&end=" << end_timestamp_s
           << "&granularity=" << granularity_str;

    std::string response_string;
    long http_code = _make_request("GET", endpoint, params.str(), true, response_string);

    if (http_code == 200) {
        try {
            nlohmann::json json_response = nlohmann::json::parse(response_string);
            return _parse_candles_response(json_response);
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[CoinbaseAdvancedTradeApi] JSON parse error for get_product_candles: " << e.what() << "\nResponse: " << response_string << std::endl;
            return {};
        }
    } else {
        std::cerr << "[CoinbaseAdvancedTradeApi] Failed to get product candles. HTTP Status: " << http_code 
                  << "\nResponse: " << response_string << std::endl;
        return {};
    }
}

long long CoinbaseAdvancedTradeApi::get_server_time() {
    if (!initialized_) {
        std::cerr << "[CoinbaseAdvancedTradeApi] API not initialized." << std::endl;
        return -1;
    }
    // Coinbase Advanced Trade API uses /api/v3/brokerage/time
    // This endpoint does not require authentication.
    std::string endpoint = "/api/v3/brokerage/time";
    std::string response_string;
    long http_code = _make_request("GET", endpoint, "", false, response_string);

    if (http_code == 200) {
        try {
            nlohmann::json json_response = nlohmann::json::parse(response_string);
            // Response: { "data": { "iso": "2023-06-15T10:00:00Z", "epochSeconds": "1686823200" } }
            if (json_response.contains("data") && json_response["data"].contains("epochSeconds")) {
                return std::stoll(json_response["data"]["epochSeconds"].get<std::string>());
            } else {
                 std::cerr << "[CoinbaseAdvancedTradeApi] Malformed server time response: 'data.epochSeconds' missing." << std::endl;
                 std::cerr << "Response: " << response_string << std::endl;
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[CoinbaseAdvancedTradeApi] JSON parse error for get_server_time: " << e.what() << std::endl;
        } catch (const std::invalid_argument& e) {
            std::cerr << "[CoinbaseAdvancedTradeApi] Conversion error for server time: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[CoinbaseAdvancedTradeApi] Failed to get server time. HTTP Status: " << http_code 
                  << "\nResponse: " << response_string << std::endl;
    }
    return -1;
}

bool CoinbaseAdvancedTradeApi::test_connectivity() {
    std::cout << "[CoinbaseAdvancedTradeApi] Testing connectivity..." << std::endl;
    long long server_time = get_server_time();
    if (server_time != -1) {
        std::cout << "[CoinbaseAdvancedTradeApi] Connectivity test successful. Server time (epoch seconds): " << server_time << std::endl;
        // For authenticated endpoints, you might want to try a simple read operation
        // like listing accounts if permissions allow, to test API key validity.
        // Example: std::vector<PriceTick> test_candles = get_product_candles("BTC-USD", time(0) - 3600, time(0), CandleGranularity::ONE_MINUTE);
        // if (!test_candles.empty()) { ... }
        return true;
    } else {
        std::cout << "[CoinbaseAdvancedTradeApi] Connectivity test failed." << std::endl;
        return false;
    }
}
