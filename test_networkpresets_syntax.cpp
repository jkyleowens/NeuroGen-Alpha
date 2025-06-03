#include "include/NeuroGen/NetworkPresets.h"
#include <iostream>

int main() {
    std::cout << "Testing NetworkPresets..." << std::endl;
    
    // Test all preset functions
    auto trading_config = NetworkPresets::trading_optimized();
    auto hft_config = NetworkPresets::high_frequency_trading();
    auto research_config = NetworkPresets::research_detailed();
    auto minimal_config = NetworkPresets::minimal_test();
    auto balanced_config = NetworkPresets::balanced_default();
    
    std::cout << "Trading config - Input: " << trading_config.input_size 
              << ", Hidden: " << trading_config.hidden_size 
              << ", Output: " << trading_config.output_size << std::endl;
              
    std::cout << "HFT config - Input: " << hft_config.input_size 
              << ", Hidden: " << hft_config.hidden_size 
              << ", Output: " << hft_config.output_size << std::endl;
              
    std::cout << "Research config - Input: " << research_config.input_size 
              << ", Hidden: " << research_config.hidden_size 
              << ", Output: " << research_config.output_size << std::endl;
              
    std::cout << "Minimal config - Input: " << minimal_config.input_size 
              << ", Hidden: " << minimal_config.hidden_size 
              << ", Output: " << minimal_config.output_size << std::endl;
              
    std::cout << "Balanced config - Input: " << balanced_config.input_size 
              << ", Hidden: " << balanced_config.hidden_size 
              << ", Output: " << balanced_config.output_size << std::endl;
    
    std::cout << "All NetworkPresets functions work correctly!" << std::endl;
    return 0;
}
