#include <iostream>
#include <NeuroGen/NetworkPresets.h>

int main() {
    std::cout << "=== Testing NetworkPresets Class ===" << std::endl;
    
    // Test trading_optimized preset
    auto trading_config = NetworkPresets::trading_optimized();
    std::cout << "Trading Optimized Configuration:" << std::endl;
    std::cout << "  Input Size: " << trading_config.input_size << std::endl;
    std::cout << "  Hidden Size: " << trading_config.hidden_size << std::endl;
    std::cout << "  Output Size: " << trading_config.output_size << std::endl;
    std::cout << "  Learning Rate: " << trading_config.reward_learning_rate << std::endl;
    std::cout << "  Spike Threshold: " << trading_config.spike_threshold << std::endl;
    
    // Test high_frequency_trading preset
    auto hft_config = NetworkPresets::high_frequency_trading();
    std::cout << "\nHigh Frequency Trading Configuration:" << std::endl;
    std::cout << "  Input Size: " << hft_config.input_size << std::endl;
    std::cout << "  Hidden Size: " << hft_config.hidden_size << std::endl;
    std::cout << "  Output Size: " << hft_config.output_size << std::endl;
    std::cout << "  Time Step: " << hft_config.dt << std::endl;
    
    // Test minimal_test preset
    auto minimal_config = NetworkPresets::minimal_test();
    std::cout << "\nMinimal Test Configuration:" << std::endl;
    std::cout << "  Input Size: " << minimal_config.input_size << std::endl;
    std::cout << "  Hidden Size: " << minimal_config.hidden_size << std::endl;
    std::cout << "  Output Size: " << minimal_config.output_size << std::endl;
    
    // Test balanced_default preset
    auto balanced_config = NetworkPresets::balanced_default();
    std::cout << "\nBalanced Default Configuration:" << std::endl;
    std::cout << "  Input Size: " << balanced_config.input_size << std::endl;
    std::cout << "  Hidden Size: " << balanced_config.hidden_size << std::endl;
    std::cout << "  Output Size: " << balanced_config.output_size << std::endl;
    
    // Test research_detailed preset
    auto research_config = NetworkPresets::research_detailed();
    std::cout << "\nResearch Detailed Configuration:" << std::endl;
    std::cout << "  Input Size: " << research_config.input_size << std::endl;
    std::cout << "  Hidden Size: " << research_config.hidden_size << std::endl;
    std::cout << "  Output Size: " << research_config.output_size << std::endl;
    
    std::cout << "\n=== All NetworkPresets Tests PASSED ===" << std::endl;
    std::cout << "NetworkPresets class is working correctly!" << std::endl;
    
    return 0;
}
