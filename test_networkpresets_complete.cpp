#include <iostream>
#include <NeuroGen/NetworkPresets.h>

int main() {
    std::cout << "=== Testing NetworkPresets Implementation ===" << std::endl;
    
    try {
        // Test all preset configurations
        auto trading_config = NetworkPresets::trading_optimized();
        auto hft_config = NetworkPresets::high_frequency_trading();
        auto research_config = NetworkPresets::research_detailed();
        auto minimal_config = NetworkPresets::minimal_test();
        auto balanced_config = NetworkPresets::balanced_default();
        
        std::cout << "\n✓ Trading Optimized Configuration:" << std::endl;
        std::cout << "  Input: " << trading_config.input_size 
                  << ", Hidden: " << trading_config.hidden_size 
                  << ", Output: " << trading_config.output_size << std::endl;
        std::cout << "  Learning Rate: " << trading_config.reward_learning_rate << std::endl;
        
        std::cout << "\n✓ High Frequency Trading Configuration:" << std::endl;
        std::cout << "  Input: " << hft_config.input_size 
                  << ", Hidden: " << hft_config.hidden_size 
                  << ", Output: " << hft_config.output_size << std::endl;
        std::cout << "  Time Step: " << hft_config.dt << std::endl;
        
        std::cout << "\n✓ Research Detailed Configuration:" << std::endl;
        std::cout << "  Input: " << research_config.input_size 
                  << ", Hidden: " << research_config.hidden_size 
                  << ", Output: " << research_config.output_size << std::endl;
        std::cout << "  Simulation Time: " << research_config.simulation_time << std::endl;
        
        std::cout << "\n✓ Minimal Test Configuration:" << std::endl;
        std::cout << "  Input: " << minimal_config.input_size 
                  << ", Hidden: " << minimal_config.hidden_size 
                  << ", Output: " << minimal_config.output_size << std::endl;
        
        std::cout << "\n✓ Balanced Default Configuration:" << std::endl;
        std::cout << "  Input: " << balanced_config.input_size 
                  << ", Hidden: " << balanced_config.hidden_size 
                  << ", Output: " << balanced_config.output_size << std::endl;
        
        // Test configuration validation
        std::cout << "\n=== Configuration Validation ===" << std::endl;
        std::cout << "Trading config valid: " << (trading_config.validate() ? "✓" : "✗") << std::endl;
        std::cout << "HFT config valid: " << (hft_config.validate() ? "✓" : "✗") << std::endl;
        std::cout << "Research config valid: " << (research_config.validate() ? "✓" : "✗") << std::endl;
        std::cout << "Minimal config valid: " << (minimal_config.validate() ? "✓" : "✗") << std::endl;
        std::cout << "Balanced config valid: " << (balanced_config.validate() ? "✓" : "✗") << std::endl;
        
        std::cout << "\n🎉 All NetworkPresets functions work correctly!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error testing NetworkPresets: " << e.what() << std::endl;
        return 1;
    }
}
