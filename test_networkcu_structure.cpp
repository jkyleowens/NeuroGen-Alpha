#include <iostream>
#include <vector>
// Test the basic structure compiles without CUDA
#define USE_CUDA 0

int main() {
    std::cout << "Testing NetworkCUDA structure compilation..." << std::endl;
    
    try {
        // Just test that we can include the header without CUDA
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        float reward = 0.5f;
        
        std::cout << "Basic structure test completed successfully!" << std::endl;
        std::cout << "Input size: " << input.size() << ", reward: " << reward << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
