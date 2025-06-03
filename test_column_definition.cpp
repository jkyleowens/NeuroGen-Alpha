#include "src/cuda/CorticalColumn.h"
#include "src/cuda/GPUNeuralStructures.h"
#include <vector>

// Test if GPUCorticalColumn is defined
void test_function() {
    std::vector<GPUCorticalColumn> columns;
    GPUCorticalColumn col;
    col.neuron_start = 0;
    col.neuron_end = 100;
    columns.push_back(col);
}

int main() {
    test_function();
    return 0;
}
