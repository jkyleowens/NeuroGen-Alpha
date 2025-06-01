#include "TopologyGenerator.h"
#include <algorithm>
#include <numeric>

TopologyGenerator::TopologyGenerator(const NetworkConfig& cfg)
: cfg_(cfg), rng_(std::random_device{}()) {}

void TopologyGenerator::buildLocalLoops(
        std::vector<GPUSynapse>&           syn,
        const std::vector<GPUCorticalColumn>& cols)
{
    std::uniform_real_distribution<float>  w_exc(cfg_.wExcMin, cfg_.wExcMax);
    std::uniform_real_distribution<float>  w_inh(cfg_.wInhMin, cfg_.wInhMax);
    std::uniform_real_distribution<float>  delay(cfg_.dMin,  cfg_.dMax);
    std::uniform_real_distribution<float>  uni(0.f, 1.f);

    syn.clear();
    syn.reserve(cfg_.totalSynapses);

    const int fanIn   = cfg_.localFanIn;
    const int fanOut  = cfg_.localFanOut;
    const int colSize = cfg_.neuronsPerColumn;
    const int excCut  = static_cast<int>(cfg_.excRatio * colSize);

    for (const auto& col : cols)
    {
        for (int n = col.neuron_start; n < col.neuron_end; ++n)
        {
            /* classify neuron once — deterministic: first X are excitatory   */
            bool isExc = (n - col.neuron_start) < excCut;

            /*  Pick `fanOut` random targets in the SAME column               */
            for (int k = 0; k < fanOut; ++k)
            {
                int tgt = col.neuron_start +
                          static_cast<int>(uni(rng_) * colSize);

                GPUSynapse s;
                s.pre_neuron_idx       = n;
                s.post_neuron_idx      = tgt;
                s.delay                = delay(rng_);
                s.weight               = isExc ? w_exc(rng_) : -w_inh(rng_);
                s.last_pre_spike_time  = -1.0f;
                s.activity_metric      = 0.f;

                syn.push_back(s);
            }
        }
    }

    /*  Pad with dummy synapses if we undershot (rare when cfg counts differ) */
    while (syn.size() < cfg_.totalSynapses)
        syn.emplace_back(GPUSynapse{0,0,0,0,-1,0});
}
