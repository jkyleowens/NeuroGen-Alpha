// SynapseInputKernel.cu — Fixed implementation file
#include <NeuroGen/cuda/CudaCompatibility.h>
#include "../../include/NeuroGen/cuda/SynapseInputKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/GridBlockUtils.cuh"
#include <cuda_runtime.h>
#include "NeuronModelConstants.h"

__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Check if presynaptic neuron spiked
    if (neurons[pre_idx].spiked) {
        // Record spike time for STDP
        synapse.last_pre_spike_time = neurons[pre_idx].last_spike_time;
        
        // Update activity metric
        synapse.activity_metric = synapse.activity_metric * 0.99f + 0.01f;
        
        // Apply synaptic input to specific compartment
        int compartment = synapse.post_compartment;
        int receptor = synapse.receptor_index;
        
        // Ensure indices are valid
        if (compartment >= 0 && compartment < MAX_COMPARTMENTS &&
            receptor >= 0 && receptor < MAX_SYNAPTIC_RECEPTORS) {
            
            // Add synaptic conductance with location-dependent scaling
            float location_factor = 1.0f;
            
            // Different processing based on compartment type
            if (compartment > 0) {
                int comp_type = neurons[post_idx].compartment_types[compartment];
                
                if (comp_type == COMPARTMENT_BASAL) {
                    // Basal dendrites may have different scaling
                    location_factor = BASAL_DENDRITE_SCALING;
                } else if (comp_type == COMPARTMENT_APICAL) {
                    // Apical dendrites may have different scaling
                    location_factor = APICAL_DENDRITE_SCALING;
                    
                    // NMDA receptors in apical dendrites can trigger calcium influx
                    if (receptor == RECEPTOR_NMDA && synapse.weight > NMDA_THRESHOLD) {
                        neurons[post_idx].ca_conc[compartment] += NMDA_CA_INFLUX;
                    }
                }
            }
            
            // Apply scaled weight to receptor conductance
            float effective_weight = synapse.weight;
            
            // Apply neuromodulatory effects on synaptic transmission
            if (synapse.effective_weight != 0.0f) {
                effective_weight = synapse.effective_weight;
            }
            
            effective_weight *= location_factor;
            
            // Update receptor conductance
            atomicAdd(&neurons[post_idx].receptor_conductances[compartment][receptor], effective_weight);
            
            // Update ion channel states directly
            if (receptor == RECEPTOR_AMPA) {
                // AMPA channels: fast excitatory
                atomicAdd(&neurons[post_idx].channels.ampa_state[compartment], effective_weight);
                atomicAdd(&neurons[post_idx].channels.ampa_g[compartment], effective_weight);
            } 
            else if (receptor == RECEPTOR_NMDA) {
                // NMDA channels: slow excitatory with Mg2+ block
                atomicAdd(&neurons[post_idx].channels.nmda_state[compartment], effective_weight);
                atomicAdd(&neurons[post_idx].channels.nmda_g[compartment], effective_weight);
                
                // NMDA receptors trigger calcium influx
                if (effective_weight > 0.0f) {
                    // Calculate calcium influx based on voltage-dependent Mg2+ block
                    float v = neurons[post_idx].voltages[compartment];
                    float mg_block = 1.0f / (1.0f + 0.28f * expf(-0.062f * v));
                    float ca_influx = effective_weight * mg_block * NMDA_CA_FRACTION;
                    
                    // Apply calcium influx modulation from neuromodulators
                    float ca_mod = neurons[post_idx].ca_influx_modulation[compartment];
                    if (ca_mod > 0.0f) {
                        ca_influx *= ca_mod;
                    }
                    
                    // Add calcium
                    atomicAdd(&neurons[post_idx].ca_conc[compartment], ca_influx * NMDA_CA_INFLUX);
                }
            }
            else if (receptor == RECEPTOR_GABA_A) {
                // GABA-A channels: fast inhibitory
                atomicAdd(&neurons[post_idx].channels.gaba_a_state[compartment], -effective_weight);
                atomicAdd(&neurons[post_idx].channels.gaba_a_g[compartment], -effective_weight);
            }
            else if (receptor == RECEPTOR_GABA_B) {
                // GABA-B channels: slow inhibitory with G-protein coupling
                atomicAdd(&neurons[post_idx].channels.gaba_b_state[compartment], -effective_weight);
                atomicAdd(&neurons[post_idx].channels.gaba_b_g_protein[compartment], -effective_weight);
            }
            
            // Update receptor state variable for kinetics
            atomicAdd(&neurons[post_idx].receptor_states[compartment][receptor], 1.0f);
        }
    }
}
