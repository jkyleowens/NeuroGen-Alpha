# Biologically Inspired Neural Network (CUDA HH‑RK4 + STDP)

## Project snapshot – June 2025

### 1 · High‑level overview

A high‑performance **spiking neural network** implemented in C++/CUDA.  Each neuron follows the *Hodgkin–Huxley* (HH) ionic model integrated with **Runge‑Kutta‑4** (RK4) for numerical stability.  **STDP/Hebbian** plasticity is implemented on synapses.  The latest refactor introduces a **cortical‑column** abstraction that groups neurons into self‑contained micro‑circuits while keeping all maths unchanged.

```text
Host code      Device memory layout (flat arrays)
┌──────────┐   ┌───────────┐
│NetworkCUDA│⇒ │GPUNeuron[]│  HH‑RK4 kernel
└──────────┘   ├───────────┤
               │GPUSynapse[]│  STDP kernels
               ├───────────┤
               │GPUColumn[] │  metadata (slices)
               └───────────┘
```

### 2 · Current implementation details (Iteration 2)

| Layer                                     | Status      | Notes                                                                                                                              |
| ----------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Neuron** (`GPUNeuronState`)             | ✅ stable    | HH variables `m h n`, single‑compartment, voltage threshold detection                                                              |
| **Synapse** (`GPUSynapse`)                | ✅ stable    | weight, delay, activity metric, last‑pre‑spike for STDP                                                                            |
| **Column metadata** (`GPUCorticalColumn`) | ✅ new       | Stores `[neuron_start,end)` & `[syn_start,end)` plus optional local dopamine buffer                                                |
| **Local topology**                        | ✅ new       | Inside each column every neuron forms `localFanOut` random connections; 80 % cells are excitatory (`w>0`), 20 % inhibitory (`w<0`) |
| **Inter‑column wiring**                   | ⏳ *planned* | Sparse, biased feed‑forward synapses (Iteration 3)                                                                                 |
| **Reward modulation**                     | ⏳ *planned* | Per‑column dopamine buffer, weight Δ = STDP × reward (Iter 4)                                                                      |
| **Structural plasticity**                 | ⏳ *planned* | Prune & regrow synapses driven by `activity_metric` (Iter 5)                                                                       |

### 3 · Build & run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./neural_sim         # default config
```

The reference regression test checks spike counts after 100 ms and ensures no NaNs.

### 4 · Configuration (`NetworkConfig`)

```cpp
struct NetworkConfig {
    int   numColumns       = 4;
    int   neuronsPerColumn = 256;
    float dt               = 0.025f;   // ms

    /* Column micro‑circuit */
    float excRatio    = 0.8f;   // 80 % excitatory
    int   localFanIn  = 30;
    int   localFanOut = 30;
    float wExcMin = 0.05f, wExcMax = 0.15f;
    float wInhMin = 0.20f, wInhMax = 0.40f;
    float dMin    = 0.5f,  dMax    = 2.0f;  // delay (ms)

    /* filled automatically */
    size_t totalSynapses = 0;
};
```

Call `finalizeConfig(cfg)` before constructing `NetworkCUDA` to compute `totalSynapses`.

### 5 · Roadmap (next iterations)

| Iter  | Feature                    | Key tasks                                                                                                   |
| ----- | -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **3** | Sparse inter‑column wiring | Generate long‑range synapses with distance‑based probability; feed‑forward bias; gating flag in STDP kernel |
| **4** | Reward modulation          | Per‑column dopamine buffer, global controller module can burst/decay; add factor `R` to weight update       |
| **5** | Structural plasticity      | CUDA kernels to prune low‑activity synapses & sprout new ones toward active targets                         |
| **6** | Column grouping (areas)    | Bundle columns into cortical areas; add hierarchical feed‑forward/feedback tables                           |
| **7** | Performance pass           | Use `__constant__` memory for column table; cooperative groups for RK4 & STDP; optional CUDA Graphs         |

### 6 · LLM Agent integration blueprint

An orchestration LLM (e.g., ChatGPT in *NeuroGen* mode) manages code gen, tests, and docs through a **prompt‑driven CI loop**:

1. **Design prompt** – user describes a feature (e.g., *Add reward modulation*).
2. **LLM generates**:

   * Header & source patches (Markdown fenced blocks or direct Canvas edits).
   * Unit‑test updates.
   * README deltas.
3. **Agent script** (Python) parses the response, applies patches, triggers `cmake --build`, runs tests, and feeds results back to the LLM for the *next* patch.
4. **Validation** – the LLM must reason on compiler/test output and iterate until green.

#### Prompt convention

```text
<directive>::<path>::<content>
```

* `add` – create new file
* `patch` – unified diff
* `replace` – full‑file replacement (small files)

Example:

```text
patch::src/RewardKernel.cuh:::
@@
-float reward         /* old */
+float reward, float mod /* new */
```

#### Agent components

| Layer                 | Role                                                      |
| --------------------- | --------------------------------------------------------- |
| **Patch applier**     | Applies LLM‑returned diffs, resolves simple conflicts     |
| **Build monitor**     | Runs `cmake` + `ctest`, captures errors, truncates logs   |
| **Feedback prompter** | Feeds succinct compile/test output back to LLM            |
| **Doc sync**          | Inserts or updates sections in `README.md`, Doxygen, etc. |

##### Safety rails

* Reject patches that add duplicate struct/class definitions (ODR guard).
* Require every new CUDA kernel to have a host wrapper & unit‑test stub.
* Auto‑format C++ with clang‑format; diff must apply cleanly.

### 7 · How to extend

1. **Describe** your feature in plain language.
2. The orchestration LLM will produce incremental patches & tests.
3. Review/merge; repeat until tests pass and profiling looks good.

---

© 2025 NeuroGen Project – MIT License
