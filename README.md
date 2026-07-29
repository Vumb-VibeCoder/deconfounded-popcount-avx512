# deconfounded-popcount-avx512

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-green.svg)](https://en.cppreference.com/)
[![ISA: AVX-512](https://img.shields.io/badge/ISA-AVX--512%20VPOPCNTDQ-red.svg)](https://en.wikipedia.org/wiki/AVX-512)

**deconfounded-popcount-avx512** is a production-grade, hardware-deconfounded C/C++ popcount library and execution engine. It delivers bit-identical correctness across dynamic CPU ISA dispatching while isolating memory subsystem latency, thermal noise, and pipeline interference on large footprints (up to ~190 MB, significantly exceeding L3 cache capacity).

Tested & Verified on **Intel® Xeon® Platinum 8481C Processor**.

---

## 🚀 Key Architectural Features

* **Hardware-Level Deconfounding:** Strict measurement isolation using `RDTSCP` + `LFENCE` memory barriers, physical core pinning (`taskset`/`sysfs`), and randomized trial execution to eliminate thermal throttling and state interference.
* **ISA Dynamic Dispatching:** Runtime CPU identification with seamless fallbacks ranging from scalar baselines to custom 8-accumulator unrolled `VPOPCNTDQ` kernels without global compiler flag hazards.
* **3-Tier Integrity & Noise Isolation:**
  * **Tier 1:** Bit-identical validation across 25,000,000 64-bit words ($25\text{M} \times 64\text{-bit}$).
  * **Tier 2:** Microarchitecture & ISA dynamic dispatch verification.
  * **Tier 3:** Statistical ANOVA random-effects decomposition separating within-run variance ($\sigma_w^2$) from environment noise ($\sigma_b^2$).
* **Bandwidth Saturation Analysis:** Identifies exact ILP (Instruction-Level Parallelism) saturation thresholds where memory bandwidth replaces instruction latency as the primary bottleneck.

---

## 📊 Empirical Benchmark Results

### 1. Portable Build (`-O3` without `-march=native`)
Explicit target-attribute AVX-512 vectorization dominates standard scalar compiler outputs when native targeting is disabled.

| Method / Kernel | Time / Word | Normalized Speedup | Microarch Status |
| :--- | :---: | :---: | :--- |
| `std::popcount` / `__builtin_popcountll` | **2.65 ns** | **1.0x** (Baseline) | ALU/Instruction Bound (~21.2 cycles/line) |
| **AVX-512 (Hand-crafted 8-acc)** | **0.65 ns** | **~4.08x** | Near Bandwidth Floor (~5.2 cycles/line) |

### 2. Native Target Build (`-march=native`)
Evaluated on a **190 MB memory footprint** (DRAM bound, exceeding L3 cache).

| Kernel Implementation | Time / Word | Cycles / Cacheline | Primary Bottleneck |
| :--- | :---: | :---: | :--- |
| GCC Auto-vectorized Baseline | 0.62 ns | ~5.16 cycles | Memory Bandwidth Ceiling |
| Manual Unrolled (2 Accumulators) | 0.62 ns | ~5.16 cycles | Memory Bandwidth Ceiling |
| **Manual Unrolled (8 Accumulators)** | **0.61 ns** | **~5.09 cycles** | **DRAM Memory Floor Hit** |

> **Key Takeaway:** Beyond L3 cache (190 MB footprint), unrolling beyond 2 accumulators yields diminishing returns (~1.01x relative gain). At **0.61 ns/word** (~5.09 ns/cacheline), the execution pipeline is completely saturated and bound by hardware DRAM bus bandwidth.

---

## 🔬 Empirical Measurements vs. Grok AI Analysis

When evaluating theoretical microarchitectural bounds, AI models (such as xAI's Grok) often assume idealized execution models (pure pipeline throughput without DRAM/TLB penalties). Below is a direct comparison between Grok's theoretical estimations and our empirical hardware measurements:

| Architectural Metric | Grok Estimated Model | Our Empirical Measurements | Hardware Reality |
| :--- | :--- | :--- | :--- |
| **L3 Cache Miss Penalty** | Uniform latency addition (~40-60ns) | Non-linear memory controller queueing | High-throughput unrolling saturates memory channels long before latency is hidden. |
| **8-Acc vs 2-Acc Speedup** | Predicted **~1.3x to 1.5x** gain via ILP | **1.01x** actual speedup | Grok assumes pipeline stall cycles; hardware is actually DRAM Bus Bandwidth bound. |
| **Compiler Auto-Vectorization** | Assumed sub-optimal vs hand-written ASM | Matches manual AVX-512 (**0.62 ns/word**) | Modern GCC (`-march=native`) emits optimal `vpopcntq` unrolling automatically. |
| **State Noise / Variance** | Neglected (Assumes i.i.d. execution) | Between-run variance ($\sigma_b^2$) isolated | Hardware state (DVFS, thermals, TLB coldness) introduces up to **15%+** variance. |

---

## 🛠️ Quick Start

### Prerequisites
* Linux environment (for `sysfs` core pinning and high-precision cycle counters).
* GCC / Clang compiler with C++20 support.
* x86_64 CPU supporting AVX-512 (`AVX512F`, `AVX512VPOPCNTDQ`).

### Build & Run Verification Suite

```bash
# Clone the repository
git clone [https://github.com/Vumb-VibeCoder/deconfounded-popcount-avx512.git](https://github.com/Vumb-VibeCoder/deconfounded-popcount-avx512.git)
cd deconfounded-popcount-avx512

# 1. Compile portable build (Dynamic Dispatch / Target Attributes)
g++ -O3 -std=c++20 popcount_v33.cpp -o popcount_portable

# 2. Compile native build (AVX-512 target)
g++ -O3 -march=native -std=c++20 popcount_v33.cpp -o popcount_native

# 3. Pin to physical core 0 for hardware-deconfounded execution
taskset -c 0 ./popcount_native
