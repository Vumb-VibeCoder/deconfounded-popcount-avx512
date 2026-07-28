# AVX-512 Popcount Microbenchmark & Microarchitectural Study

An end-to-end, hardware-deconfounded microarchitectural study evaluating vectorized popcount performance on large memory footprints (up to ~190 MB, significantly exceeding L3 cache capacity). Tested on an Intel Xeon Platinum 8481C processor.

---

## Key Highlights

* Hardware-Level Deconfounding: Strict isolation via RDTSCP + LFENCE memory barriers, physical core pinning (taskset/sysfs), and randomized trial execution to eliminate thermal throttling and state interference.
* ISA Dynamic Dispatching: Runtime fallback and execution paths ranging from scalar baselines to custom 8-accumulator unrolled VPOPCNTDQ instructions.
* Bandwidth Saturation Analysis: Identifies exact ILP (Instruction-Level Parallelism) saturation thresholds where memory bandwidth replaces instruction latency as the primary bottleneck.

---

## Benchmark Results

### 1. Portable Build (-O3 without -march=native)
Manual AVX-512 vectorization dominates scalar compiler outputs when hardware targeting is implicit.

* std::popcount / __builtin_popcountll: 2.65 ns/word (~21.2 cycles/line) - Baseline (1.0x)
* AVX-512 (Hand-crafted, 8-acc): 0.65 ns/word (~5.2 cycles/line) - Speedup ~4.2x

### 2. Native Build (-march=native)
GCC auto-vectorizer achieves impressive saturation, hitting the memory bandwidth ceiling.

* GCC Auto-vectorized Baseline: 0.62 ns/word - Bound by Memory Bandwidth
* Manual Unrolled (2 Accumulators): 0.62 ns/word - Bound by Memory Bandwidth
* Manual Unrolled (8 Accumulators): 0.61 ns/word - Memory Floor Hit (~1.01x relative gain)

Takeaway: Beyond L3 cache (190MB footprint), unrolling beyond 2 accumulators yields diminishing returns (~1.01x). At 0.61 ns/word (~5.09 ns/cacheline), the execution pipeline is completely bound by DRAM bandwidth.

---

## Comparison: Empirical Measurements vs. Grok Analysis

When evaluating theoretical microarchitectural bounds, AI models (like xAI's Grok) often assume idealized execution models (pure pipeline throughput without DRAM/TLB penalties). Below is a comparison between Grok's baseline estimations and our empirical hardware measurements:

1. L3 Cache Miss Penalty Impact
   * Grok Estimated Model: Uniform latency addition (~40-60ns)
   * Our Empirical Measurements: Non-linear memory controller queueing
   * Reality: High-throughput unrolling saturates memory channels long before latency is hidden.

2. 8-Acc Unroll Speedup (vs 2-Acc)
   * Grok Estimated Model: Predicted ~1.3x to 1.5x gain via ILP
   * Our Empirical Measurements: 1.01x actual speedup
   * Reality: Grok assumes pipeline stall cycles; hardware is actually DRAM Bus Bandwidth bound.

3. Compiler Auto-Vectorization
   * Grok Estimated Model: Assumed sub-optimal vs hand-written ASM
   * Our Empirical Measurements: Matches manual AVX-512 (0.62 ns/word)
   * Reality: Modern GCC (-march=native) emits optimal vpopcntq unrolling automatically.

4. State Noise / Variance
   * Grok Estimated Model: Neglected (Assumes i.i.d. execution)
   * Our Empirical Measurements: Between-run variance (sigma_b^2) isolated
   * Reality: Hardware state (DVFS, thermals, TLB coldness) introduces up to 15%+ between-run variance.

---

## Verification & Rigor

* Data Integrity: 100% bit-identical verification across 25,000,000 64-bit words (25M x 64-bit).
* Variance Decomposition: ANOVA random-effects (method-of-moments) implemented directly to separate within-run noise (sigma_w^2) from process/environment variance (sigma_b^2).

---

## Quick Start

### Prerequisites
* GCC / Clang with C++20 support.
* x86_64 CPU supporting AVX-512 (AVX512F, AVX512VPOPCNTDQ).
* Linux environment (for sysfs core pinning and cycle counters).

### Build & Run

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Compile portable build
g++ -O3 -std=c++20 main.cpp -o popcount_portable

# Compile native build (AVX-512 target)
g++ -O3 -march=native -std=c++20 main.cpp -o popcount_native

# Pin to physical core 0 for clean benchmarking
taskset -c 0 ./popcount_native
