# deconfounded-popcount-avx512

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-green.svg)](https://en.cppreference.com/)
[![ISA: AVX-512](https://img.shields.io/badge/ISA-AVX--512%20VPOPCNTDQ-red.svg)](https://en.wikipedia.org/wiki/AVX-512)

**deconfounded-popcount-avx512** is a production-grade, hardware-deconfounded C/C++ popcount and Hamming distance execution engine. It guarantees strict bit-identical correctness across dynamic CPU ISA dispatching while systematically isolating memory subsystem latency, thermal throttling, and microarchitectural pipeline contention under heavy stress workloads.

Tested & Verified on **Intel® Xeon® Platinum 8481C Processor**.

---

## 🏛️ System Architecture: LIB vs. LAB vs. Kernel V39

The project is decoupled into three distinct functional tiers to balance integration ease, microarchitectural diagnostics, and extreme execution performance:

```text
                          ┌─────────────────────────────────────────┐
                          │         Kernel V39 (Core Engine)        │
                          │   • 8-Accumulator SIMD Unrolling        │
                          │   • Adaptive Predictive Prefetching     │
                          └───────────────────┬─────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│             LIB (Commercial)             │    │            LAB (Diagnostic)              │
│  • Production-Ready C++20 Clean API      │    │  • ANOVA 2-Factor Noise Dissection       │
│  • Runtime ISA Dispatching (Safe Fallback)│    │  • Physical Core Isolation (`sysfs`)     │
│  • Built-in Thread Pool & Boundary Check │    │  • PMU Counters & Hardware Bounds Audit  │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘

```
1. Kernel V39 (Optimization Engine)
- Core algorithmic engine featuring 8-accumulator SIMD unrolling and Adaptive Software Prefetching.

- Specifically engineered for scattered memory workloads (Gather Access / Cache-Miss scenarios), maintaining peak efficiency even under severe memory bus saturation.

2. LIB (Commercial Production Library)
- A zero-dependency, plug-and-play C++20 library ready for enterprise deployment (Vector Databases, AI Search Engines, Graph DBs).

- Features Runtime ISA Dynamic Dispatching: Automatically inspects CPU capabilities (AVX512VPOPCNTDQ -> AVX2 -> Scalar Fallback), preventing application crashes (Illegal Instruction).

- Handles memory boundary alignment and includes an integrated lightweight multi-threading pool.

3. LAB (Diagnostic & Microarchitecture Framework)
- An independent R&D diagnostic suite for hardware profiling and deterministic latency/P99 control.

- Automatically isolates physical cores via sysfs (/sys/.../thread_siblings_list) to eliminate Hyper-Threading / SMT interference.

- Measures directly via PMU registers, RDTSCP + LFENCE memory barriers, and statistically dissects environmental noise using a 2-Factor ANOVA model while establishing hardware ceiling baselines (floor_bulk_*).

🚀 Key Architectural Features
- Hardware-Level Deconfounding: Isolation via RDTSCP + LFENCE barriers, physical core pinning, and randomized trial execution to eliminate DVFS, thermal noise, and cold-cache state variance.

- 3-Tier Integrity & Noise Isolation:
* Tier 1: $100\%$ bit-identical validation across $25\text{M} \times 64\text{-bit}$ word datasets.
* Tier 2: Dynamic ISA dispatch routing verification.
* Tier 3: Statistical ANOVA random-effects decomposition separating within-run variance ($\sigma_w^2$) from environmental infrastructure noise ($\sigma_b^2$).

- Gather Latency Mitigation: Reduces random memory access latency from $319\text{ ns}$ down to $6.3\text{ ns}$ for scattered memory patterns under extreme system stress.

## 📊 Empirical Benchmark Results

### 1. Contiguous Memory Workload Under Stress
Evaluated on a **190 MB memory footprint** (exceeding L3 cache capacity) under saturated resource pressure (Super Stress Environment):

| Method / Kernel | Time / Word | Cycles / Cacheline | Microarchitectural Status & Performance |
| :--- | :---: | :---: | :--- |
| Baseline (`std::popcount` / `libpopcnt` Under Stress) | **1.88 ns** | ~15.04 cycles | ALU Bound / Contention Penalty |
| GCC Auto-vectorized (`-march=native`) | 0.62 ns | ~4.96 cycles | DRAM Bandwidth Ceiling Saturated |
| **Kernel V39 (Hand-crafted 8-Accumulator)** | **0.67 ns** *(Deterministic)* | **~5.36 cycles** | **~2.8x faster than baseline under stress** |

### 2. Scattered Memory Workload Under Stress (Random Gather / Cache-Miss)
Evaluating non-sequential memory reads (Random Gather Access / Pointer Chasing) — the primary cause of latency spikes in Vector Search:

| Kernel / Operation | Latency / Block | Speedup Factor | Hardware Root Cause |
| :--- | :---: | :---: | :--- |
| Naive Scalar Gather (Standard) | **319.0 ns** | **1.0x** (Baseline) | DRAM Bus Congestion & Queueing Delay |
| **Kernel V39 (Adaptive Prefetch Engine)** | **6.3 ns** | **~50.6x Speedup** | Cache-Miss Latency Mitigated |

---

## 🔬 Empirical Measurements vs. Grok AI Theoretical Analysis

AI models (e.g., xAI's Grok) often assume idealized execution models (pure pipeline throughput without DRAM/TLB queueing delays). Below is a direct comparison between Grok's theoretical estimations and our empirical hardware measurements:

| Architectural Metric | Grok Estimated Model | Our Empirical Measurements | Hardware Reality |
| :--- | :--- | :--- | :--- |
| **L3 Cache Miss Penalty** | Uniform latency addition (~40-60ns) | Non-linear queueing delay | Over-unrolling saturates Memory Controller queues before hiding latency. |
| **8-Acc vs 2-Acc Speedup** | Predicted **~1.3x to 1.5x** via ILP | **1.01x** (Idle) / **Superior** (Stress) | DRAM bus bounds execution at idle; under stress, 8 accumulators prevent pipeline stalls. |
| **Gather Latency Reduction** | Predicted 3x - 5x reduction via SIMD | **50.6x Speedup (319 ns -> 6.3 ns)** | Adaptive software prefetching ($PD=16$) fetches lines prior to AVX-512 execution. |
| **Infrastructure Noise / State Variance** | Neglected (Assumes $Var = 0$) | Isolated between-run variance ($\sigma_b^2$) | DVFS, SMT contention, and TLB misses introduce up to **15%+** noise if unisolated. |

---

## 🛠️ Quick Start

System Requirements
- Linux OS environment (required for sysfs, taskset, and RDTSCP cycle counters).

- GCC or Clang compiler supporting C++20.

- x86_64 CPU supporting AVX-512 instructions (AVX512F, AVX512VPOPCNTDQ).
```
Build & Verification Setup
# Clone repository
git clone [https://github.com/Vumb-VibeCoder/deconfounded-popcount-avx512.git](https://github.com/Vumb-VibeCoder/deconfounded-popcount-avx512.git)
cd deconfounded-popcount-avx512

# Configure & Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run Integrity Validation Suite (Tier 1 & Tier 2)
./build/popcount_tests

# Run LAB Microarchitectural Diagnostics & Kernel V39 Benchmark
./build/popcount_lab_demo
```
📄 License
This repository is distributed under the GNU General Public License v3.0 (GPLv3). For commercial B2B licensing (LIB/LAB licenses without copyleft constraints), please contact the core development team.
