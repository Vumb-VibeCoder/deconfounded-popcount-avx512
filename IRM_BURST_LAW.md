# The IRM-Burst Law of Equivalence

## Abstract

The **IRM-Burst Law of Equivalence** establishes a closed-form, deterministic mathematical framework for predicting cache occupancy and miss rates in Least Recently Used (LRU) cache systems under Independent Reference Model (IRM) traffic augmented with localized burst dynamics. By reformulating cache behavior through exchangeability probability across the working set, this model eliminates the need for expensive trace-driven simulations or heuristic curve-fitting during low-level microarchitectural diagnostics.

---

## 1. Physical Model & Parameter Definitions

The system models a fully associative (or set-associative with uniform distribution) LRU cache interacting with a memory access stream characterized by random block selection followed by localized sequential references.

### Parameters:
- **$C$ (Cache Capacity):** Total capacity of the cache, expressed in cache lines ($C \in \mathbb{N}^+$).
- **$W$ (Working-Set Footprint):** Total number of unique cache lines actively referenced by the workload ($W \in \mathbb{N}^+$).
- **$L$ (Burst Length):** Number of consecutive, identical memory accesses made to a single cache line per burst ($L \in \mathbb{N}^+, L \ge 1$).

---

## 2. Derivation of the Law

### 2.1 Steady-State Occupancy Probability $\Omega(C, W)$

Under exchangeable memory access over $W$ distinct cache lines, each line $i \in \{1, 2, \dots, W\}$ shares an equal long-term probability of residing within the top $C$ positions of the LRU stack. 

The expected cache occupancy fraction $\Omega(C, W)$ at steady state is given by the ratio of cache capacity to working-set footprint, bounded by full saturation:

$$\Omega(C, W) = \min\left(1, \frac{C}{W}\right)$$

- **Sub-Saturated Regime ($W > C$):**
  $$\Omega(C, W) = \frac{C}{W}$$
  The probability that any arbitrary target line $i$ is present in the cache prior to a new burst is identically $\frac{C}{W}$.

- **Saturated Regime ($W \le C$):**
  $$\Omega(C, W) = 1$$
  The entire working set resides permanently in cache post-warmup; zero evictions occur.

---

### 2.2 Expected Miss Rate per Access $\mu(C, W, L)$

Consider a single burst request sequence of length $L$ directed at line $i$:

1. **Access 1 (Burst Head):**
   - Triggers a lookup in the LRU cache.
   - The probability of a cache miss corresponds to the complement of the cache occupancy:
     $$P(\text{Miss}_1) = 1 - \Omega(C, W) = 1 - \min\left(1, \frac{C}{W}\right) = \max\left(0, \frac{W - C}{W}\right)$$

2. **Accesses $2 \dots L$ (Burst Tail):**
   - The first access guarantees line $i$ is loaded into the Most Recently Used (MRU) position.
   - Because $L$ consecutive requests target line $i$ without intermediate accesses to other lines, accesses $2$ through $L$ are guaranteed cache hits:
     $$P(\text{Miss}_k \mid k > 1) = 0$$

3. **Total Expected Miss Rate ($\mu$):**
   Combining the expected misses over a full burst of length $L$:

   $$\mu(C, W, L) = \frac{1 \cdot P(\text{Miss}_1) + \sum_{k=2}^{L} P(\text{Miss}_k)}{L}$$

   $$\mu(C, W, L) = \frac{1 - \min\left(1, \dfrac{C}{W}\right)}{L}$$

   Expressed explicitly across operating regimes:

   $$\mu(C, W, L) = \begin{cases} 0 & \text{if } W \le C \\[10pt] \dfrac{W - C}{W \cdot L} & \text{if } W > C \end{cases}$$

---

## 3. Asymptotic Properties & Boundary Analysis

1. **Burst Damping Limit ($L \to \infty$):**
   $$\lim_{L \to \infty} \mu(C, W, L) = 0$$
   Higher burst length $L$ acts as an inverse linear damping factor on the total miss rate, isolating temporal locality effects from capacity limitations.

2. **Phase Transition Boundary ($W \to C^+$):**
   $$\lim_{W \to C^+} \mu(C, W, L) = 0$$
   A sharp phase transition occurs at $W = C$. Beyond this threshold, miss rate scales linearly with relative footprint expansion $\frac{W - C}{W}$.

3. **Fully Uncached / Thrashing Limit ($W \gg C$):**
   $$\lim_{W \to \infty} \mu(C, W, L) = \frac{1}{L}$$
   In severe thrashing, exactly 1 miss occurs per burst block, yielding an asymptotic ceiling of $\frac{1}{L}$.

---

## 4. In-Process Validation (C++ Implementation)

The following executable C++ program provides an in-process Monte Carlo self-test. It verifies theoretical predictions against an exact software LRU cache simulation.

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <random>
#include <cmath>
#include <iomanip>
#include <cassert>

// Exact Software LRU Cache Simulator
class LRUCacheSim {
private:
    size_t capacity;
    std::list<uint64_t> lru_list;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> cache_map;
    uint64_t hits = 0;
    uint64_t misses = 0;

public:
    explicit LRUCacheSim(size_t cap) : capacity(cap) {}

    bool access(uint64_t line_id) {
        auto it = cache_map.find(line_id);
        if (it != cache_map.end()) {
            lru_list.erase(it->second);
            lru_list.push_front(line_id);
            cache_map[line_id] = lru_list.begin();
            hits++;
            return true;
        } else {
            if (cache_map.size() >= capacity) {
                uint64_t last = lru_list.back();
                lru_list.pop_back();
                cache_map.erase(last);
            }
            lru_list.push_front(line_id);
            cache_map[line_id] = lru_list.begin();
            misses++;
            return false;
        }
    }

    double get_miss_rate() const {
        return static_cast<double>(misses) / (hits + misses);
    }

    void reset_stats() {
        hits = 0;
        misses = 0;
    }
};

// Analytical Law Calculation
double calculate_theoretical_miss_rate(size_t C, size_t W, size_t L) {
    if (W <= C) return 0.0;
    double occupancy = static_cast<double>(C) / static_cast<double>(W);
    return (1.0 - occupancy) / static_cast<double>(L);
}

int main() {
    // Test Configurations
    const size_t C = 64;             // Cache Capacity
    const size_t W = 256;            // Working Set Footprint
    const size_t L = 8;              // Burst Length
    const size_t NUM_BURSTS = 500000;
    const size_t WARMUP_BURSTS = 50000;

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<uint64_t> dist(0, W - 1);

    LRUCacheSim sim(C);

    // Warmup Phase (Eliminate cold-start misses)
    for (size_t i = 0; i < WARMUP_BURSTS; ++i) {
        uint64_t line = dist(rng);
        for (size_t b = 0; b < L; ++b) {
            sim.access(line);
        }
    }

    sim.reset_stats();

    // Measurement Phase
    for (size_t i = 0; i < NUM_BURSTS; ++i) {
        uint64_t line = dist(rng);
        for (size_t b = 0; b < L; ++b) {
            sim.access(line);
        }
    }

    double empirical = sim.get_miss_rate();
    double theoretical = calculate_theoretical_miss_rate(C, W, L);
    double abs_error = std::abs(empirical - theoretical);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== IRM-BURST LAW MONTE CARLO VALIDATION ===" << std::endl;
    std::cout << "Cache Capacity (C)    : " << C << " lines" << std::endl;
    std::cout << "Working Footprint (W) : " << W << " lines" << std::endl;
    std::cout << "Burst Length (L)      : " << L << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Theoretical Miss Rate : " << theoretical << std::endl;
    std::cout << "Empirical Miss Rate   : " << empirical << std::endl;
    std::cout << "Absolute Error        : " << abs_error << std::endl;

    assert(abs_error < 0.005 && "Monte Carlo convergence check failed!");
    std::cout << "Validation Result     : PASSED" << std::endl;

    return 0;
}

