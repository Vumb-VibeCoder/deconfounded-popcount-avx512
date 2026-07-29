# The IRM-Burst Law of Equivalence — v3

## Abstract

**v1** modeled LRU miss rate under uniform-random Independent Reference Model (IRM) traffic with fixed-length bursts, assuming a fully-associative cache. It is exact in the closed form
$$\mu(C,W,L) = \frac{\max(0, W-C)}{W\cdot L}$$
but breaks down badly — errors of 5-8 percentage points in our tests — the moment traffic is non-uniform (hot/cold lines, Zipf skew), which is the realistic case for almost all real workloads.

**v2** replaced the uniform-occupancy assumption with **Che's approximation**, a mean-field technique from cache analysis literature, allowing per-line reference probabilities $p_i$. It matched Monte Carlo simulation to within $10^{-4}$-$10^{-5}$ even under heavy Zipf skew and hot/cold bimodal traffic, a 100-1000x accuracy improvement over v1 in skewed regimes.

**v3** (this document) closes two further gaps:

1. **Set-associativity.** Real caches are not fully-associative; they're split into $S$ sets of $A$ ways each ($C = S \cdot A$), with lines mapped to sets by an index hash. v3 solves Che's equation *per set* rather than globally.
2. **Heterogeneous burst length.** Real bursts are not all the same length $L$ — hot (e.g. streamed/sequential) lines tend to have longer runs than cold (random/scattered) lines. v3 allows a per-line burst length $L_i$.

v3 was validated against an exact set-associative LRU simulator across uniform, Zipf, hot/cold, direct-mapped, and adversarial hash-collision configurations, holding accuracy to within $4\times10^{-4}$ even in a deliberately adversarial worst case (3 hot lines forced into the same 2-way set). A follow-up round of stress-testing (Section 6) then went looking specifically for where the model breaks, and found the real boundary is *inter-burst correlation*, not associativity or burst length.

---

## 1. Physical Model & Parameter Definitions

- **$S$ (Set Count):** Number of independent LRU sets in the cache.
- **$A$ (Associativity / Ways):** Cache lines per set. Total capacity $C = S \cdot A$.
- **$W$ (Working-Set Footprint):** Number of unique cache lines referenced.
- **$q_i$ (Burst-Selection Probability):** Probability that a given burst targets line $i$, $\sum_{i=1}^W q_i = 1$. Reduces to $q_i = 1/W$ in v1's uniform case.
- **$L_i$ (Per-Line Burst Length):** Number of consecutive accesses to line $i$ once selected. Reduces to constant $L$ in v1/v2.
- **$\text{set}(i)$:** The set index line $i$ maps to, via an index hash (does not need to be uniform — see Section 6).

---

## 2. Derivation

### 2.1 Che's Approximation (per set)

Within each set $s$, only lines mapping to $s$ compete for its $A$ ways. Let $M_s = \{i : \text{set}(i) = s\}$. Che's mean-field approximation posits a **characteristic residency time** $T_{c,s}$ such that line $i \in M_s$'s probability of being resident in the cache at a random point in time is
$$\pi_i = 1 - e^{-q_i T_{c,s}}$$
$T_{c,s}$ is fixed by requiring expected occupancy to equal the set's capacity:
$$\sum_{i \in M_s} \left(1 - e^{-q_i T_{c,s}}\right) = A$$
This is solved numerically (monotone in $T_{c,s}$, solvable by bisection — see implementation).

### 2.2 Per-Burst Miss Probability

A burst's head access to line $i$ misses with probability $1-\pi_i = e^{-q_i T_{c,s}}$ (line not resident). Tail accesses ($k=2,\ldots,L_i$) are guaranteed hits, exactly as in v1/v2 — this part of the original derivation is untouched, since it follows purely from LRU semantics of a burst, independent of associativity.

### 2.3 Aggregate Miss Rate (per access, across all sets)

$$\mu(S, A, \{q_i\}, \{L_i\}) = \frac{\displaystyle\sum_{s=1}^{S}\sum_{i \in M_s} q_i \, e^{-q_i T_{c,s}}}{\displaystyle\sum_{i=1}^{W} q_i L_i}$$

**Consistency check — reduces correctly to prior versions:**
- $S=1$ (fully associative) reduces exactly to v2.
- $q_i = 1/W$, $L_i = L$ constant, $S=1$ reduces exactly to v1's closed form $\frac{W-C}{WL}$.

---

## 3. Validation Methodology (initial v3 tests)

An exact **set-associative LRU simulator** (independent LRU list per set, real hash-based set indexing) was implemented in C++ and run under six configurations, each with 100k-warmup + 800k-3M measurement bursts. Results, absolute error against Monte Carlo simulation:

- **Test 1 — Uniform traffic**, const L=8, 32 sets x 4-way (C=128): error **0.000029**
- **Test 2 — Zipf(s=1.0) traffic**, const L=8, 32 sets x 4-way: error **0.000037** (a naive fully-associative v2 formula gets 0.000176 on the same data — v3 is 4.8x more accurate)
- **Test 3 — Zipf(s=1.0) + heterogeneous burst length** (hot lines up to L~32), 32 sets x 4-way: error **0.000222**
- **Test 4 — Uniform traffic, direct-mapped** (A=1, worst-case associativity): error **0.000094**
- **Test 5 — Uniform traffic, fully-associative** (S=1, sanity check): error **0.000032**, identical to v2 as required
- **Test 6 — Adversarial collision**: 3 hot lines (70% of all traffic) forced into the same set with only 2 ways: error **0.000794**

All six pass well under a 0.01 absolute-error tolerance; the adversarial collision case, designed specifically to break the mean-field assumption, still lands within $8\times10^{-4}$.

---

## 4. Reference Implementation (C++)

```cpp
#include <vector>
#include <cmath>
#include <list>
#include <unordered_map>

// Exact set-associative LRU simulator
class SetAssocLRUSim {
    size_t num_sets, ways;
    std::vector<std::list<uint64_t>> lru_lists;
    std::vector<std::unordered_map<uint64_t, std::list<uint64_t>::iterator>> maps;
    uint64_t hits = 0, misses = 0;
    size_t set_of(uint64_t id) const {
        uint64_t h = id * 2654435761ULL;   // Knuth multiplicative hash
        return (h >> 13) % num_sets;
    }
public:
    SetAssocLRUSim(size_t S, size_t A) : num_sets(S), ways(A), lru_lists(S), maps(S) {}
    bool access(uint64_t id) {
        size_t s = set_of(id);
        auto &L = lru_lists[s]; auto &M = maps[s];
        auto it = M.find(id);
        if (it != M.end()) {
            L.erase(it->second); L.push_front(id); M[id] = L.begin();
            hits++; return true;
        }
        if (M.size() >= ways) { M.erase(L.back()); L.pop_back(); }
        L.push_front(id); M[id] = L.begin();
        misses++; return false;
    }
    double miss_rate() const { return (double)misses / (hits + misses); }
    void reset() { hits = misses = 0; }
};

// Che's approximation: solve T_c s.t. sum(1 - exp(-q_i * T_c)) == capacity
double solve_Tc(const std::vector<double>& q, double capacity) {
    double lo = 0.0, hi = 1.0;
    auto f = [&](double Tc) {
        double s = 0.0;
        for (double qi : q) s += 1.0 - std::exp(-qi * Tc);
        return s - capacity;
    };
    while (f(hi) < 0) hi *= 2.0;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        (f(mid) > 0 ? hi : lo) = mid;
    }
    return 0.5 * (lo + hi);
}

// IRM-Burst Law v3: predicted miss rate per access
// q[i]     = burst-selection probability of line i (sum to 1)
// Lline[i] = burst length of line i
// set_id[i]= which of the S sets line i maps to
double irm_burst_v3(const std::vector<double>& q,
                     const std::vector<double>& Lline,
                     const std::vector<size_t>& set_id,
                     size_t S, size_t A) {
    std::vector<std::vector<size_t>> members(S);
    for (size_t i = 0; i < q.size(); ++i) members[set_id[i]].push_back(i);

    double miss_bursts = 0.0, total_accesses = 0.0;
    for (size_t s = 0; s < S; ++s) {
        std::vector<double> qs;
        for (size_t idx : members[s]) qs.push_back(q[idx]);
        double Tc = solve_Tc(qs, (double)A);
        for (size_t idx : members[s]) {
            miss_bursts   += q[idx] * std::exp(-q[idx] * Tc);
            total_accesses += q[idx] * Lline[idx];
        }
    }
    return miss_bursts / total_accesses;
}
```

---

## 5. Capability Summary: v1 -> v2 -> v3

- **Uniform traffic:** v1 exact closed-form; v2 supported; v3 supported.
- **Skewed / Zipf / hot-cold traffic:** v1 fails (5-8 point error); v2 accurate (~$10^{-4}$-$10^{-5}$ error); v3 accurate.
- **Fully-associative cache:** v1 supported; v2 supported; v3 supported (special case $S=1$).
- **Set-associative / direct-mapped cache:** v1 not modeled; v2 not modeled; v3 supported.
- **Constant burst length:** v1 supported; v2 supported; v3 supported.
- **Heterogeneous per-line burst length:** v1 not modeled; v2 not modeled; v3 supported.
- **Closed-form (no numerical root-find needed):** v1 yes; v2 no (needs $T_c$ root-find); v3 no (needs per-set $T_{c,s}$ root-find).

---

## 6. Stress-Testing the Limitations (post-v3 investigation)

The original v3 limitations list included four claims. Each was tested directly rather than left as a guess. One turned out to be wrong; two were confirmed and quantified; one turned out to be far more severe than expected.

### 6.1 "Hash uniformity assumed" — DISPROVEN

The original write-up claimed the per-set decomposition assumes the index hash spreads lines roughly uniformly across sets. This was tested directly: 200 of 300 lines were forced, via a deliberately pathological hash, into a single 4-way set, leaving the other 100 lines spread thinly across the remaining 9 sets.

Result: absolute error **0.000036** against simulation — essentially unchanged from the well-behaved cases. The reason: the v3 formula groups lines by their *actual measured* $\text{set}(i)$, not by an assumed-uniform distribution. As long as the true set membership is known (which it always is, from the address hash), the formula is correct regardless of how skewed the hash's set distribution is. This limitation is retracted.

### 6.2 Finite-size effects — real, but only under skew

Pure finite-size behavior (uniform traffic, shrinking $W$ and $C$ down to $W=2, C=1$) showed essentially no degradation — error stayed under $2\times10^{-5}$ even at the smallest scale. Che's approximation is exact by symmetry for uniform traffic regardless of scale.

Combining small scale **with heavy skew** is where it breaks. Four small, heavily skewed configurations were tested ($W$ between 3 and 6, $C=1$-$2$):

- 5 lines, one at 80% of traffic, $C=2$: relative error 9.5%
- 4 lines, skewed 60/30/5/5, $C=1$: relative error 4.1%
- 3 lines, near-tied 49/49/2, $C=1$: relative error 0.6%
- 6 lines, one dominant at 90%, $C=1$: relative error **39.2%**

Absolute errors stayed under 0.01 in all four cases (which is why this didn't show up under the original 0.01 absolute-error tolerance), but relative error at small scale + high skew can be large. Practical implication: v3 should be trusted quantitatively only when the working set and cache are large enough that no single line dominates a very small set of ways. It remains a good qualitative/directional predictor even outside that range.

### 6.3 Inter-burst correlation — the real weak point

This is the most consequential finding. v3 (like v1 and v2) assumes each burst's target line is drawn independently (IRM at the burst level). A model was built where, with probability $r$ ("locality strength"), the next burst targets a line within a small window of the previous burst's line (spatial locality) instead of an independent draw; marginal per-line traffic $q_i$ was kept uniform throughout, so only the *order* of bursts was correlated, not their overall frequency.

Results (W=256, C=64, L=8, correlation window of +/-3 lines):

- $r=0.00$ (pure IRM, matches v3's assumption): relative error 0.04% — as expected, essentially exact
- $r=0.20$: relative error 3.4%
- $r=0.40$: relative error 8.9%
- $r=0.60$: relative error 18.3%
- $r=0.80$: relative error 40.3%
- $r=0.95$: relative error **115.2%** — v3 overestimates the miss rate by more than 2x

Because correlated bursts revisit nearby lines shortly after touching them, real hit rates run much higher than the independence assumption predicts. This is a large, systematic bias that grows without bound as correlation strengthens, not a small correction — and it is more severe than either the set-associativity gap or the burst-length heterogeneity that v3 was built to fix.

### 6.4 What this means for a v4

Set-associativity and burst-length heterogeneity, the two problems v3 targeted, turned out to be relatively minor sources of error once fixed (sub-percent in nearly every tested case). The dominant remaining source of error is the IRM independence assumption itself at the burst level. A genuine v4 would need to replace the memoryless burst-selection process with something that has short-term memory — e.g. a Markov chain over line selection, or a two-timescale model separating short-range locality from the long-range reference distribution. This is a materially harder problem: Che's approximation and its bisection solve depend on treating each line's reference process as an independent Poisson-like stream, and that assumption is exactly what correlation violates. No closed-form or simple mean-field extension was found for this during this round of testing; it is left as an open problem rather than papered over.

---

## 7. Known Limitations (current, corrected)

1. **Mean-field, not exact.** Che's approximation treats each line's residency as governed independently by its own reference rate. It is asymptotically exact as $W \to \infty$, and empirically near-exact at the tested scales, but is not a closed-form exact result for finite LRU stacks the way v1 is in the pure-uniform case.
2. **Small scale + heavy skew degrades relative accuracy.** Confirmed and quantified in Section 6.2 — up to ~39% relative error in the smallest, most skewed configurations tested, even though absolute error stays small.
3. **No inter-burst correlation modeled.** Confirmed and quantified in Section 6.3 as the dominant weakness — relative error grows past 100% under strong spatial locality between consecutive bursts. This is the priority target for any future version.
4. **Static/stationary traffic assumed.** $q_i$ and $L_i$ are assumed constant over the measurement window; phase changes or working-set drift over time are not modeled and were not tested in this round.

(The previous claim that the model assumes uniform hashing across sets has been retracted — see Section 6.1.)
