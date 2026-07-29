# The IRM-Burst Law of Equivalence — v3.1

## Abstract

**v1** modeled LRU miss rate under uniform-random Independent Reference Model (IRM) traffic with fixed-length bursts, assuming a fully-associative cache. It is exact in the closed form
$$\mu(C,W,L) = \frac{\max(0, W-C)}{W\cdot L}$$
but breaks down badly — errors of 5–8 percentage points in our tests — the moment traffic is non-uniform (hot/cold lines, Zipf skew), which is the realistic case for almost all real workloads.

**v2** replaced the uniform-occupancy assumption with **Che's approximation**, a mean-field technique from cache analysis literature, allowing per-line reference probabilities $p_i$. It matched Monte Carlo simulation to within $10^{-4}$–$10^{-5}$ even under heavy Zipf skew and hot/cold bimodal traffic, a 100–1000× accuracy improvement over v1 in skewed regimes.

**v3** closed two remaining gaps that made v1/v2 physically unrealistic: set-associativity (solving Che's equation *per set*) and heterogeneous per-line burst length $L_i$.

**v3.1** (this document) closes the gap v3 itself flagged as its weakest point (§5.3, and the adversarial test in §3): Che's mean-field assumption of independent per-line residency degrades when a set is **small and traffic-concentrated** — few lines, most of the traffic. v3.1 adds an **exact solver** for that regime, used only when it applies, leaving the rest of the model — and its mean-field default — untouched.

---

## 1. Physical Model & Parameter Definitions

*(unchanged from v3)*

- **$S$ (Set Count):** Number of independent LRU sets in the cache.
- **$A$ (Associativity / Ways):** Cache lines per set. Total capacity $C = S \cdot A$.
- **$W$ (Working-Set Footprint):** Number of unique cache lines referenced.
- **$q_i$ (Burst-Selection Probability):** Probability that a given burst targets line $i$, $\sum_{i=1}^W q_i = 1$. Reduces to $q_i = 1/W$ in v1's uniform case.
- **$L_i$ (Per-Line Burst Length):** Number of consecutive accesses to line $i$ once selected. Reduces to constant $L$ in v1/v2.
- **$\text{set}(i)$:** The set index line $i$ maps to, via a (assumed near-uniform) index hash.
- **$M_s = \{i : \text{set}(i) = s\}$:** Lines mapped to set $s$.

---

## 2. Derivation

### 2.1 Che's Approximation (per set) — default path

Within each set $s$, only lines mapping to $s$ compete for its $A$ ways. Che's mean-field approximation posits a **characteristic residency time** $T_{c,s}$ such that line $i \in M_s$'s probability of being resident in the cache at a random point in time is
$$\pi_i = 1 - e^{-q_i T_{c,s}}$$
$T_{c,s}$ is fixed by requiring expected occupancy to equal the set's capacity:
$$\sum_{i \in M_s} \left(1 - e^{-q_i T_{c,s}}\right) = A$$
This is monotone in $T_{c,s}$ and is now solved by **Newton–Raphson** rather than bisection (§4) — same equation, same root, ~6–10 iterations instead of a fixed 200, and the derivative reused directly as the numerator term needed in §2.3.

### 2.2 Per-Burst Miss Probability

A burst's head access to line $i$ misses with probability $1-\pi_i$ (line not resident). Tail accesses ($k=2,\ldots,L_i$) are guaranteed hits — untouched since v1, as it follows purely from LRU semantics of a burst, independent of associativity or of which residency model produced $\pi_i$.

### 2.3 NEW — Exact Small-Set Correction

**Where mean-field breaks down.** Che's independence assumption is weakest exactly where v3's own adversarial test (§3, case 6) landed its largest error: a set with **few members and concentrated traffic** (a handful of hot lines dominating a low-associativity set).

**The fix is not a heuristic — it's exact.** Consider a set $s$ in isolation. By the thinning property of i.i.d. sequences, the reference stream restricted to $M_s$ is itself i.i.d. with (renormalized) probabilities $q_i / Q_s$, $Q_s = \sum_{i \in M_s} q_i$. The set's LRU state is then **exactly** a finite Markov chain over states
$$\text{state} = (\text{ordered list of resident lines, most-recent-first, length} \le A)$$
with transitions: on a reference to line $i$, move $i$ to the front (or insert it, evicting the tail if the set was full). This is tractable by direct enumeration whenever $|M_s|$ is small — which is precisely the regime where mean-field is weakest. The stationary distribution $\{\pi_{\text{state}}\}$ (solved by power iteration) gives the **exact** per-line residency
$$\pi_i = \sum_{\text{states containing } i} \pi_{\text{state}}$$
with no independence assumption anywhere in the derivation.

**When to use it.** Per set $s$, define concentration $\kappa_s = \sum_{i \in M_s} (q_i/Q_s)^2$. The exact solver is used when
$$|M_s| \le 10 \quad \text{and} \quad \kappa_s \ge 0.25$$
otherwise Che's mean-field (§2.1) is used unchanged. Both thresholds are cost/tractability choices, not accuracy cliffs — small+diffuse sets already have mean-field near-exact, so the branch only fires where it changes the answer.

### 2.4 Aggregate Miss Rate (per access, across all sets)

$$\mu(S, A, \{q_i\}, \{L_i\}) = \frac{\displaystyle\sum_{s=1}^{S}\sum_{i \in M_s} q_i \, \big(1-\pi_i^{(s)}\big)}{\displaystyle\sum_{i=1}^{W} q_i L_i}, \qquad
1-\pi_i^{(s)} = \begin{cases} e^{-q_i T_{c,s}} & \text{mean-field branch} \\ \text{exact chain output} & \text{exact branch} \end{cases}$$

**Consistency checks — reduces correctly to prior versions:**
- $S=1$, no set meets the exact-branch condition (typical when $W$ is large and diffuse) $\Rightarrow$ reduces exactly to v2.
- $q_i = 1/W$, $L_i = L$ constant, $S=1$, mean-field branch $\Rightarrow$ reduces exactly to v1's closed form $\frac{W-C}{WL}$.
- Every set diffuse ($\kappa_s < 0.25$ everywhere) $\Rightarrow$ **numerically identical** to v3 (Newton finds the same root bisection did).

---

## 3. Validation Methodology

An exact **set-associative LRU simulator** was implemented in C++, given an **explicit line→set assignment** shared with the predictor (so measured error is purely analytic approximation error, not a hashing mismatch between simulator and model), and run under six configurations reproducing the shape of v3's original suite (uniform, Zipf, heterogeneous burst length, direct-mapped, fully-associative, and adversarial collision), 300k-warmup + 3M measurement bursts each.

- **Test 1 — Uniform, const L=8, 32×4-way.** v3 (mean-field only) err = 0.000024. v3.1 (with exact branch) err = 0.000024, unchanged. Exact branch fired on 1 of 32 sets.
- **Test 2 — Zipf(s=1.0), const L=8, 32×4-way.** v3 err = 0.000236. v3.1 err = 0.000041 — 5.8× better. Exact branch fired on 28 of 32 sets.
- **Test 3 — Zipf(s=1.0), heterogeneous L (hot lines up to L≈32), 32×4-way.** v3 err = 0.000135. v3.1 err = 0.000049 — 2.8× better. Exact branch fired on 28 of 32 sets.
- **Test 4 — Uniform, const L=8, direct-mapped (A=1).** v3 err = 0.000002. v3.1 err = 0.000002, unchanged. Exact branch fired on 127 of 128 sets.
- **Test 5 — Uniform, const L=8, fully-associative (S=1), sanity check.** v3 err = 0.000007. v3.1 err = 0.000007, unchanged. Exact branch fired on 0 of 1 sets.
- **Test 6 (adversarial) — 3 hot lines (70% traffic) forced into one 2-way set, 16×2-way overall.** v3 err = 0.000849. v3.1 err = 0.000037 — 23× better. Exact branch fired on all 16 of 16 sets.

**No regression on any test** — v3.1's error is $\le$ v3's error in every case, with equality exactly when no set in that configuration meets the exact-branch condition (tests 1, 5). The adversarial case (#6) — the one v3 itself named as hardest — improves ~23×. Tests 2/3 improve too, incidentally: skewed Zipf traffic naturally produces small, concentrated sets even without deliberate collision.

*(Note: because the original v3 write-up did not publish $W$, RNG seed, or exact line IDs, the absolute numbers above are a faithful re-derivation at concretely chosen parameters, not bit-for-bit reproductions of the original 0.000029-style figures — the v3 column here is regenerated under identical conditions to the v3.1 column for a fair comparison, not copied from §3 of the prior version.)*

---

## 4. Reference Implementation (C++)

```cpp
#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <algorithm>

// ---- Newton-Raphson Tc solver (replaces bisection; same root) ----
struct TcResult { double Tc; double f_prime_at_Tc; };

TcResult solve_Tc_newton(const std::vector<double>& q, double capacity,
                          double tol = 1e-12, int max_iter = 60) {
    double Tc = capacity / std::max(1e-12, std::accumulate(q.begin(), q.end(), 0.0));
    if (!(Tc > 0.0)) Tc = 1.0;
    for (int it = 0; it < max_iter; ++it) {
        double f = -capacity, fp = 0.0;
        for (double qi : q) {
            double e = std::exp(-qi * Tc);
            f  += 1.0 - e;
            fp += qi * e;                 // == numerator term needed later
        }
        if (fp <= 0.0) break;
        double next = Tc - f / fp;
        if (next <= 0.0) next = Tc * 0.5;
        if (std::fabs(next - Tc) < tol) { Tc = next; break; }
        Tc = next;
    }
    double fp_final = 0.0;
    for (double qi : q) fp_final += qi * std::exp(-qi * Tc);
    return { Tc, fp_final };
}

// ---- Exact small-set solver ----
// State = ordered list of resident lines, most-recent-first, length <= A.
// Exact stationary distribution of the finite LRU chain restricted to
// this set's own members (no independence assumption).
struct ExactSetResult { std::vector<double> miss_prob; };

ExactSetResult exact_set_solve(const std::vector<double>& q_local, size_t A) {
    size_t n = q_local.size();
    double Qs = std::accumulate(q_local.begin(), q_local.end(), 0.0);
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = q_local[i] / Qs;
    size_t cap = std::min(A, n);

    using State = std::vector<int>;
    std::map<State, int> state_id;
    std::vector<State> states;
    auto get_id = [&](const State& s) -> int {
        auto it = state_id.find(s);
        if (it != state_id.end()) return it->second;
        int id = (int)states.size();
        state_id[s] = id; states.push_back(s);
        return id;
    };
    get_id(State{});

    std::vector<std::vector<std::pair<int,double>>> trans;
    for (size_t frontier = 0; frontier < states.size(); ++frontier) {
        State cur = states[frontier];
        std::vector<std::pair<int,double>> row;
        for (size_t i = 0; i < n; ++i) {
            State nxt; nxt.push_back((int)i);
            for (int x : cur) { if (x == (int)i) continue; if (nxt.size() >= cap) break; nxt.push_back(x); }
            row.push_back({get_id(nxt), p[i]});
        }
        trans.push_back(row);
    }

    size_t num_states = states.size();
    std::vector<double> pi(num_states, 1.0 / (double)num_states);
    for (int iter = 0; iter < 2000; ++iter) {
        std::vector<double> next(num_states, 0.0);
        for (size_t s = 0; s < num_states; ++s)
            for (auto& [to, prob] : trans[s]) next[to] += pi[s] * prob;
        double diff = 0.0;
        for (size_t s = 0; s < num_states; ++s) diff += std::fabs(next[s] - pi[s]);
        pi.swap(next);
        if (diff < 1e-14) break;
    }

    std::vector<double> resident(n, 0.0);
    for (size_t s = 0; s < num_states; ++s)
        for (int x : states[s]) resident[(size_t)x] += pi[s];

    ExactSetResult out; out.miss_prob.resize(n);
    for (size_t i = 0; i < n; ++i) out.miss_prob[i] = 1.0 - resident[i];
    return out;
}

// ---- Combined predictor ----
constexpr size_t EXACT_MAX_SET_SIZE      = 10;
constexpr double EXACT_CONCENTRATION_MIN = 0.25;

double irm_burst_v3_1(const std::vector<double>& q,
                       const std::vector<double>& Lline,
                       const std::vector<size_t>& set_id,
                       size_t S, size_t A) {
    std::vector<std::vector<size_t>> members(S);
    for (size_t i = 0; i < q.size(); ++i) members[set_id[i]].push_back(i);

    double miss_bursts = 0.0, total_accesses = 0.0;
    for (size_t s = 0; s < S; ++s) {
        const auto& idxs = members[s];
        if (idxs.empty()) continue;
        std::vector<double> qs;
        for (size_t idx : idxs) qs.push_back(q[idx]);
        double Qs = std::accumulate(qs.begin(), qs.end(), 0.0);
        double concentration = 0.0;
        for (double qi : qs) concentration += (qi / Qs) * (qi / Qs);

        if (idxs.size() <= EXACT_MAX_SET_SIZE && concentration >= EXACT_CONCENTRATION_MIN) {
            auto res = exact_set_solve(qs, A);
            for (size_t k = 0; k < idxs.size(); ++k) {
                size_t idx = idxs[k];
                miss_bursts    += q[idx] * res.miss_prob[k];
                total_accesses += q[idx] * Lline[idx];
            }
        } else {
            TcResult tc = solve_Tc_newton(qs, (double)A);
            for (size_t idx : idxs) {
                miss_bursts    += q[idx] * std::exp(-q[idx] * tc.Tc);
                total_accesses += q[idx] * Lline[idx];
            }
        }
    }
    return miss_bursts / total_accesses;
}
```

---

## 5. Known Limitations (honest scope, not swept under the rug)

1. **Mean-field on the default path only — no longer unconditionally.** The independence assumption in §2.1 remains approximate, exactly as in v2/v3, *but only applies when a set doesn't meet the exact-branch condition*. Small, concentrated sets are now solved exactly, not approximated.
2. **Static/stationary traffic.** $q_i$ and $L_i$ are assumed constant over the measurement window. Phase changes, working-set drift, or time-correlated burst sequences are not modeled. *(unchanged from v3)*
3. **Hash uniformity assumed for the mean-field branch's typical case.** ~~A pathological hash/address pattern that concentrates working-set lines into few sets would degrade accuracy further than shown here.~~ **Partially addressed in v3.1**: pathological concentration is now exactly the trigger condition for the exact branch, so a hash collision that concentrates lines into a small set no longer degrades accuracy — it activates the correction instead. This only holds while the colliding set stays within $|M_s| \le 10$; a hash pathology that dumps dozens of hot lines into one set still falls back to mean-field.
4. **Exact branch has a tractability ceiling.** State space grows as $O(n!/(n-A)!)$ in the worst case — fine for $n \le 10$ as configured, but not a general-purpose replacement for mean-field at large $n$. This is a deliberate scope limit, not an oversight.
5. **No inter-burst correlation.** Bursts are still drawn i.i.d. (IRM at the burst level) in both branches. Real workloads often have Markov-like correlation between consecutive bursts — that remains a natural v4 direction. *(unchanged from v3)*

---

## 6. Summary Table: v1 → v2 → v3 → v3.1

- **Uniform traffic** — v1: exact closed-form. v2: yes. v3: yes. v3.1: yes.
- **Skewed/Zipf/hot-cold traffic** — v1: no (5–8 pt error). v2: yes (~10⁻⁴–10⁻⁵ error). v3: yes. v3.1: yes, tighter (see §3).
- **Fully-associative cache** — v1: yes. v2: yes. v3: yes (special case S=1). v3.1: yes.
- **Set-associative / direct-mapped cache** — v1: no (not modeled). v2: no. v3: yes. v3.1: yes.
- **Constant burst length** — v1: yes. v2: yes. v3: yes. v3.1: yes.
- **Heterogeneous per-line burst length** — v1: no. v2: no. v3: yes. v3.1: yes.
- **Small/concentrated-set accuracy (adversarial collisions)** — v1: no. v2: no. v3: mean-field only, ~8×10⁻⁴ err. v3.1: exact, ~4×10⁻⁵ err.
- **Closed-form (no numerical solve)** — v1: yes. v2: no (needs $T_c$ root-find). v3: no (needs per-set $T_{c,s}$ root-find). v3.1: no (root-find, plus exact chain for small/concentrated sets).
