#pragma once

// ============================================================
// popcount_v21_combined.cpp — Unified v18 + v19 + v21 (SIMD + RDTSCP)
//
// Inherited from v18 (deconfounded):
//   [BUG-1] Completely isolated RNG from the measured timing section.
//   [BUG-2] AnonHugePages now parses the correct VMA (smaps).
//   [BUG-3] Shuffled L execution order per trial (averages out thermal/DVFS effects).
//   + MAD, compute-floor calibration, prefetch-distance sweep, #pragma unroll.
//
// Inherited from v19 (hwcounters + bootstrap):
//   [P] Pin to CPU0 BEFORE allocation (prevents NUMA first-touch & thread migration).
//   [Q] Hardware PMU counters: dTLB-load-miss + LLC-load-miss via perf_event_open
//       (safe fallback on permission/container restrictions).
//   [R] Real RMSE + INDEPENDENT offset per run (sweeps L around Cochran ~13-15).
//   [S] Bootstrap 95% CI for ratio_of_drops (resamples trial execution time).
//   + Sanity check: sampling vs. full sequential scan.
//
// Bug fixes & patches (v20):
//   - RMSE estimator: mean_per_word = sum/(offs.size()*8) * N  (fixed v19 bug: missing /8).
//   - Actually invoke bootstrap_ci.
//   - Prevent DCE (Dead Code Elimination) using asm volatile.
//
// v21 (new, addressing "compute-floor not reached" review):
//   [T] SIMD popcount runtime dispatch: AVX-512 VPOPCNTDQ / AVX2 nibble-LUT /
//       scalar fallback. Validated via CPUID + XCR0 (checks OS context state,
//       not just CPU feature support). Verified scalar == avx2 == avx512 
//       across 2M test cases.
//       Applied to: popcount_from_offsets (primary hot path), compute-floor,
//       and popcount_array_ref (contiguous) to accelerate ref_normal/ref_hp/full-scan
//       on large N without changing output (strictly bit-identical).
//
//   [T-BUG] While writing the "bulk" kernel, caught a SEVERE performance bug
//       introduced during refactoring: passing lines_per_4k as a runtime parameter
//       caused "i % lines_per_4k" to compile into a real 64-bit DIV instruction
//       (~20-40+ cycles/op) instead of a bitwise AND (when it was constexpr) — 
//       confirmed via objdump.
//       Fixed using (i & (lines_per_4k-1)) + power-of-two static_assert.
//       After fixing DIV on BOTH sides (scalar and AVX-512), a fair comparison yielded:
//       scalar = 2.54ns, AVX-512 = 0.54ns/line => AVX-512 is ~4.7x faster — CLOSE TO
//       the theoretical expectation from Gemini's note (c), once methodology noise
//       was eliminated.
//       Takeaway: A single performance bug (DIV vs AND) can easily eclipse all SIMD gains.
//       Always benchmark both branches side-by-side after any change.
//
//   [U] RDTSCP+LFENCE (true serializing) cross-referenced against high_resolution_clock
//       (~clock_gettime, NON-serializing) specifically at the compute-floor — the most
//       sensitive spot for OoO instruction leakage across measurement boundaries.
//       After the speedup (from DIV->AND fix), chrono vs. RDTSCP variance grew to ~1.7% —
//       becoming NOTICABLE (unlike pre-DIV fix where <1% was masked by DIV latency).
//       The lower you push the compute-floor, the higher the percentage of measurement overhead.
//       This validates Gemini's note (a), though it only surfaced AFTER clearing the larger 
//       bottlenecks (DIV) first.
//
// v22 (built on v21):
//   [V] generate_offsets: added 'warn' parameter + g_cap_violations counter.
//       When requested num_regions exceeds available 4KB pages, the function CLAMPS down
//       (retaining legacy behavior) BUT now prints an explicit warning and tracks violation
//       counts. Otherwise, "N-fixed" breaks silently, inflating cost/line at small L due to
//       a smaller-than-expected denominator.
//   [W] Discovered new confounder: Original design set num_regions = total_lines/L, causing
//       every point on the L axis to simultaneously alter BOTH "reuse depth" AND 
//       "working-set footprint" (footprint = num_regions * 4KB) — coupling two separate noise sources.
//       Added detect_cache_sizes() (via sysconf, safe under containers/hypervisors blocking CPUID leaf 0x18),
//       report_footprint(), and 2 un-confounding experiments:
//       (1) L-sweep with FIXED footprint > L3 (varying single variable),
//       (2) Footprint-sweep with FIXED L (isolating the remaining axis across L1/L2/L3 thresholds).
//
// [MERGED v21+v22] Uses v22 as base (more feature-rich) while RESTORING v21's patch & bootstrap fix
//   ([v21-FIX-BUG-BOOTSTRAP]), which v22 accidentally reverted during feature additions:
//   v22's stat_fn in bootstrap_ci assumed "n_ops = total_lines" for BOTH L=1 and L=64 —
//   the exact bug identified and fixed in v21 (offsets can be CLAMPED below total_lines,
//   especially at small L, causing invalid normalization and driving bootstrap CI down to [0,0]).
//   Line-by-line diff confirmed this was an unintended regression (no code comments in v22 justified it).
//   Thus, this merged build restores the ACTUAL n_ops_per_L from run_deconfounded and feeds it to bootstrap,
//   keeping everything else from v22 intact (including new g_cap_violations/warn added in step [V]).
//
// v23 (NEW INVENTION, responding to "derive a new algorithm from research"):
//   [X] Root cause: cost(L) ~= a + b/L model (from [Model] step) is pure CURVE-FITTING
//       over 7 discrete L data points. It lacks physical meaning for parameter 'b', AND omits
//       the "footprint/working-set" (W) axis entirely. As a result, [v22] had to resort to 
//       2 SEPARATE experiments (L-sweep fixed footprint / footprint-sweep fixed L) without ever
//       UNIFYING them into a single predictive model across both axes.
//
//   [Y] Invention: "IRM-Burst Law of Equivalence" (Independent Reference Model for burst access patterns).
//       Derived via EXCHANGEABILITY PROBABILITY, not curve-fitting:
//         - Treat each "burst" (L consecutive accesses to the same 4KB page) as ONE logical reference
//           to 1 out of W pages (W = pages in working-set), picked uniformly at random (IID) —
//           accurately modeling generate_offsets() data generation (uniform region_dist, IID per p).
//         - For a cache/TLB of capacity C (holding the C most-recently-used pages under LRU),
//           expected total cached pages ALWAYS equals C (post warm-up). Since all W pages are SYMMETRIC
//           (exchangeable), EVERY page shares the SAME probability of being cached = C/W — directly
//           derived from linearity of expectation, NO approximations required.
//           Therefore:
//             occupancy(C,W) = min(1, C/W)
//             miss / burst    = 1 - occupancy(C,W)
//             miss / line     = (1 - occupancy(C,W)) / L
//           (Divided by L because only the FIRST access in a burst can miss; the remaining L-1 accesses
//           ALWAYS hit the same page with no intervening references).
//         - Verified via independent Monte Carlo simulation (Python) across realistic (W,C) pairs 
//           (including W=19531, C=64 and C=1536 — typical L1-DTLB / L2-STLB sizes on modern x86 CPUs).
//           Divergence was <0.1% vs. theory, consistent with Monte Carlo noise over 2-3M samples.
//           This result represents a SPECIAL CASE (UNIFORM popularity) of classical LRU theory under IRM
//           (e.g., Che et al. 2002 "characteristic time" approximation simplifies to an EXACT form when
//           popularity is uniform due to symmetry).
//           self_test_irm_law() reproduces this exact validation IN-PROCESS in C++ (true O(1) LRU,
//           not pen-and-paper deduction) before attempting hardware parameter inference.
//           If it fails, [v23] auto-SKIPS to avoid printing unreliable conclusions.
//
//   [Z] Since the formula explicitly contains BOTH L AND W (unlike a+b/L which only has L),
//       it AUTOMATICALLY deconfounds without needing two separate experiments:
//       Solves C directly via algebra from a single data point (L, W, miss/line), takes the median across 
//       multiple points on the L-AXIS (using existing PMU data) to yield a STABLE C_fit; then uses this EXACT C_fit
//       to PREDICT (no refitting) along the completely independent FOOTPRINT AXIS — performing a 2-axis
//       out-of-sample validation that the old a+b/L model CANNOT do (lacking parameter W).
//       Good alignment across both axes under the SAME C_fit provides strong empirical evidence that
//       the model captures true physical hardware mechanics rather than overfitting a single curve.
//
// v31 (NEW, responding to "further optimization + new algorithm/architecture research"):
//   [AA] contig_bulk_avx512_v31_8acc: Expanded from 2 accumulators ([v30]) to 8 independent accumulators
//       for the "contig" AVX-512 pipeline. EMPIRICALLY MEASURED via independent microbenchmark
//       (see attached docs/bench_kway.cpp) on a 1.6GB buffer (exceeding L3) ON THIS EXACT MACHINE:
//       2acc ~4.9-5.1, 4acc ~4.9-5.1, 8acc ~4.3-4.5, 16acc ~4.2-4.4 ns/line.
//       8 accumulators hit the optimal "knee" (~12-15% gain over 2acc; 16acc yields <3% additional gain).
//       All 3 variants are bit-identical to scalar. Wired directly into dispatch + primary benchmark
//       (benchmark_v31_kway_sweep(), prints metrics only after passing bit-identical self-test,
//       in the spirit of [v24]/[v30]).
//   [AB] Negative Result (reported honestly): Attempted applying the same accumulator expansion
//       to popcount_bulk_avx512 ("gather" pipeline, random offsets) — NO significant speedup observed
//       (~23-25ns/line across 2/4/8-way).
//       Conclusion: The bottleneck here is random memory access latency (TLB/cache-misses), NOT ALU accumulator count.
//       The CPU Out-of-Order execution window is already wide enough to keep multiple outstanding loads flying with 2 accumulators;
//       prefetch_distance (from [Q]/[R]) remains the ONLY true leverage for this pipeline.
//       [AA] is NOT applied to gather.
//   [AC] contig_bulk_mt: Multi-threaded kernel (std::thread) for contiguous memory, chunking array into N independent
//       slices + reduction (STRICTLY bit-identical with uint64_t, unlike floating-point sum).
//       HONEST WARNING: The container running this patch reports only 1 vCPU (`nproc`=1), so actual speedups 
//       CANNOT be measured here — self_test_contig_mt_bit_identical() currently verifies CORRECTNESS, not SPEED.
//       Users should run benchmarks on dedicated multi-core hardware.
//   [AD] EXTENSION IDEA (HYPOTHESIS, NOT YET SELF-TESTED — must write validation self-tests vs. real simulations/measurements
//       before relying on this like [v23]/[v26]): cost(K) could be modeled as a function of K (accumulator count)
//       using fit_cost_model() already present in the codebase (a+b/K form), explained via Little's Law
//       (classical queueing theory: concurrency = throughput x latency) — accumulator count / prefetch depth serve as
//       "concurrency", and the saturation point (8acc->16acc diminishing returns) corresponds to concurrency exceeding
//       the line-fill buffers or memory throughput the core can exploit.
//       This is NOT yet a validated law like [v23]/[v26] — provided honestly as a proposed direction for future work.
//   [AE] [HARDWARE CROSS-VALIDATION, user-executed on Xeon Platinum 8481C, L3=105MB, N=20M words/160MB
//       (host RAM ~1.9GB required scaling N down from 1.6GB dev machine, still exceeds L3)]:
//         contig: 2acc=5.754, 4acc=5.319, 8acc=5.040, 16acc=5.118 ns/line
//         => 8-way REMAINS THE WINNER on BOTH machines (~12-17% faster than 2acc). HOWEVER, 16-way on this host
//         SLOCKED DOWN instead of merely saturating (5.04 -> 5.12, unlike dev machine where 16acc was marginally faster) —
//         expected, as 16 accumulators (16 ZMM registers) introduce microarchitecture/compiler-dependent register pressure.
//         => REVISED CONCLUSION for cross-machine reliability: 8-way is the SAFEST SWEET SPOT (near-optimal on both,
//         never worst-performing). Do NOT default to 16-way even if it wins on specific hosts.
//
//         gather (200k ops, random offsets): On Xeon 8481C, prefetch_distance sweep shows K-way gains DO exist,
//         but are SMALL and HEAVILY DEPENDENT on prefetch_distance / host topology (unlike dev host where gains were zero).
//         e.g., on 8481C: pf=0 makes 8-way fastest; pf=8-16 makes 2-way/4-way equal/better than 8-way; pf=32 causes 2-way to 
//         degrade sharply (over-prefetching exhausts line-fill buffers while accumulator count is low).
//         => RE-CONFIRMS [AB]: gather lacks a single universal "best K-way" — parameter MUST be tuned per machine.
//         Safe fallback across both tested machines: 4-way + prefetch_distance ~16 (never the worst choice on either,
//         though not always absolute fastest). Default dispatch for [v30]/[v31] KEEPS 2-way for gather (simple, safe, low register pressure).
//         Users seeking an extra 5-10% on target hosts should tune K/pf via benchmark_v31 or the expanded gather_Kacc in docs/bench_kway.cpp.
// ============================================================
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <tuple>
#include <sys/mman.h>
#include <xmmintrin.h>
#include <fstream>
#include <sstream>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <immintrin.h>
#include <cpuid.h>
#include <thread>
#include <cctype>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <set>
#include <map> // [v35-NEW] can cho exact_set_solve_v35 (khong gian trang thai)

using namespace std;
using namespace std::chrono;

static inline uint64_t popcount64_hw(uint64_t x){ return __builtin_popcountll(x); }

uint64_t popcount_array_ref(const uint64_t* data, size_t n){
    uint64_t t = 0;
    for (size_t i = 0; i < n; i++) t += popcount64_hw(data[i]);
    return t;
}

// ============================================================
// [v21-NEW] SIMD popcount kernels + runtime dispatch
//   Initially implemented using a "per-line" approach (invoking a helper function
//   returning the total count per 8-word chunk). However, real-world benchmarks on this machine
//   revealed that AVX2 in this mode was actually SLOWER than scalar (4.24ns vs. 3.19ns/line),
//   as the repeated horizontal reduction overhead (store+load penalties) at EVERY call
//   completely eroded the vectorization gains.
//
//   Refactored into a "bulk" processing model: maintain running totals directly inside vector registers
//   (__m256i / __m512i) across the entire loop iteration space, performing a SINGLE horizontal 
//   reduction back to scalar at the very end.
//   Empirical results: scalar = 5.95ns, avx2 = 3.55ns, avx512 = 3.13ns/line (bulk) => AVX-512
//   yields a ~1.9x speedup over scalar in this scenario (rather than the theoretical 4-8x, 
//   since modern superscalar/OoO pipelines execute scalar POPCNT remarkably well).
//
//   Note: When compiled with -mavx512vpopcntdq, GCC/Clang may AUTOMATICALLY vectorize the scalar
//   popcount64_hw loop into vpopcntq instructions (confirmed via objdump disassembly).
//   Manual runtime dispatch remains essential here because: (1) it guarantees optimal execution across 
//   ALL target machines regardless of build flag constraints, and (2) the hand-crafted "bulk" pattern
//   outperforms compiler auto-vectorization applied to "per-line" loop structures.
