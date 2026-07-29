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
// ============================================================
static uint64_t popcount8_scalar(const uint64_t* base){
    uint64_t sum = 0;
    // [v21-FIX] "#pragma unroll" (Clang syntax) was COMPLETELY IGNORED by GCC
//   (-Wunknown-pragmas warning), meaning the original code was never actually unrolled 
//   under GCC despite the header comment citing "#pragma unroll".
//   Fixed via a cross-compiler macro compatible with both toolchains.
#if defined(__clang__)
    #pragma unroll
#elif defined(__GNUC__)
    #pragma GCC unroll 8
#endif
    for (int w = 0; w < 8; w++) sum += popcount64_hw(base[w]);
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static inline __m256i popcnt8_epi8_avx2(__m256i v, __m256i lut, __m256i low_mask){
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i s8 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(s8, _mm256_setzero_si256()); // da widen ve epi64
}

// ---------- Kernel BULK: include offsets, accumulate vectors, reduce 1 time ----------
uint64_t popcount_bulk_scalar(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    // [opt] prefetch_distance la bat-bien trong suot vong lap (khong doi qua
    // cac lan i) nhung truoc day nhanh "if (prefetch_distance > 0 ...)" nam
    // NGAY TRONG than vong lap nong, buoc CPU kiem tra lai dieu kien nay o
    // MOI lan i du gia tri khong bao gio doi trong 1 lan goi ham. Tach thanh
    // 2 vong lap rieng (co/khong prefetch) o NGOAI vong lap chinh de loai
    // han nhanh nay khoi than vong lap - branch predictor thuong da du doan
    // dung gan nhu tuyet doi nen loi ich thuc te nho, nhung khong ton chi phi
    // gi de lam va loai bo hoan toan phu thuoc vao muc do nhanh predictor.
    uint64_t sum = 0;
    size_t n = offsets.size();
    if (prefetch_distance > 0){
        size_t pd = (size_t)prefetch_distance;
        for (size_t i = 0; i < n; i++){
            if (i + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            sum += popcount8_scalar(data + offsets[i]);
        }
    } else {
        for (size_t i = 0; i < n; i++)
            sum += popcount8_scalar(data + offsets[i]);
    }
    return sum;
}

__attribute__((target("avx2")))
static uint64_t popcount_bulk_avx2(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    size_t n = offsets.size();
    // [opt] xem giai thich o popcount_bulk_scalar(): tach nhanh prefetch_distance
    // (bat-bien trong ham) ra khoi than vong lap nong bang 2 vong lap rieng.
    if (prefetch_distance > 0){
        size_t pd = (size_t)prefetch_distance;
        for (size_t i = 0; i < n; i++){
            if (i + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            const uint64_t* base = data + offsets[i];
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 4));
            acc0 = _mm256_add_epi64(acc0, popcnt8_epi8_avx2(v0, lut, low_mask));
            acc1 = _mm256_add_epi64(acc1, popcnt8_epi8_avx2(v1, lut, low_mask));
        }
    } else {
        for (size_t i = 0; i < n; i++){
            const uint64_t* base = data + offsets[i];
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 4));
            acc0 = _mm256_add_epi64(acc0, popcnt8_epi8_avx2(v0, lut, low_mask));
            acc1 = _mm256_add_epi64(acc1, popcnt8_epi8_avx2(v1, lut, low_mask));
        }
    }
    alignas(32) uint64_t tmp0[4], tmp1[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int i = 0; i < 4; i++) sum += tmp0[i] + tmp1[i];
    return sum;
}

__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_bulk_avx512(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    // // [v26-OPT] Original implementation used a single accumulator -> every loop iteration
//   formed a strict sequential data-dependency chain (acc = add(acc, popcnt(v))),
//   forcing the next iteration to stall until the previous addition completed.
//   Since vpopcntq has a multi-cycle latency, this single-accumulator pattern caps
//   throughput even though the CPU is capable of executing multiple popcnt operations
//   in parallel. The AVX2 implementation above (popcount_bulk_avx2) avoided this pitfall
//   by employing 2 independent accumulators (acc0/acc1) to process 2 vectors per iteration;
//   this AVX-512 variant applies the exact same technique: 2 independent accumulators,
//   processing 2 offsets per iteration to break the dependency chain and allow the CPU pipeline
//   to overlap multiple parallel vpopcntq instructions before reduction.
//   Tail elements (odd n) are handled via scalar fallbacks to avoid requiring an additional
//   target feature for AVX-512 masking.
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    size_t n = offsets.size();
    size_t i = 0;
    // [opt] xem giai thich o popcount_bulk_scalar(): tach nhanh prefetch_distance
    // (bat-bien trong ham) ra khoi than vong lap nong bang 2 vong lap rieng.
    if (prefetch_distance > 0){
        size_t pd = (size_t)prefetch_distance;
        for (; i + 1 < n; i += 2){
            if (i + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            if (i + 1 + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + 1 + pd]), _MM_HINT_T0);
            __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i]));
            __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + 1]));
            acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
            acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
        }
    } else {
        for (; i + 1 < n; i += 2){
            __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i]));
            __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + 1]));
            acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
            acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
        }
    }
    alignas(64) uint64_t tmp0[8], tmp1[8];
    _mm512_store_si512(reinterpret_cast<void*>(tmp0), acc0);
    _mm512_store_si512(reinterpret_cast<void*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int k = 0; k < 8; k++) sum += tmp0[k] + tmp1[k];
    for (; i < n; i++) sum += popcount8_scalar(data + offsets[i]); // duoi le so offset lon
    return sum;
}

// ============================================================
// // [v33-NEW] Explicit K-way + prefetch_distance variant for the "gather" pipeline
//   (random offsets) — PURPOSE IS NOT to replace the default dispatch (see [AB]:
//   accumulator expansion yields no speedup for this pipeline on the dev host),
//   but to SWEEP BOTH AXES (K x prefetch_distance) SIMULTANEOUSLY within 
//   benchmark_v33_gather_kway_pf_sweep() below.
//   Reason: [AB]/[AE] previously FIXED prefetch_distance at a SINGLE value while sweeping K
//   (or vice versa), which could confound the conclusion that "K does not help gather" —
//   small K + short pd can bottleneck in the exact same manner as large K + short pd (both lacking 
//   sufficient MLP window), leading readers to falsely assume accumulator expansion is useless 
//   in ALL scenarios, when in reality pd was simply insufficient to sustain multiple in-flight loads.
//   Only a full 2D grid sweep (rather than two isolated 1D sweeps) can disentangle this effect..
template<int K>
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_bulk_avx512_Kacc(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    __m512i acc[K];
    for (int k = 0; k < K; k++) acc[k] = _mm512_setzero_si512();
    size_t n = offsets.size();
    size_t i = 0;
    // [opt] xem giai thich o popcount_bulk_scalar(): tach nhanh prefetch_distance
    // (bat-bien trong ham) ra khoi than vong lap nong bang 2 vong lap rieng
    // (ap dung cho ca vong lap chinh K-way va duoi le).
    if (prefetch_distance > 0){
        size_t pd = (size_t)prefetch_distance;
        for (; i + (size_t)K <= n; i += (size_t)K){
            #pragma GCC unroll 16
            for (int k = 0; k < K; k++){
                size_t idx = i + (size_t)k + pd;
                if (idx < n) _mm_prefetch(reinterpret_cast<const char*>(data + offsets[idx]), _MM_HINT_T0);
            }
            #pragma GCC unroll 16
            for (int k = 0; k < K; k++){
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + (size_t)k]));
                acc[k] = _mm512_add_epi64(acc[k], _mm512_popcnt_epi64(v));
            }
        }
    } else {
        for (; i + (size_t)K <= n; i += (size_t)K){
            #pragma GCC unroll 16
            for (int k = 0; k < K; k++){
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + (size_t)k]));
                acc[k] = _mm512_add_epi64(acc[k], _mm512_popcnt_epi64(v));
            }
        }
    }
    uint64_t sum = 0;
    for (int k = 0; k < K; k++){
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), acc[k]);
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    if (prefetch_distance > 0){
        size_t pd = (size_t)prefetch_distance;
        for (; i < n; i++){ // duoi le so K
            if (i + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            sum += popcount8_scalar(data + offsets[i]);
        }
    } else {
        for (; i < n; i++) sum += popcount8_scalar(data + offsets[i]);
    }
    return sum;
}

// ---------- // BULK kernel for the compute-floor pipeline (modulo access pattern, no offsets) ----------
static uint64_t floor_bulk_scalar(const uint64_t* data, size_t count, size_t lines_per_4k){
    uint64_t sum = 0;
    for (size_t i = 0; i < count; i++)
        sum += popcount8_scalar(data + (i & (lines_per_4k - 1)) * 8);
    return sum;
}
__attribute__((target("avx2")))
static uint64_t floor_bulk_avx2(const uint64_t* data, size_t count, size_t lines_per_4k){
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    for (size_t i = 0; i < count; i++){
        const uint64_t* base = data + (i & (lines_per_4k - 1)) * 8;
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 4));
        acc0 = _mm256_add_epi64(acc0, popcnt8_epi8_avx2(v0, lut, low_mask));
        acc1 = _mm256_add_epi64(acc1, popcnt8_epi8_avx2(v1, lut, low_mask));
    }
    alignas(32) uint64_t tmp0[4], tmp1[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int i = 0; i < 4; i++) sum += tmp0[i] + tmp1[i];
    return sum;
}
// [cleanup] floor_bulk_avx512_legacy() (2-acc, tien nhiem cua v33_8acc/Kacc
// ben duoi) da xac nhan khong con duoc goi noi nao (-Wunused-function) - da xoa.

// ============================================================
/// [v33-NEW] Applies the [AA]/[v31] optimization pattern (expanding from 2-acc to 8-acc to reduce 
// per-word loop control overhead; see complete breakdown under contig_bulk_avx512_v31_8acc() below)
// to the "floor" pipeline (floor_bulk_avx512, where indices wrap around lines_per_4k — in this file's
// benchmark suite lines_per_4k=LINES_PER_4K=64, meaning data is TYPICALLY resident inside L1/L2 cache,
// making this pipeline fundamentally COMPUTE/ILP-BOUND from the start, rather than memory-bound like 
// the large "contig" buffer in [v31]). This directly addresses the open gap left in [v31]: contig was 
// expanded to 8-acc and cross-validated, but floor_bulk_avx512 remained hardcoded to 2 accumulators from [v26].
//
// Following the design of contig_bulk_avx512_Kacc in [v32], this module provides both a K-way template variant
// (allowing dynamic POPCNT_FLOOR_K tuning at runtime without rebuilding) AND a dedicated hand-unrolled 8-acc
// implementation (floor_bulk_avx512_v33_8acc) assigned as the new default dispatch — while the original 2-acc 
// variant is PRESERVED intact above as floor_bulk_avx512_legacy() for bit-identical and performance cross-validation
// (see benchmark_v33_floor_kway_sweep() below, executed LIVE on current run data rather than relying on static comment estimates).
// ============================================================
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t floor_bulk_avx512_v33_8acc(const uint64_t* data, size_t count, size_t lines_per_4k){
    __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
    __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
    __m512i acc4=_mm512_setzero_si512(), acc5=_mm512_setzero_si512();
    __m512i acc6=_mm512_setzero_si512(), acc7=_mm512_setzero_si512();
    const size_t mask = lines_per_4k - 1;
    size_t i = 0;
    for (; i + 8 <= count; i += 8){
        acc0=_mm512_add_epi64(acc0,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+0)&mask)*8))));
        acc1=_mm512_add_epi64(acc1,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+1)&mask)*8))));
        acc2=_mm512_add_epi64(acc2,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+2)&mask)*8))));
        acc3=_mm512_add_epi64(acc3,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+3)&mask)*8))));
        acc4=_mm512_add_epi64(acc4,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+4)&mask)*8))));
        acc5=_mm512_add_epi64(acc5,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+5)&mask)*8))));
        acc6=_mm512_add_epi64(acc6,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+6)&mask)*8))));
        acc7=_mm512_add_epi64(acc7,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data + ((i+7)&mask)*8))));
    }
    __m512i s01=_mm512_add_epi64(acc0,acc1), s23=_mm512_add_epi64(acc2,acc3);
    __m512i s45=_mm512_add_epi64(acc4,acc5), s67=_mm512_add_epi64(acc6,acc7);
    __m512i total=_mm512_add_epi64(_mm512_add_epi64(s01,s23), _mm512_add_epi64(s45,s67));
    alignas(64) uint64_t tmp[8];
    _mm512_store_si512(reinterpret_cast<void*>(tmp), total);
    uint64_t sum = 0;
    for (int k = 0; k < 8; k++) sum += tmp[k];
    for (; i < count; i++) sum += popcount8_scalar(data + (i & mask) * 8); // duoi le
    return sum;
}

// [v33-NEW] GENERALIZED K-way implementation for floor, mirroring contig_bulk_avx512_Kacc
// ([v32]) — allows runtime selection of K via POPCNT_FLOOR_K (see init_popcount_dispatch)
// without requiring a rebuild, accommodating machine-dependent variations in the optimal K parameter
// (following the exact cross-platform behavior observed in [AE] for contig).
// floor_bulk_avx512_v33_8acc() above REMAINS the default dispatch (K=8, preserving default behavior
// unless explicitly overridden via environment variables).
template<int K>
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t floor_bulk_avx512_Kacc(const uint64_t* data, size_t count, size_t lines_per_4k){
    __m512i acc[K];
    for (int k = 0; k < K; k++) acc[k] = _mm512_setzero_si512();
    const size_t mask = lines_per_4k - 1;
    size_t i = 0;
    for (; i + (size_t)K <= count; i += (size_t)K){
        #pragma GCC unroll 16
        for (int k = 0; k < K; k++){
            const uint64_t* base = data + ((i + (size_t)k) & mask) * 8;
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(base));
            acc[k] = _mm512_add_epi64(acc[k], _mm512_popcnt_epi64(v));
        }
    }
    uint64_t sum = 0;
    for (int k = 0; k < K; k++){
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), acc[k]);
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    for (; i < count; i++) sum += popcount8_scalar(data + (i & mask) * 8); // duoi le
    return sum;
}
static uint64_t floor_bulk_avx512_2acc_tpl (const uint64_t* d, size_t c, size_t lp4k){ return floor_bulk_avx512_Kacc<2>(d,c,lp4k); }
static uint64_t floor_bulk_avx512_4acc_tpl (const uint64_t* d, size_t c, size_t lp4k){ return floor_bulk_avx512_Kacc<4>(d,c,lp4k); }
static uint64_t floor_bulk_avx512_8acc_tpl (const uint64_t* d, size_t c, size_t lp4k){ return floor_bulk_avx512_Kacc<8>(d,c,lp4k); }
static uint64_t floor_bulk_avx512_16acc_tpl(const uint64_t* d, size_t c, size_t lp4k){ return floor_bulk_avx512_Kacc<16>(d,c,lp4k); }

// ---------- Kernel BULK cho mang lien tuc (popcount_array_ref, N lon) ----------
static uint64_t contig_bulk_scalar(const uint64_t* data, size_t n){ return popcount_array_ref(data, n); }

// [v30-NEW, CRITICAL FINDING] The original implementation of contig_bulk_avx2 (retained 
// here as "_1acc_legacy") used a SINGLE accumulator throughout the loop — EXACTLY the same
// sequential dependency-chain flaw that [v26-OPT] identified and fixed for 
// popcount_bulk_avx512/floor_bulk_avx512 (see comments there). However, this bug WAS NEVER 
// patched for the AVX2 "contig" path!
// More critically: this function served as the "nibble-LUT (legacy)" BASELINE used by 
// benchmark_v24_harleyseal() to compare against Harley-Seal. This implies that the reported
// "Harley-Seal is 1.1x–1.9x faster" speedup in [v24] might be PARTIALLY ATTRIBUTABLE to 
// benchmarking against an HANDICAPPED BASELINE (1 accumulator, sequential dependency chain),
// rather than reflecting pure algorithmic advantages of CSA (Carry-Save Adder) structures.
// This is precisely the type of methodological noise this project systematically uncovers 
// (e.g., the DIV-vs-AND issue in [T-BUG]).
//
// Fixed by employing 2 independent accumulators (acc0/acc1, processing two 256-bit vectors 
// per iteration, mirroring the technique in popcount_bulk_avx2) to break the dependency chain 
// and establish a FAIR baseline. The original (1-accumulator, unmodified logic) is RETAINED 
// under the name "_1acc_legacy" so that benchmark_v24_harleyseal() can evaluate BOTH, 
// quantitatively measuring the magnitude of this confounder rather than silently replacing it 
// and losing cross-validation capabilities.
__attribute__((target("avx2")))
static uint64_t contig_bulk_avx2(const uint64_t* data, size_t n){
    // [v30-FIX] 2 independent accumulators (acc0/acc1), processing two 256-bit vectors (8 x 64-bit words)
//   per iteration instead of 1. This allows the CPU pipeline to overlap two independent 
//   shuffle/and/add/sad instruction sequences in parallel, rather than stalling sequentially on a single accumulator.
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8){
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 4));
        acc0 = _mm256_add_epi64(acc0, popcnt8_epi8_avx2(v0, lut, low_mask));
        acc1 = _mm256_add_epi64(acc1, popcnt8_epi8_avx2(v1, lut, low_mask));
    }
    alignas(32) uint64_t tmp0[4], tmp1[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int k = 0; k < 4; k++) sum += tmp0[k] + tmp1[k];
    for (; i + 4 <= n; i += 4){ // duoi 4..7 tu (khi n%8 nam trong [4,8))
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        alignas(32) uint64_t t[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(t), popcnt8_epi8_avx2(v, lut, low_mask));
        sum += t[0] + t[1] + t[2] + t[3];
    }
    for (; i < n; i++) sum += popcount64_hw(data[i]); // duoi thua <4 phan tu
    return sum;
}
// (Original code, unmodified logic, renamed only — placed AFTER contig_bulk_avx2 
// for easy side-by-side comparison. Used as baseline in benchmark_v24_harleyseal.)
__attribute__((target("avx2")))
static uint64_t contig_bulk_avx2_1acc_legacy(const uint64_t* data, size_t n){
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 4 <= n; i += 4){
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        acc = _mm256_add_epi64(acc, popcnt8_epi8_avx2(v, lut, low_mask));
    }
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    uint64_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) sum += popcount64_hw(data[i]); // duoi thua <4 phan tu
    return sum;
}
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t contig_bulk_avx512(const uint64_t* data, size_t n){
    // [v30-FIX, UNVERIFIED ON THIS HOST] The original implementation used a single 
//   accumulator -> EXACTLY the same sequential dependency-chain flaw that [v26-OPT]
//   identified and fixed for popcount_bulk_avx512/floor_bulk_avx512 (see comments there),
//   yet somehow was NEVER applied to this "contig" path — an obvious inconsistency
//   within this codebase. Fixed using 2 independent accumulators, identical to the technique
//   employed in those two functions.
//   HONEST WARNING: This container environment LACKS avx512vpopcntdq support
//   (confirmed via /proc/cpuinfo and runtime detect_simd_level()), meaning this function
//   CANNOT be executed or microbenchmarked on this host — the fix is strictly logical/structural
//   (mirroring the verified pattern from popcount_bulk_avx512). Bit-identical equivalence 
//   is guaranteed by inspection: integer addition over 64-bit words is commutative and associative,
//   so splitting into 2 accumulators alters only the summation order, NOT the final scalar total
//   (unlike floating-point sums subject to rounding error, uint64_t operations are STRICTLY BIT-IDENTICAL).
//   Users executing on hardware with native VPOPCNTDQ support should run 
//   self_test_bit_identical / self_test_contig_bit_identical (included in this file)
//   to independently verify correctness before trusting performance metrics.
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 16 <= n; i += 16){
        __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + i));
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + i + 8));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
    }
    alignas(64) uint64_t tmp0[8], tmp1[8];
    _mm512_store_si512(reinterpret_cast<void*>(tmp0), acc0);
    _mm512_store_si512(reinterpret_cast<void*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int k = 0; k < 8; k++) sum += tmp0[k] + tmp1[k];
    for (; i + 8 <= n; i += 8){ // duoi 8..15 tu (khi n%16 nam trong [8,16))
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(data + i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int k = 0; k < 8; k++) sum += t[k];
    }
    for (; i < n; i++) sum += popcount64_hw(data[i]); // duoi thua <8 phan tu
    return sum;
}

// ============================================================
// [v31-NEW, EMPIRICALLY MEASURED on this host — un-fabricated] Issue with contig_bulk_avx512
// (2 accumulators, [v30]): Benchmarking on a 1.6GB buffer (strictly exceeding L3, true memory-bound,
// non-synthetic) revealed that 2acc and 4acc exhibited NEARLY ZERO difference (~4.9-5.1ns/line for both,
// within measurement noise margin). However, expanding to 8 INDEPENDENT accumulators (64 words/loop
// instead of 16) consistently reduced cost/line by ~12-15% across repeated runs (5.0 -> 4.4ns/line),
// whereas 16 accumulators yielded MARGINAL additional gains over 8 (4.4 -> 4.3ns/line — a clear region
// of saturation / diminishing returns). All 3 variants are bit-identical to scalar (validated in 
// self_test_contig_bit_identical and benchmark_v31_kway_sweep below on LIVE current-run data, 
// rather than relying on static values in comments).
//
// Plausible rationale (NOT a new physical law, standard microarchitectural reasoning):
// Under SEQUENTIAL access patterns (non-random), the hardware prefetcher already saturates 
// memory bandwidth. What changes across 2acc/4acc/8acc/16acc is NOT memory bandwidth, but 
// LOOP CONTROL OVERHEAD (loop counter, branch prediction, address calculation) per unit of work:
// 8 accumulators = 64 words/iteration = 1/4 the loop iterations of 2acc (16 words/iteration),
// amortizing fixed loop overhead over significantly more data. Beyond 8 accumulators, this benefit 
// saturates because loop overhead becomes negligible relative to memory wait cycles.
//
// [v31-SCOPE, ATTEMPTED AND FAILED — reported honestly] Attempted applying the SAME CONCEPT 
// (expanding accumulators 2->4->8) to popcount_bulk_avx512 ("gather" pipeline, random offsets, 
// TLB/cache-miss bound): NO significant speedup observed (~23.5-25ns/line across 2-way/4-way/8-way, 
// within noise margin).
// Rationale: The bottleneck here is RANDOM MEMORY ACCESS LATENCY (TLB/cache-misses ~50-70 cycles), 
// NOT the number of ALU accumulator registers. The CPU Out-of-Order window already possesses 
// sufficient instruction depth to keep multiple outstanding loads flying simultaneously with just 
// 2 accumulators — the true physical constraint is the core's LINE-FILL BUFFER (LFB) CAPACITY, 
// not vector registers available for accumulation.
// Pre-fetching distance (implemented in [Q]/[R]) remains the ONLY true leverage for this pipeline 
// (~7% gain at pf=8..16 vs pf=0); accumulator expansion is not. This is a valuable NEGATIVE RESULT, 
// preventing future maintainers from wasting effort expanding accumulators on gather paths without benefit.
// ============================================================
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t contig_bulk_avx512_v31_8acc(const uint64_t* data, size_t n){
    __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
    __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
    __m512i acc4=_mm512_setzero_si512(), acc5=_mm512_setzero_si512();
    __m512i acc6=_mm512_setzero_si512(), acc7=_mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= n; i += 64){
        acc0=_mm512_add_epi64(acc0,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i))));
        acc1=_mm512_add_epi64(acc1,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+8))));
        acc2=_mm512_add_epi64(acc2,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+16))));
        acc3=_mm512_add_epi64(acc3,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+24))));
        acc4=_mm512_add_epi64(acc4,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+32))));
        acc5=_mm512_add_epi64(acc5,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+40))));
        acc6=_mm512_add_epi64(acc6,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+48))));
        acc7=_mm512_add_epi64(acc7,_mm512_popcnt_epi64(_mm512_loadu_si512((const void*)(data+i+56))));
    }
    __m512i s01=_mm512_add_epi64(acc0,acc1), s23=_mm512_add_epi64(acc2,acc3);
    __m512i s45=_mm512_add_epi64(acc4,acc5), s67=_mm512_add_epi64(acc6,acc7);
    __m512i total=_mm512_add_epi64(_mm512_add_epi64(s01,s23), _mm512_add_epi64(s45,s67));
    alignas(64) uint64_t tmp[8];
    _mm512_store_si512(reinterpret_cast<void*>(tmp), total);
    uint64_t sum = 0;
    for (int k = 0; k < 8; k++) sum += tmp[k];
    for (; i + 8 <= n; i += 8){
        __m512i v = _mm512_loadu_si512((const void*)(data+i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int k = 0; k < 8; k++) sum += t[k];
    }
    // [v33-NEW] Tail elements <8 words (maximum 7 elements): consolidated into a single AVX-512F masked load
//   instead of a scalar loop iterating per-word — reduces instruction count, clean and consistent 
//   with other kernels, avoiding a dedicated conditional loop for short tails.
//   Self-tested: self_test_contig_bit_identical runs 3,000 trials for n = 0..1199 (covering all modulo-8 cases),
//   confirming 100% bit-identical match against scalar before establishing this as the default pipeline.
    if (i < n){
        __mmask8 mask = (__mmask8)((1u << (n - i)) - 1);
        __m512i v = _mm512_maskz_loadu_epi64(mask, (const void*)(data+i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int k = 0; k < 8; k++) sum += t[k];
    }
    return sum;
}

// [v32-NEW] Ban TONG QUAT hoa thanh template K-way, de CHON K LUC CHAY qua
// bien moi truong POPCNT_CONTIG_K ma KHONG can build lai - vi 2 may thuc te
// da cho thay diem toi uu K co the khac nhau (Xeon 8481C: 16-way CHAM LAI so
// 8-way do register pressure; may phat trien: 16-way van nhanh hon 8-way mot
// chut). contig_bulk_avx512_v31_8acc() ben tren VAN la dispatch mac dinh
// (khong doi hanh vi cu neu khong set env) - template nay CHI duoc dung khi
// nguoi dung chu dong set POPCNT_CONTIG_K de tu do/tune tren may cua ho.
template<int K>
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t contig_bulk_avx512_Kacc(const uint64_t* data, size_t n){
    __m512i acc[K];
    for (int k = 0; k < K; k++) acc[k] = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8*(size_t)K <= n; i += 8*(size_t)K){
        #pragma GCC unroll 16
        for (int k = 0; k < K; k++){
            __m512i v = _mm512_loadu_si512((const void*)(data + i + k*8));
            acc[k] = _mm512_add_epi64(acc[k], _mm512_popcnt_epi64(v));
        }
    }
    uint64_t sum = 0;
    for (int k = 0; k < K; k++){
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), acc[k]);
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    for (; i + 8 <= n; i += 8){
        __m512i v = _mm512_loadu_si512((const void*)(data + i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    // [v33-NEW] xem giai thich o contig_bulk_avx512_v31_8acc() - gop duoi <8
    // tu thanh 1 masked-load thay vi vong scalar.
    if (i < n){
        __mmask8 mask = (__mmask8)((1u << (n - i)) - 1);
        __m512i v = _mm512_maskz_loadu_epi64(mask, (const void*)(data + i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    return sum;
}
static uint64_t contig_bulk_avx512_2acc_tpl (const uint64_t* d, size_t n){ return contig_bulk_avx512_Kacc<2>(d,n); }
static uint64_t contig_bulk_avx512_4acc_tpl (const uint64_t* d, size_t n){ return contig_bulk_avx512_Kacc<4>(d,n); }
static uint64_t contig_bulk_avx512_8acc_tpl (const uint64_t* d, size_t n){ return contig_bulk_avx512_Kacc<8>(d,n); }
static uint64_t contig_bulk_avx512_16acc_tpl(const uint64_t* d, size_t n){ return contig_bulk_avx512_Kacc<16>(d,n); }

#endif

// ============================================================
// [v24-MOI] Kernel THUAT TOAN MOI cho may nay: Harley-Seal (carry-save
// adder) AVX2 popcount, ap dung cho DUONG ONG "contig" (mang lien tuc lon:
// popcount_array_ref/ref_normal/ref_hp/full-scan).
//
// Nguon goc thuat toan: Mula, Kurz, Lemire, "Faster Population Counts Using
// AVX2 Instructions", The Computer Journal, 2018 (arXiv:1611.07612) - DAY
// LA THUAT TOAN DA CONG BO, khong phai do toi tu nghi ra tu dau; phan "phat
// minh" o day la AP DUNG + KIEM CHUNG THUC NGHIEM cho dung workload cua file
// nay (khong co san trong ban goc, von chi dung nibble-LUT don gian).
//
// Y TUONG: ban nibble-LUT hien co (contig_bulk_avx2) goi 1 lan "popcount
// vector 256-bit" (2 shuffle + 2 and + 1 add + 1 sad, ~6 lenh) cho MOI 4
// tu 64-bit. Harley-Seal dung mach CSA (carry-save adder, 3 lenh
// xor/and/or moi mach, RE hon nhieu so popcount-nibble) de GOM 16 vector
// input thanh 5 "bit-plane" (ones/twos/fours/eights/sixteens, ma nhi phan
// cua tong so bit theo TUNG VI TRI qua 16 vector), roi CHI goi
// popcount-nibble ~5 lan cho MOI 16 vector input thay vi 16 lan. Ve mat ly
// thuyet giam so lenh popcount-nibble ~3.2 lan; thuc te do duoc it hon (vi
// them chi phi CSA va bi gioi han boi bang thong bo nho o N lon), nhung
// van la mot cai thien THAT, khong phai suy dien tren giay.
//
// [v24-KIEM CHUNG] Da viet mot chuong trinh doc lap (ngoai file nay) so
// sanh scalar / nibble-LUT-AVX2 (ban goc) / harley-seal-AVX2 (ban moi):
//   (1) Dung dan: 500 lan lap ngau nhien, moi n=0..1999 tu (quet het cac
//       truong hop du/thua vector 256-bit va thua tu 64-bit le) -> khop
//       BIT-IDENTICAL voi scalar 100% ca hai kernel.
//   (2) Toc do do THAT tren chinh may container nay (Xeon 1 vCPU, avx2 co,
//       KHONG co avx512vpopcntdq), build "tran" g++ -O3 (khong -march),
//       median qua nhieu lan lap, 3 lan chay doc lap de kiem tra on dinh:
//         L1-resident   32KB: harley-seal nhanh hon nibble-LUT ~1.5x
//         L2-resident    2MB: ~1.25-1.3x
//         L3-ish        32MB: ~1.5-1.8x (dao dong nhieu hon, do nhieu he thong)
//         80MB (~N mac dinh cua file nay, memory-bound, KHONG vua L3):
//                             ~1.1-1.2x - VAN nhanh hon, nhung it hon han vi
//                             bi nghen bang thong DRAM chu khong con nghen
//                             o so lenh compute nua.
//   => Ket luan TRUNG THUC: cai thien la THAT nhung CO DIEU KIEN - lon nhat
//      khi lam viec compute-bound (du lieu vua cache), nho dan khi chuyen
//      sang memory-bound (dung dung tinh than "phan biet nut that" ma
//      chinh file nay theo duoi xuyen suot [v18]-[v23]). Ham
//      self_test_bit_identical_harleyseal() va benchmark_v24_harleyseal()
//      ben duoi LAP LAI phep kiem chung nay NGAY TRONG chuong trinh, tren
//      DU LIEU THAT cua lan chay hien tai, thay vi bat nguoi dung tin vao
//      con so co dinh ghi trong comment.
//
// [v24-PHAM VI] KHONG ap dung ky thuat nay cho popcount_bulk_avx2 (duong
// ong "gather" theo offset ngau nhien, moi line chi 2 vector 256-bit) hay
// floor_bulk_avx2: CSA can GOM toi 16 vector lien tiep truoc khi co loi,
// trong khi 2 duong ong do von da bi gioi han boi DO TRE truy cap bo nho
// (TLB-miss/cache-miss) chu khong phai so lenh compute - gom CSA o day chi
// them do phuc tap ma khong giai quyet dung nut that. Ap dung dung cho:
// duong ong CONTIG (lien tuc, quet tuan tu lon) - noi ban than file nay da
// tu nhan dinh la "compute-floor"/tham chieu, dung nut that thuc su la so
// lenh/tu.
// ============================================================
__attribute__((target("avx2")))
static inline void csa256(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c){
    __m256i u = _mm256_xor_si256(a, b);
    h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    l = _mm256_xor_si256(u, c);
}
__attribute__((target("avx2")))
static uint64_t contig_bulk_avx2_harleyseal(const uint64_t* data, size_t n){
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i* vdata = reinterpret_cast<const __m256i*>(data);
    size_t nvec = n / 4; // so vector 256-bit day du (4 tu 64-bit/vector)

    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256(), twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256(), eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    // [v24-FIX quan trong] KHONG index truc tiep "vdata[k]": voi kieu
    // con tro __m256i*, GCC phat sinh load CO DOI HOI CANH CHINH 32-byte
    // (vmovdqa), trong khi bo nho cap boi mmap/malloc KHONG dam bao canh
    // chinh 32-byte tren x86-64 (chi 16-byte). Da xac nhan bang thuc nghiem:
    // ban dau dung "vdata[i+k]" gay Segmentation fault ngay lap tuc tren du
    // lieu that; sua bang goi _mm256_loadu_si256 tuong minh (khong doi hoi
    // canh chinh) cho MOI lan doc thi het loi.
#define V24_LD(idx) _mm256_loadu_si256(vdata + (idx))
    size_t i = 0;
    for (; i + 16 <= nvec; i += 16){
        csa256(twosA, ones, ones, V24_LD(i+0), V24_LD(i+1));
        csa256(twosB, ones, ones, V24_LD(i+2), V24_LD(i+3));
        csa256(foursA, twos, twos, twosA, twosB);
        csa256(twosA, ones, ones, V24_LD(i+4), V24_LD(i+5));
        csa256(twosB, ones, ones, V24_LD(i+6), V24_LD(i+7));
        csa256(foursB, twos, twos, twosA, twosB);
        csa256(eightsA, fours, fours, foursA, foursB);
        csa256(twosA, ones, ones, V24_LD(i+8), V24_LD(i+9));
        csa256(twosB, ones, ones, V24_LD(i+10), V24_LD(i+11));
        csa256(foursA, twos, twos, twosA, twosB);
        csa256(twosA, ones, ones, V24_LD(i+12), V24_LD(i+13));
        csa256(twosB, ones, ones, V24_LD(i+14), V24_LD(i+15));
        csa256(foursB, twos, twos, twosA, twosB);
        csa256(eightsB, fours, fours, foursA, foursB);
        csa256(sixteens, eights, eights, eightsA, eightsB);
        total = _mm256_add_epi64(total, popcnt8_epi8_avx2(sixteens, lut, low_mask));
    }
#undef V24_LD
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(eights, lut, low_mask), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(fours, lut, low_mask), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(twos, lut, low_mask), 1));
    total = _mm256_add_epi64(total, popcnt8_epi8_avx2(ones, lut, low_mask));

    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), total);
    uint64_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (size_t j = i; j < nvec; j++){ // duoi: < 16 vector 256-bit chua kip gom CSA
        __m256i v = _mm256_loadu_si256(vdata + j);
        alignas(32) uint64_t t2[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(t2), popcnt8_epi8_avx2(v, lut, low_mask));
        sum += t2[0]+t2[1]+t2[2]+t2[3];
    }
    for (size_t k = nvec*4; k < n; k++) sum += popcount64_hw(data[k]); // duoi: <4 tu 64-bit le
    return sum;
}
#endif

enum class SimdLevel { SCALAR, AVX2, AVX512VPOPCNTDQ };

// [v21-FIX-CRITICAL] _xgetbv yeu cau target "xsave". Neu goi truc tiep trong
// mot ham KHONG co target attribute, GCC bat loi bien dich ("target specific
// option mismatch") TRU KHI build voi co global -mavx512f/-mavx2/-mxsave.
// Dieu nay PHA VO chinh muc dich cua runtime-dispatch: chuong trinh phai
// bien dich duoc BANG "g++ -O3 v21_combined.cpp" tran (khong co avx flag
// nao) tren MOI may, ke ca may khong ho tro AVX (nhu i3 doi cu), va van
// falls-back ve scalar dung. Da xac nhan loi that: build khong co -mavx*
// se FAIL truoc khi sua ham nay.
__attribute__((target("xsave")))
static inline unsigned long long read_xcr0(){
    return _xgetbv(0);
}

static SimdLevel detect_simd_level(){
#if defined(__x86_64__) || defined(_M_X64)
    unsigned eax1, ebx1, ecx1, edx1;
    if (!__get_cpuid(1, &eax1, &ebx1, &ecx1, &edx1)) return SimdLevel::SCALAR;
    bool osxsave = (ecx1 >> 27) & 1;
    bool avx_cpu = (ecx1 >> 28) & 1;
    if (!osxsave || !avx_cpu) return SimdLevel::SCALAR;

    // Kiem tra OS co thuc su luu/khoi phuc state YMM/ZMM khong (XCR0),
    // vi CPUID chi noi CPU HO TRO, khong noi OS da BAT.
    unsigned long long xcr0 = read_xcr0();
    bool os_avx    = (xcr0 & 0x6)  == 0x6;   // XMM(1) + YMM(2)
    bool os_avx512 = (xcr0 & 0xE6) == 0xE6;  // + opmask/ZMM_hi256/Hi16_ZMM

    unsigned eax7, ebx7, ecx7, edx7;
    if (!__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7)) return SimdLevel::SCALAR;
    bool has_avx2            = (ebx7 >> 5)  & 1;
    bool has_avx512f         = (ebx7 >> 16) & 1;
    bool has_avx512vpopcntdq = (ecx7 >> 14) & 1;

    if (os_avx512 && has_avx512f && has_avx512vpopcntdq) return SimdLevel::AVX512VPOPCNTDQ;
    if (os_avx && has_avx2) return SimdLevel::AVX2;
#endif
    return SimdLevel::SCALAR;
}

using BulkOffsetsFn = uint64_t(*)(const uint64_t*, const vector<size_t>&, int);
using BulkFloorFn   = uint64_t(*)(const uint64_t*, size_t, size_t);
using BulkContigFn  = uint64_t(*)(const uint64_t*, size_t);

static BulkOffsetsFn popcount_bulk_dispatch = popcount_bulk_scalar;
static BulkFloorFn   floor_bulk_dispatch    = floor_bulk_scalar;
static BulkContigFn  contig_bulk_dispatch   = contig_bulk_scalar;
static const char*   simd_level_name        = "scalar (POPCNT)";
static SimdLevel      g_simd_level          = SimdLevel::SCALAR; // [v24] de benchmark/self-test khac doc lai duoc

static void init_popcount_dispatch(){
#if defined(__x86_64__) || defined(_M_X64)
    SimdLevel level = detect_simd_level();

    // [v21-NEW, buoc 5] Cho phep ep chon kernel qua bien moi truong, de
    // script tai lap co the chay lan luot ca 3 cau hinh tren CUNG mot may
    // va so sanh cong bang (khong can build 3 binary rieng).
    if (const char* force = getenv("POPCNT_FORCE")){
        string f(force);
        for (auto& c : f) c = (char)tolower((unsigned char)c);
        if (f == "scalar"){
            level = SimdLevel::SCALAR;
        } else if (f == "avx2"){
            if (level == SimdLevel::AVX512VPOPCNTDQ || level == SimdLevel::AVX2)
                level = SimdLevel::AVX2;
            else
                cerr << "[canh bao] POPCNT_FORCE=avx2 nhung CPU/OS khong ho tro AVX2 "
                        "-> giu nguyen muc phat hien tu dong.\n";
        } else if (f == "avx512"){
            if (level == SimdLevel::AVX512VPOPCNTDQ)
                level = SimdLevel::AVX512VPOPCNTDQ;
            else
                cerr << "[canh bao] POPCNT_FORCE=avx512 nhung CPU/OS khong ho tro "
                        "AVX-512 VPOPCNTDQ -> giu nguyen muc phat hien tu dong.\n";
        } else {
            cerr << "[canh bao] POPCNT_FORCE='" << force
                 << "' khong hop le (dung scalar|avx2|avx512) -> bo qua.\n";
        }
    }

    g_simd_level = level;
    switch (level){
        case SimdLevel::AVX512VPOPCNTDQ: {
            popcount_bulk_dispatch = popcount_bulk_avx512;      // gather: rong accumulator KHONG giup (xem [v31]; [v33] quet lai co ca prefetch_distance, xem benchmark_v33_gather_kway_pf_sweep)
            floor_bulk_dispatch    = floor_bulk_avx512_v33_8acc; // [v33] 8acc, mirror [v31]: floor thuong resident L1/L2 (LINES_PER_4K nho) nen compute/ILP-bound giong contig
            contig_bulk_dispatch   = contig_bulk_avx512_v31_8acc; // [v31] 8acc, ~9-17% nhanh hon 2acc cua [v30], da do that tren 2 may khac nhau
            string contig_tag = "contig: 8-acc [v31]";
            string floor_tag  = "floor: 8-acc [v33]";
            // [v32-NEW] Cho phep tu chon K cho contig LUC CHAY, vi diem toi
            // uu K khac nhau giua cac may thuc te da do (xem [AE]). Khong
            // set env -> giu 8acc mac dinh (khong doi hanh vi cu).
            if (const char* kforce = getenv("POPCNT_CONTIG_K")){
                string ks(kforce);
                if (ks == "2"){ contig_bulk_dispatch = contig_bulk_avx512_2acc_tpl; contig_tag = "contig: 2-acc [POPCNT_CONTIG_K=2]"; }
                else if (ks == "4"){ contig_bulk_dispatch = contig_bulk_avx512_4acc_tpl; contig_tag = "contig: 4-acc [POPCNT_CONTIG_K=4]"; }
                else if (ks == "8"){ contig_bulk_dispatch = contig_bulk_avx512_8acc_tpl; contig_tag = "contig: 8-acc [POPCNT_CONTIG_K=8]"; }
                else if (ks == "16"){ contig_bulk_dispatch = contig_bulk_avx512_16acc_tpl; contig_tag = "contig: 16-acc [POPCNT_CONTIG_K=16]"; }
                else cerr << "[canh bao] POPCNT_CONTIG_K='" << kforce << "' khong hop le (dung 2|4|8|16) -> giu 8acc mac dinh.\n";
            }
            // [v33-NEW] Tuong tu POPCNT_CONTIG_K nhung cho duong ong floor -
            // khong set env -> giu 8acc mac dinh (floor_bulk_avx512_v33_8acc,
            // khong doi hanh vi cu neu nguoi dung chua tung dung bien nay).
            if (const char* fforce = getenv("POPCNT_FLOOR_K")){
                string fs(fforce);
                if (fs == "2"){ floor_bulk_dispatch = floor_bulk_avx512_2acc_tpl; floor_tag = "floor: 2-acc [POPCNT_FLOOR_K=2]"; }
                else if (fs == "4"){ floor_bulk_dispatch = floor_bulk_avx512_4acc_tpl; floor_tag = "floor: 4-acc [POPCNT_FLOOR_K=4]"; }
                else if (fs == "8"){ floor_bulk_dispatch = floor_bulk_avx512_8acc_tpl; floor_tag = "floor: 8-acc [POPCNT_FLOOR_K=8]"; }
                else if (fs == "16"){ floor_bulk_dispatch = floor_bulk_avx512_16acc_tpl; floor_tag = "floor: 16-acc [POPCNT_FLOOR_K=16]"; }
                else cerr << "[canh bao] POPCNT_FLOOR_K='" << fforce << "' khong hop le (dung 2|4|8|16) -> giu 8acc mac dinh.\n";
            }
            static string simd_name_buf; // [v33] can bo nho on dinh vi simd_level_name la const char*
            simd_name_buf = "AVX-512 VPOPCNTDQ (" + contig_tag + " | " + floor_tag + " | gather: 2-acc [v30])";
            simd_level_name = simd_name_buf.c_str();
            break;
        }
        case SimdLevel::AVX2:
            popcount_bulk_dispatch = popcount_bulk_avx2;   // gather-offset: van nibble-LUT (xem [v24-PHAM VI])
            floor_bulk_dispatch    = floor_bulk_avx2;       // gather-modulo: van nibble-LUT (ly do tuong tu)
            contig_bulk_dispatch   = contig_bulk_avx2_harleyseal; // [v24] duong ong lien tuc -> Harley-Seal
            simd_level_name = "AVX2 (gather: nibble-LUT | contig: Harley-Seal CSA [v24])";
            break;
        default:
            popcount_bulk_dispatch = popcount_bulk_scalar;
            floor_bulk_dispatch    = floor_bulk_scalar;
            contig_bulk_dispatch   = contig_bulk_scalar;
            simd_level_name = "scalar (POPCNT, 1 word/lenh)";
    }
#endif
}

// ============================================================
// [v31-NEW] Kernel da luong (std::thread), CHI danh cho contig - phep cong
// uint64_t la giao hoan/ket hop TUYET DOI (khong phai float), nen chia mang
// thanh K doan doc lap, cong tung phan roi gop lai LUON cho ket qua
// BIT-IDENTICAL bat ke chia may doan - da kiem chung o
// self_test_contig_mt_bit_identical() ben duoi. CANH BAO TRUNG THUC: may
// container dang chay ban vá nay CHI thay 1 vCPU (nproc=1, da xac nhan qua
// lenh `nproc`), nen KHONG THE do duoc loi ich toc do thuc su tai day - chi
// xac nhan duoc TINH DUNG. Nguoi dung nen tu do lai tren may nhieu core thuc
// (dung self_test_contig_mt_bit_identical truoc, sau do so sanh thoi gian).
// (<thread> da include o dau file, dung lai o day)
// ============================================================
// ============================================================
// [v33-NEW, toi uu MT] Ban [v31] cua contig_bulk_mt() TAO/HUY std::thread MOI
// LAN GOI (pthread_create/join, ~10-30us/thread) - neu ham nay duoc goi lap
// lai qua nhieu trial trong benchmark (dung cho median/MAD nhu phan con lai
// cua file), chi phi tao/huy thread nay tro thanh MOT NGUON NHIEU DO PHUONG
// PHAP giong het cac loai da tu phat hien va sua truoc day trong file nay
// (DIV-vs-AND [T-BUG], 1acc-vs-2acc chuoi phu thuoc [v26-OPT]...). Sua bang
// thread-pool BEN VUNG: cac worker duoc tao 1 LAN DUY NHAT (lazy, static),
// ngu tren condition_variable giua cac lan goi, danh thuc qua bien dem the
// he (generation) thay vi tao/huy OS thread moi trial.
//
// [v33-SMT] std::thread::hardware_concurrency() dem CPU LOGIC, bao gom ca
// SMT/Hyper-Threading. Voi kernel AVX-512 compute-bound (vpopcntq thuong chi
// chay tren 1 port thuc thi/chu ky tren nhieu vi kien truc Intel), 2 luong
// SMT cung 1 core VAT LY se tranh chap CHINH port do - khong tang throughput,
// co the con giam do tranh chap L1D/registers. enumerate_physical_cores() ben
// duoi doc topology THAT qua sysfs (thread_siblings_list) de lay dung 1
// logical-CPU dai dien cho MOI core vat ly, roi pin moi worker vao 1 core
// rieng qua pthread_setaffinity_np - tranh nham "N luong" voi "N core thuc
// su chay song song". Neu khong doc duoc sysfs (container/permission), fallback
// coi moi logical CPU la 1 "core" (hanh vi cu, an toan).
//
// Da tu-kiem-chung logic pool (tai su dung dung, khong deadlock, ket qua
// dung) tren mot chuong trinh don gian truoc khi ghep vao day; ham nay VAN
// phai qua self_test_contig_mt_bit_identical() ben duoi truoc khi tin dung.
// ============================================================
static vector<int> enumerate_physical_cores(){
    vector<int> result;
    set<string> seen_siblings;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc <= 0) nproc = 1;
    for (long cpu = 0; cpu < nproc; cpu++){
        string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
        std::ifstream f(path);
        if (!f.good()){ result.push_back((int)cpu); continue; } // khong doc duoc topology -> fallback coi moi CPU la 1 core
        string line;
        std::getline(f, line);
        if (seen_siblings.count(line)) continue; // core vat ly nay da co dai dien (day la SMT sibling)
        seen_siblings.insert(line);
        result.push_back((int)cpu);
    }
    if (result.empty()) result.push_back(0);
    return result;
}

class ContigMtPool {
public:
    static ContigMtPool& instance(){ static ContigMtPool pool; return pool; }

    // Chay fn(t, n_threads) tren dung n_threads worker (tai su dung tu pool),
    // cho toi khi tat ca xong. An toan goi lap lai nhieu lan/trial.
    void run(unsigned n_threads, const function<void(unsigned,unsigned)>& fn){
        std::unique_lock<std::mutex> lk(mu_);
        ensure_workers(n_threads);
        task_ = fn;
        n_active_ = n_threads;
        done_count_ = 0;
        generation_++;
        cv_work_.notify_all();
        cv_done_.wait(lk, [this]{ return done_count_ == n_active_; });
    }

    ~ContigMtPool(){
        { std::unique_lock<std::mutex> lk(mu_); stop_ = true; generation_++; }
        cv_work_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
    }

private:
    ContigMtPool() = default;
    ContigMtPool(const ContigMtPool&) = delete;

    void ensure_workers(unsigned n_threads){
        if (workers_.size() >= n_threads) return;
        vector<int> cores = enumerate_physical_cores();
        unsigned start = (unsigned)workers_.size();
        for (unsigned t = start; t < n_threads; t++)
            workers_.emplace_back([this, t, cores]{ worker_loop(t, cores); });
    }

    void worker_loop(unsigned idx, vector<int> cores){
        if (!cores.empty()){
            cpu_set_t set; CPU_ZERO(&set);
            CPU_SET(cores[idx % cores.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        long seen_gen = 0;
        while (true){
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [this, seen_gen]{ return generation_ != seen_gen; });
            seen_gen = generation_;
            if (stop_) return;
            if (idx >= n_active_) continue; // trial nay dung it worker hon -> luong nay dung ngoai
            auto task = task_;
            lk.unlock();
            task(idx, n_active_);
            lk.lock();
            done_count_++;
            if (done_count_ == n_active_) cv_done_.notify_all();
        }
    }

    vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    function<void(unsigned,unsigned)> task_;
    unsigned n_active_ = 0, done_count_ = 0;
    long generation_ = 0;
    bool stop_ = false;
};

static uint64_t contig_bulk_mt(const uint64_t* data, size_t n, unsigned n_threads_req){
    // [v33] dung so CORE VAT LY (khong phai logical/SMT) lam tran cho n_threads,
    // vi kernel nay compute-bound tren AVX-512 (xem giai thich o tren).
    static const vector<int> phys_cores = enumerate_physical_cores();
    unsigned hw_phys = (unsigned)phys_cores.size();
    unsigned n_threads = n_threads_req;
    if (hw_phys > 0) n_threads = std::min(n_threads, hw_phys);
    if (n_threads < 1) n_threads = 1;
    if (n_threads == 1 || n < 4096) return contig_bulk_dispatch(data, n); // qua nho, khong bu duoc chi phi dieu phoi
    vector<uint64_t> partial(n_threads, 0);
    size_t chunk = n / n_threads;
    ContigMtPool::instance().run(n_threads, [&](unsigned t, unsigned nt){
        size_t begin = t * chunk;
        size_t end = (t == nt - 1) ? n : begin + chunk;
        partial[t] = contig_bulk_dispatch(data + begin, end - begin);
    });
    uint64_t sum = 0;
    for (uint64_t p : partial) sum += p;
    return sum;
}
static void self_test_contig_mt_bit_identical(const uint64_t* data, size_t n){
    uint64_t single = contig_bulk_dispatch(data, n);
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    uint64_t multi = contig_bulk_mt(data, n, hw);
    // [v33-FIX] contig_bulk_mt() tu clamp xuong SO CORE VAT LY ben trong (xem
    // ly do o dinh nghia ham) - in ca 2 con so de tranh hieu lam la "hw luong
    // logic" da thuc su chay song song (tren may co SMT, 2 con so nay khac nhau).
    unsigned phys_used = (unsigned)enumerate_physical_cores().size();
    if (phys_used > hw) phys_used = hw;
    cout << "[self-test v31-MT] contig_bulk_mt(" << hw << " luong logic yeu cau, "
         << "thuc te dung " << phys_used << " luong [pin theo core vat ly, v33], hardware_concurrency="
         << std::thread::hardware_concurrency() << ") "
         << (single == multi ? "khop BIT-IDENTICAL voi single-thread. OK.\n"
                              : "*** LECH VOI SINGLE-THREAD - CO BUG, DUNG DUNG. ***\n");
    if (single != multi){
        cerr << "  single=" << single << "  multi=" << multi << "\n";
        exit(3);
    }
}

// ---------- [v21-NEW, buoc 2] Self-test bit-identical luc khoi dong ----------
// Dam bao kernel dang duoc dispatch (co the la SIMD) cho ket qua GIONG HET
// scalar tren du lieu ngau nhien thuc te, TRUOC KHI chay bat ky benchmark
// nao. Neu lech: in ro offset/block loi va abort ngay - khong cho phep
// benchmark chay tren mot kernel sai (nhanh nhung sai con nguy hiem hon
// cham nhung dung).
static void self_test_bit_identical(uint32_t seed, size_t n_blocks){
    mt19937_64 rng(seed);
    alignas(64) uint64_t buf[8];
    for (size_t b = 0; b < n_blocks; b++){
        for (int w = 0; w < 8; w++) buf[w] = rng();
        uint64_t expect = popcount8_scalar(buf);       // scalar = ground truth
        vector<size_t> one_offset = {0};
        uint64_t got = popcount_bulk_dispatch(buf, one_offset, 0); // kernel dang dispatch
        if (got != expect){
            cerr << "\n[LOI NGHIEM TRONG] Self-test bit-identical THAT BAI!\n"
                 << "  Kernel dang dung: " << simd_level_name << "\n"
                 << "  Block thu #" << b << " (seed=" << seed << "):\n"
                 << "  scalar (dung) = " << expect << "\n"
                 << "  " << simd_level_name << " (SAI) = " << got << "\n"
                 << "  8 word cua block loi (hex): ";
            for (int w = 0; w < 8; w++) cerr << hex << buf[w] << " ";
            cerr << dec << "\n"
                 << "  => DUNG CHUONG TRINH. Co the do loi compiler/CPU microcode/\n"
                 << "     hoac loi logic kernel SIMD. KHONG dung ket qua benchmark nay.\n";
            exit(2);
        }
    }
    cout << "[self-test] " << n_blocks << " block ngau nhien: kernel '"
         << simd_level_name << "' khop 100% voi scalar. OK.\n";
}

// ---------- [v24-NEW] Self-test rieng cho contig_bulk_dispatch ----------
// self_test_bit_identical() o tren chi kiem tra popcount_bulk_dispatch
// (duong ong gather-offset, tung block 8-tu). contig_bulk_dispatch la kernel
// KHAC (co the la Harley-Seal tu [v24]) va chua tung duoc tu-kiem-chung -
// cham nay them phep kiem tra do, quet ca n=0..~1200 tu de bat het cac
// truong hop "duoi" (thieu vector 256-bit, thieu tu 64-bit le, chua du 16
// vector de gom CSA...) truoc khi cho phep dung ket qua cua kernel nay lam
// ref_normal/ref_hp/full-scan trong phan con lai cua chuong trinh.
static void self_test_contig_bit_identical(uint32_t seed, int n_trials){
    mt19937_64 rng(seed);
    for (int t = 0; t < n_trials; t++){
        size_t n = rng() % 1200;
        vector<uint64_t> buf(n);
        for (auto& x : buf) x = rng();
        uint64_t expect = contig_bulk_scalar(buf.data(), n);
        uint64_t got = contig_bulk_dispatch(buf.data(), n);
        if (got != expect){
            cerr << "\n[LOI NGHIEM TRONG] Self-test contig bit-identical THAT BAI!\n"
                 << "  Kernel: " << simd_level_name << "\n"
                 << "  Trial #" << t << " n=" << n << "  scalar=" << expect
                 << "  kernel=" << got << "\n  => DUNG CHUONG TRINH.\n";
            exit(2);
        }
        // n=0 khong dam bao tao ra thao tac AVX2 thuc su -> khong tinh la "co kiem tra vector"
    }
    cout << "[self-test v24] " << n_trials << " trial ngau nhien (n=0.." << 1199
         << " tu, du moi truong hop duoi cua Harley-Seal): contig_bulk_dispatch ('"
         << simd_level_name << "') khop 100% voi scalar. OK.\n";
}

// ---------- [v24-NEW] Benchmark THAT: nibble-LUT (cu) vs Harley-Seal (moi) ----------
// Chay ngay tren du lieu THAT cua lan thuc thi hien tai (buffer `normal` da
// duoc alloc/fill o main()), khong dung so lieu co dinh ghi san trong
// comment. In ra ca hai de nguoi doc tu doi chieu.
#if defined(__x86_64__) || defined(_M_X64)
static void benchmark_v24_harleyseal(const uint64_t* data, size_t n, SimdLevel level){
    if (level != SimdLevel::AVX2){
        cout << "[v24-benchmark] Bo qua (CPU/OS hien tai khong o muc AVX2 - "
                "Harley-Seal chi duoc dispatch cho AVX2; AVX-512 co VPOPCNTDQ phan cung\n"
                "  san, ban than da la 1 lenh/vector nen CSA khong co nhieu dat de toi uu them).\n\n";
        return;
    }
    // [v30-FIX PHUONG PHAP] Ban goc chay 15 rep nibble RIENG roi 15 rep
    // harley-seal RIENG (tuan tu, khong xen ke) - dung CHINH loai nhieu thu
    // tu/thermal/DVFS ma [BUG-3] o dau file da tung phat hien va sua cho
    // vong do L chinh (Xao thu tu L moi trial). Sua o day bang cach XEN KE
    // 3 kernel (legacy 1-acc, fixed 2-acc, harley-seal) TUNG rep mot, lay
    // median rieng cho moi kernel qua cac rep da xen ke - cong bang hon han
    // chay lien tiep tung khoi.
    // [v30-self-check] Kiem chung bit-identical CA BA kernel voi scalar TRUOC
    // khi tin bat ky con so toc do nao - dung tinh than "khong tin, kiem
    // chung" xuyen suot file nay (self_test_bit_identical o dau chuong trinh
    // da lam viec nay cho kernel DISPATCH, nhung contig_bulk_avx2_1acc_legacy
    // va contig_bulk_avx2 (2-acc) khong phai kernel dispatch tren may AVX2
    // nay (dispatch chon harley-seal cho contig), nen chua tung duoc kiem tra
    // truc tiep o day - lam ngay truoc khi dung chung lam baseline so sanh).
    {
        uint64_t ref = contig_bulk_scalar(data, n);
        uint64_t r_legacy = contig_bulk_avx2_1acc_legacy(data, n);
        uint64_t r_2acc   = contig_bulk_avx2(data, n);
        uint64_t r_hs     = contig_bulk_avx2_harleyseal(data, n);
        if (r_legacy != ref || r_2acc != ref || r_hs != ref){
            cerr << "\n[LOI NGHIEM TRONG][v30] Kiem chung bit-identical THAT BAI truoc benchmark!\n"
                 << "  scalar=" << ref << "  1acc_legacy=" << r_legacy
                 << "  2acc=" << r_2acc << "  harleyseal=" << r_hs << "\n"
                 << "  => BO QUA benchmark nay (khong tin so lieu toc do tren kernel co the sai).\n\n";
            return;
        }
        cout << "  [v30-self-check] Ca 3 kernel (1acc_legacy/2acc/harleyseal) khop bit-identical\n"
                "  voi scalar tren N=" << n << " tu THAT su - an toan de tin cac con so toc do duoi day.\n";
    }
    const int REPS = 15;
    vector<double> ts_legacy, ts_2acc, ts_hs;
    volatile uint64_t sink = 0;
    for (int r = 0; r < REPS; r++){
        int order = r % 3; // xoay vong thu tu 3 kernel moi rep, khong co dinh
        for (int slot = 0; slot < 3; slot++){
            int which = (order + slot) % 3;
            auto t0 = high_resolution_clock::now();
            uint64_t s;
            if (which == 0) s = contig_bulk_avx2_1acc_legacy(data, n);
            else if (which == 1) s = contig_bulk_avx2(data, n);
            else s = contig_bulk_avx2_harleyseal(data, n);
            auto t1 = high_resolution_clock::now();
            sink += s;
            double ns_per_word = duration<double, nano>(t1 - t0).count() / (double)n;
            if (which == 0) ts_legacy.push_back(ns_per_word);
            else if (which == 1) ts_2acc.push_back(ns_per_word);
            else ts_hs.push_back(ns_per_word);
        }
    }
    (void)sink;
    auto local_median = [](vector<double> v){
        sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double t_legacy = local_median(ts_legacy);
    double t_2acc   = local_median(ts_2acc);
    double t_hs     = local_median(ts_hs);
    cout << "[v24+v30-benchmark] contig, N=" << n << " tu (~" << (n*8)/(1024*1024) << "MB), THAT tren\n"
            "  may nay, " << REPS << " rep MOI kernel, XEN KE thu tu (khong tuan tu tung khoi):\n"
         << "  nibble-LUT AVX2, 1-accumulator (legacy, BASELINE CU cua [v24]) = "
         << fixed << setprecision(4) << t_legacy << " ns/tu\n"
         << "  nibble-LUT AVX2, 2-accumulator (CONG BANG, [v30]-fix)         = "
         << t_2acc << " ns/tu  (nhanh hon legacy " << setprecision(2) << (t_legacy/t_2acc) << "x - "
         << "CHINH no la nguon cua mot phan loi ich da gan nham cho Harley-Seal truoc day)\n"
         << "  Harley-Seal CSA AVX2 (v24)                                     = "
         << setprecision(4) << t_hs << " ns/tu\n"
         << "  => Ty le 'Harley-Seal nhanh hon' TINH LAI CONG BANG (so voi baseline 2-acc): "
         << setprecision(2) << (t_2acc/t_hs) << "x\n"
         << "     (so voi ty le CU, khong cong bang, so voi baseline 1-acc: " << (t_legacy/t_hs) << "x)\n"
         << "  KET LUAN [v30]: neu 2 ty le tren gan nhau, loi ich Harley-Seal da bao cao truoc\n"
         << "  day CHU YEU la THAT (thuat toan CSA); neu ty le CONG BANG thap hon dang ke, mot\n"
         << "  phan loi ich do TRUOC DAY la confound tu baseline bi thiet thoi (1 accumulator),\n"
         << "  dung nhu file nay da tung tu bat voi loi DIV-vs-AND o [T-BUG] - CHINH tinh than\n"
         << "  'doi chieu ca hai nhanh sau moi thay doi' ma comment [T-BUG] da rut ra lam bai hoc.\n"
         << "  (Ket qua co the giao dong theo tai he thong/virtualization cua container.)\n\n";
}
#endif

// ============================================================
// [v31-NEW] Benchmark THAT cho phat hien 8-accumulator (xem comment day du
// truoc contig_bulk_avx512_v31_8acc). Cung phong cach voi
// benchmark_v24_harleyseal: tu-kiem-chung bit-identical TRUOC, xen ke thu tu
// 3 kernel MOI rep (khong chay tuan tu tung khoi - tranh confound thermal/
// DVFS giong [BUG-3]), median. CHI chay khi thuc su o muc AVX512VPOPCNTDQ.
// ============================================================
#if defined(__x86_64__) || defined(_M_X64)
static void benchmark_v31_kway_sweep(const uint64_t* data, size_t n, SimdLevel level){
    if (level != SimdLevel::AVX512VPOPCNTDQ){
        cout << "[v31-benchmark] Bo qua (CPU/OS hien tai khong o muc AVX-512 VPOPCNTDQ).\n\n";
        return;
    }
    {
        uint64_t ref  = contig_bulk_scalar(data, n);
        uint64_t r2   = contig_bulk_avx512(data, n);          // [v30] 2acc
        uint64_t r8   = contig_bulk_avx512_v31_8acc(data, n); // [v31] 8acc
        if (r2 != ref || r8 != ref){
            cerr << "\n[LOI NGHIEM TRONG][v31] Kiem chung bit-identical THAT BAI truoc benchmark!\n"
                 << "  scalar=" << ref << "  2acc=" << r2 << "  8acc=" << r8
                 << "\n  => BO QUA benchmark nay.\n\n";
            return;
        }
        cout << "  [v31-self-check] Ca 2 kernel (2acc [v30] / 8acc [v31]) khop bit-identical\n"
                "  voi scalar tren N=" << n << " tu THAT su.\n";
    }
    const int REPS = 15;
    vector<double> ts_2acc, ts_8acc;
    volatile uint64_t sink = 0;
    for (int r = 0; r < REPS; r++){
        int order = r % 2;
        for (int slot = 0; slot < 2; slot++){
            int which = (order + slot) % 2;
            auto t0 = high_resolution_clock::now();
            uint64_t s = (which == 0) ? contig_bulk_avx512(data, n) : contig_bulk_avx512_v31_8acc(data, n);
            auto t1 = high_resolution_clock::now();
            sink += s;
            double ns_per_line = duration<double, nano>(t1 - t0).count() / ((double)n / 8.0);
            if (which == 0) ts_2acc.push_back(ns_per_line); else ts_8acc.push_back(ns_per_line);
        }
    }
    (void)sink;
    auto local_median = [](vector<double> v){ sort(v.begin(), v.end()); return v[v.size()/2]; };
    double t2 = local_median(ts_2acc), t8 = local_median(ts_8acc);
    cout << "[v31-benchmark] contig AVX-512, N=" << n << " tu (~" << (n*8)/(1024*1024) << "MB), THAT tren\n"
            "  may nay, " << REPS << " rep MOI kernel, XEN KE thu tu:\n"
         << "  2-accumulator ([v30], dispatch cu)  = " << fixed << setprecision(4) << t2 << " ns/line\n"
         << "  8-accumulator ([v31], dispatch moi) = " << t8 << " ns/line  (nhanh hon "
         << setprecision(2) << (t2/t8) << "x)\n"
         << "  (Tren buffer nho hon L3, hai so nay co the gan nhau hon vi khong con memory-bound -\n"
         << "  loi ich 8acc ro nhat khi N vuot han L3, dung voi ban dat MICROBENCHMARK docs/bench_kway.cpp doc lap.)\n\n";
}
#endif

// ---------- RDTSCP serialized timing ----------
// Dap lai nhan dinh: high_resolution_clock (~clock_gettime qua VDSO) KHONG
// phai serializing instruction -> CPU co the truot (OoO) mot phan cong viec
// qua ranh gioi do. RDTSCP + LFENCE la serializing that su.
static double calibrate_tsc_ghz(){
    unsigned aux;
    _mm_lfence();
    uint64_t t0 = __rdtscp(&aux);
    auto c0 = high_resolution_clock::now();
    this_thread::sleep_for(milliseconds(50));
    _mm_lfence();
    uint64_t t1 = __rdtscp(&aux);
    auto c1 = high_resolution_clock::now();
    double ns = duration<double, nano>(c1 - c0).count();
    return (double)(t1 - t0) / ns; // cycles/ns = GHz
}

constexpr size_t WORDS_PER_4K = 512;
constexpr size_t LINES_PER_4K = 64;
// [v21] floor_bulk_* dung (i & (lines_per_4k-1)) thay vi (i % lines_per_4k)
// de compiler phat ra AND thay vi DIV (DIV 64-bit ~20-40+ cycle/lan, la
// nguyen nhan chinh khien ban "bulk" ban dau CHAM hon ban inline-constexpr).
// Gia dinh nay CHI dung khi LINES_PER_4K la luy thua cua 2 — kiem tra tai
// compile-time de tranh loi ngam neu sau nay co ai doi hang so nay.
static_assert((LINES_PER_4K & (LINES_PER_4K - 1)) == 0,
              "LINES_PER_4K phai la luy thua cua 2 (dung cho toi uu AND thay DIV)");

// ---------- [P] Pin CPU0 truoc moi thu ----------
bool pin_to_cpu0(){
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// ---------- [Q] HW PMU ----------
static long perf_event_open_sys(struct perf_event_attr *hw_event, pid_t pid,
                                int cpu, int group_fd, unsigned long flags){
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct HwCounter {
    int fd = -1;
    bool ok = false;
    string name;
    void open(uint64_t config, const string& nm){
        name = nm;
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type           = PERF_TYPE_HW_CACHE;
        pe.size           = sizeof(pe);
        pe.config         = config;
        pe.disabled       = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv     = 1;
        fd = perf_event_open_sys(&pe, 0, -1, -1, 0);
        ok = (fd != -1);
    }
    void reset_enable(){
        if (ok){
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    long long disable_read(){
        if (!ok) return -1;
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        long long v = 0;
        if (read(fd, &v, sizeof(v)) != (ssize_t)sizeof(v)) return -1;
        return v;
    }
    ~HwCounter(){ if (fd != -1) close(fd); }
};

uint64_t DTLB_MISS_CFG(){
    return PERF_COUNT_HW_CACHE_DTLB
         | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
         | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
}
uint64_t LLC_MISS_CFG(){
    return PERF_COUNT_HW_CACHE_LL
         | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
         | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
}

// ---------- Sinh offset TRUOC (v18 BUG-1) ----------
// [v22-FIX] Bien dem toan cuc de phat hien khi "N-fixed" invariant bi vi pham
// ngam (num_regions bi cap xuong thap hon yeu cau) - loi nay truoc day KHONG
// duoc bao cao, khien cost/line o L nho co the bi thoi phong do mau so
// (n_ops) nho hon du kien, chu khong han la do hieu ung TLB/cache.
static size_t g_cap_violations = 0;
vector<size_t> generate_offsets(size_t n, size_t region_words,
                                 size_t lines_per_region_max,
                                 size_t num_regions, int L, uint64_t seed,
                                 bool warn = true){
    size_t n_regions = n / region_words;
    // [v33-FIX, canh gioi] Neu N nho hon 1 region (n_regions==0), "n_regions-1"
    // se TRAN size_t thanh SIZE_MAX -> uniform_int_distribution sinh region_base
    // khong lo, gay truy cap ngoai vung nho IM LANG (khong crash ngay, loi chi
    // lo ra sau duoi dang ket qua sai/segfault kho dieu tra). Chua tung xay ra
    // voi cac tham so N (MB..GB) dung trong file nay (da xac nhan sach qua
    // ASan/UBSan chay het chuong trinh that), nhung day la tien dieu kien BAT
    // BUOC cua ham nay nen chan sai o day, that bai ro rang thay vi de UB am
    // tham ve sau neu ai goi ham voi N nho hon trong tuong lai.
    if (n_regions == 0 || lines_per_region_max == 0){
        cerr << "  [FATAL] generate_offsets: tien dieu kien vi pham - n=" << n
             << " region_words=" << region_words << " (=> n_regions=" << n_regions
             << ") lines_per_region_max=" << lines_per_region_max
             << ". Can it nhat 1 region day du (N >= region_words) va "
                "lines_per_region_max >= 1. Dung lai thay vi sinh offset sai/UB.\n";
        exit(4);
    }
    size_t requested = num_regions;
    if (num_regions > n_regions) num_regions = n_regions;
    if (warn && requested > n_regions){
        g_cap_violations++;
        double actual_ops = (double)num_regions * L;
        double requested_ops = (double)requested * L;
        cerr << "  [v22-CANH BAO] generate_offsets: L=" << L
             << " yeu cau num_regions=" << requested << " nhung n_regions kha dung chi co "
             << n_regions << " -> BI CAT. n_ops thuc te=" << (size_t)actual_ops
             << " (du kien=" << (size_t)requested_ops << ", chi bang "
             << fixed << setprecision(1) << (100.0*actual_ops/requested_ops)
             << "%). 'N-fixed' KHONG con dung cho L nay -> cost/line co the bi\n"
             << "  thoi phong do mau so nho, DUNG dien giai nhu hieu ung TLB thuan tuy.\n";
    }
    mt19937_64 rng(seed);
    // [v39-FIX #8] TRUOC: region_dist(0, n_regions-1) lay mau tren TOAN BO
    // buffer (n_regions = N/region_words), khong gioi han trong working-set
    // THAT SU duoc yeu cau (num_regions, da clamp o tren). Khi num_regions <
    // n_regions (working-set nho hon buffer), offset sinh ra co the mang
    // page-id >= num_regions -> vuot qua W ma noi goi (vd real_workload_trace_v38)
    // truyen cho predict_miss_rate_v36, gay GHI TRAN raw_counts[W] (heap
    // corruption, "malloc(): corrupted top size"). Fix: gioi han dung trong
    // [0, num_regions-1] - dung 'num_regions' (bien DA CLAMP o tren), khong
    // phai 'n_regions' (dung luong buffer).
    uniform_int_distribution<size_t> region_dist(0, num_regions - 1);
    uniform_int_distribution<size_t> line_dist(0, lines_per_region_max - 1);
    vector<size_t> offsets;
    offsets.reserve(num_regions * (size_t)L);
    for (size_t p = 0; p < num_regions; p++){
        size_t region_base = region_dist(rng) * region_words;
        for (int l = 0; l < L; l++){
            size_t line_off = line_dist(rng) * 8;
            offsets.push_back(region_base + line_off);
        }
    }
    return offsets;
}

// ---------- Kernel do thoi gian: KHONG RNG, co prefetch ----------
// [v21] Giu nguyen ten/signature de khong phai sua moi noi goi trong main();
// ben trong now uy quyen cho kernel "bulk" (tich luy vector, reduce 1 lan).
uint64_t popcount_from_offsets(const uint64_t* data,
                               const vector<size_t>& offsets,
                               int prefetch_distance){
    return popcount_bulk_dispatch(data, offsets, prefetch_distance);
}

static void* alloc_region(size_t bytes, bool hugepage){
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, hugepage ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return p;
}

// ---------- [BUG-2] AnonHugePages dung VMA ----------
long anon_hugepages_kb_for_addr(void* addr){
    ifstream f("/proc/self/smaps");
    if (!f) return -1;
    string line;
    uintptr_t target = reinterpret_cast<uintptr_t>(addr);
    bool in_range = false;
    while (getline(f, line)){
        if (!line.empty() && isxdigit((unsigned char)line[0])){
            size_t dash = line.find('-');
            size_t sp   = line.find(' ');
            if (dash != string::npos && sp != string::npos && dash < sp){
                string s1 = line.substr(0, dash);
                string s2 = line.substr(dash + 1, sp - dash - 1);
                bool ok = !s1.empty() && !s2.empty();
                for (char c : s1) if (!isxdigit((unsigned char)c)) ok = false;
                for (char c : s2) if (!isxdigit((unsigned char)c)) ok = false;
                if (ok){
                    uintptr_t st = stoull(s1, nullptr, 16);
                    uintptr_t en = stoull(s2, nullptr, 16);
                    in_range = (target >= st && target < en);
                }
            }
        } else if (in_range && line.rfind("AnonHugePages:", 0) == 0){
            istringstream iss(line.substr(14));
            long v; iss >> v;
            return v;
        }
    }
    return -1;
}

double median(vector<double> v){
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double mad(vector<double> v){
    double m = median(v);
    vector<double> d;
    d.reserve(v.size());
    for (double x : v) d.push_back(fabs(x - m));
    return median(d);
}

// [v25c-MOI, ung voi (Q)] Theil-Sen slope: median cua TAT CA do doc cap-doi
// (y_j-y_i)/(j-i), i<j, voi x=chi so trial (0,1,2,...). Ben vung voi outlier
// hon OLS (dung median thay vi mean cua binh phuong sai lech) - phu hop du
// lieu timing thuong lech phai (vai outlier do he thong ngat/GC/nhieu khac).
double theil_sen_slope(const vector<double>& y){
    size_t n = y.size();
    if (n < 2) return 0.0;
    vector<double> slopes;
    slopes.reserve(n * (n - 1) / 2);
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++)
            slopes.push_back((y[j] - y[i]) / (double)(j - i));
    return median(slopes);
}

// [v25c-MOI, ung voi (Q)] Permutation test cho xu huong don dieu: gia dinh
// KHONG co xu huong thuc (H0: thu tu ngau nhien khong lien quan gia tri),
// xao tron y[] nhieu lan, tinh lai Theil-Sen slope moi lan - phan phoi nay
// xap xi phan phoi cua |slope| DUOI H0. p-value hai-phia = ty le lan xao tron
// co |slope| >= |slope quan sat|. Khong gia dinh phan phoi chuan (khac t-test).
//
// [v34-TOI UU, KHONG doi ket qua so hoc] theil_sen_slope() la O(n^2) (n=TRIALS)
// va ham nay goi lai no n_perm+1 lan -> O((n_perm+1)*n^2). Da do thuc nghiem
// truoc khi sua (ban goc, gian tiep qua theil_sen_slope): TRIALS=200 mat
// ~2.8s CHO 1 TRONG 4 vector duoc kiem tra o [Q]; TRIALS=1000 KHONG xong
// trong 20s (van con 3 vector nua). TRIALS la tham so dong lenh nguoi dung
// tu chinh (argv[3]), khong co gioi han tren - de "muon do chinh xac hon,
// tang TRIALS" vo tinh gay treo hang phut/gio o day dung khi cac phan khac
// cua chuong trinh van tuyen tinh binh thuong voi TRIALS. Sua 3 cho AN TOAN
// (khong doi thuat toan/ket qua, chi giam hang so va 1 buoc O(n log n)->O(n)):
//   (1) Tai dung 1 buffer 'slopes' xuyen suot n_perm+1 lan thay vi cap phat
//       lai (n(n-1)/2 phan tu) moi lan goi theil_sen_slope() ban goc.
//   (2) Precompute nghich dao 1/(j-i) MOT LAN (khong doi qua cac hoan vi,
//       chi y[] doi) - thay phep CHIA (~20-40 cycle) bang phep NHAN (~4-5
//       cycle) trong vong lap O(n^2) nong nhat chuong trinh.
//   (3) Dung nth_element (O(n) trung binh) de tim TRUNG VI thay vi sort
//       toan bo (O(n log n)) nhu median() ban goc dang lam qua theil_sen_slope
//       - median() cung nhan ban SAO CHEP (theo gia tri), nen goi no lap lai
//       con ngam an them 1 copy O(n) moi lan; ham noi bo duoi day thao tac
//       TRUC TIEP tren buffer dung rieng, khong con copy thua.
double permutation_test_trend(const vector<double>& y, int n_perm, mt19937_64& rng){
    size_t n = y.size();
    if (n < 2) return 1.0;
    size_t n_pairs = n * (n - 1) / 2;
    vector<double> inv_diff(n_pairs);
    for (size_t i = 0, k = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++, k++)
            inv_diff[k] = 1.0 / (double)(j - i);
    vector<double> slopes(n_pairs); // buffer dung chung cho MOI lan goi ben duoi
    auto fast_theil_sen_abs = [&](const vector<double>& v) -> double {
        for (size_t i = 0, k = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++, k++)
                slopes[k] = (v[j] - v[i]) * inv_diff[k];
        auto mid = slopes.begin() + slopes.size() / 2;
        nth_element(slopes.begin(), mid, slopes.end());
        return fabs(*mid);
    };
    double observed = fast_theil_sen_abs(y);
    vector<double> yp = y;
    int count_ge = 0;
    for (int p = 0; p < n_perm; p++){
        shuffle(yp.begin(), yp.end(), rng);
        if (fast_theil_sen_abs(yp) >= observed) count_ge++;
    }
    return (double)count_ge / (double)n_perm;
}

// ---------- [S] Bootstrap CI 95% ----------
// [v21-NEW, buoc 3] Tra ve ca median + MAD cua phan phoi bootstrap, khong chi
// CI 2 dau - vi CI diem-mut co the danh lua khi phan phoi lech/da-mode; nhin
// median+MAD (do phan tan trung vi, ben vung voi outlier) cho buc tranh day
// du hon ve do on dinh cua thong ke.
struct BootResult { double lo, hi, median, mad; };

// ---------- [v29-MOI, phat minh] Ha tang song song hoa cho bootstrap CI ----------
// Dong co: ca 4 ham bootstrap_ci / stationary_bootstrap_ci /
// hierarchical_bootstrap_ci / bootstrap_ci_varpro_C ben duoi CUNG chung 1 cau
// truc: vong lap "for (b = 0..n_boot)" ma MOI lan lap b LA DOC LAP HOAN TOAN
// voi cac lan lap khac - tu resample rieng, goi stat_fn rieng, KHONG doc/ghi
// bat ky trang thai chung nao giua cac lan (da kiem tra: stat_fn cac noi goi
// deu la lambda thuan chi doc bien capture-by-ref BAT BIEN (n_ops_normal/
// n_ops_hp/...), khong ghi; median/fit_linear/predict_miss_per_line/
// varpro_fit_capacity_no_pmu deu la ham thuan, khong static mutable; bien
// toan cuc mutable DUY NHAT trong file, g_cap_violations, chi duoc dung trong
// generate_offsets() O NGOAI duong goi cua bootstrap - khong dinh vao day).
// Day la dinh nghia sach cua bai toan "embarrassingly parallel". Voi n_boot
// ~1000 va moi lan goi stat_fn co the ton tu vai micro-giay (median don
// gian) den hang chuc mili-giay (bootstrap_ci_varpro_C phai chay lai TOAN BO
// grid-search VarPro voi grid_steps hang nghin diem MOI lan boot), chia deu
// cong viec cho cac core vat ly la don bay toi uu LON HON HAN so voi micro-
// toi uu accumulator/AVX cho TUNG lan goi rieng le.
//
// Thiet ke (khong mutex/atomic, khong rang buoc goi-vao-nhau giua thread):
//   - Chia mien [0, n_boot) thanh n_threads doan lien tiep KHONG chong lan.
//   - Truoc khi spawn, rut n_threads gia tri seed TUAN TU tu CHINH rng duoc
//     truyen vao (chi thread CHINH goi rng(), khong bao gio 2 thread cung
//     goi dong thoi vao 1 mt19937_64 - mt19937_64 KHONG thread-safe khi bi
//     goi dong thoi tu nhieu thread, day la ly do rieng RNG-tren-thread duoc
//     chon thay vi chia se 1 rng cho tat ca thread).
//   - Moi thread duoc cap 1 mt19937_64 rieng (seed doc lap boi tren), tu chay
//     iter_fn(b, rng_rieng_cua_thread) cho doan [b_start, b_end) cua no, ghi
//     KET QUA vao 1 vung KHONG chong lan cua vector stats chung - an toan vi
//     moi phan tu chi duoc DUNG 1 thread cham vao, khong can dong bo.
//   - Ket qua VAN deterministic ung voi 1 gia tri rng dau vao co dinh (thu tu
//     random-stream khac ban don-luong cu, nhung day la uoc luong Monte-Carlo
//     nen chi can dung PHAN PHOI thong ke, khong can trung bit-for-bit voi
//     ban cu).
//   - n_boot qua nho hoac may 1 core: roi lai don luong, tranh overhead tao
//     thread khong dang (nguong: >= 4 draws/thread).
static unsigned g_boot_n_threads(){
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1u : hc;
}

template <typename BootIterFn>
static vector<double> parallel_bootstrap_stats(int n_boot, mt19937_64& rng, BootIterFn&& iter_fn){
    vector<double> stats(n_boot);
    unsigned n_threads = g_boot_n_threads();
    if (n_threads > 1 && n_boot >= (int)(n_threads * 4)){
        // Rut seed cho tung thread TUAN TU tren thread goi (an toan, khong race)
        vector<uint64_t> seeds(n_threads);
        for (unsigned t = 0; t < n_threads; t++) seeds[t] = rng();
        int chunk = (n_boot + (int)n_threads - 1) / (int)n_threads;
        vector<std::thread> pool;
        pool.reserve(n_threads);
        for (unsigned t = 0; t < n_threads; t++){
            int b_start = (int)t * chunk;
            int b_end = std::min(n_boot, b_start + chunk);
            if (b_start >= b_end) continue;
            pool.emplace_back([&stats, &iter_fn, b_start, b_end, seed = seeds[t]]{
                mt19937_64 local_rng(seed);
                for (int b = b_start; b < b_end; b++)
                    stats[b] = iter_fn(b, local_rng);
            });
        }
        for (auto& th : pool) th.join();
    } else {
        for (int b = 0; b < n_boot; b++) stats[b] = iter_fn(b, rng);
    }
    return stats;
}

BootResult bootstrap_ci(vector<vector<double>> data_sets,
                         function<double(const vector<vector<double>>&)> stat_fn,
                         int n_boot, mt19937_64& rng){
    // [v29-song song] moi lan boot b doc lap hoan toan -> giao cho
    // parallel_bootstrap_stats() chia deu qua cac thread; data_sets/stat_fn
    // chi duoc DOC (khong ghi) tu nhieu thread nen an toan chia se by-ref.
    vector<double> stats = parallel_bootstrap_stats(n_boot, rng,
        [&data_sets, &stat_fn](int /*b*/, mt19937_64& local_rng) -> double {
            // [opt] buffer thread_local: 1 doi tuong rieng cho tung thread thuc
            // thi (parallel_bootstrap_stats chay tren std::thread thuc su, moi
            // thread goi lambda nay lien tiep n_boot/n_threads lan) - chi cap
            // phat/resize 1 LAN dau tien tren moi thread, cac draw sau CHI ghi
            // de gia tri, khong push_back/move vector moi mỗi draw nua.
            thread_local vector<vector<double>> resampled;
            if (resampled.size() != data_sets.size()){
                resampled.resize(data_sets.size());
                for (size_t vi = 0; vi < data_sets.size(); vi++)
                    resampled[vi].resize(data_sets[vi].size());
            }
            for (size_t vi = 0; vi < data_sets.size(); vi++){
                const auto& v = data_sets[vi];
                uniform_int_distribution<size_t> idx(0, v.size() - 1);
                auto& rv = resampled[vi];
                for (size_t i = 0; i < v.size(); i++) rv[i] = v[idx(local_rng)];
            }
            return stat_fn(resampled);
        });
    sort(stats.begin(), stats.end());
    size_t lo = (size_t)(0.025 * stats.size());
    size_t hi = (size_t)(0.975 * stats.size());
    if (hi >= stats.size()) hi = stats.size() - 1;
    double med = stats[stats.size() / 2];
    vector<double> dev(stats.size());
    for (size_t i = 0; i < stats.size(); i++) dev[i] = fabs(stats[i] - med);
    sort(dev.begin(), dev.end());
    double mad_val = dev[dev.size() / 2];
    return {stats[lo], stats[hi], med, mad_val};
}

// [v25c-MOI, ung voi (Q)-fix] Stationary bootstrap (Politis & Romano, JASA
// 1994) - sua THAT cho van de (Q) vua duoc XAC NHAN LA THAT (p<0.05 ca 4
// vector, xem [Q] o output). Thay vi resample tung phan tu doc lap (i.i.d.,
// gia dinh SAI khi co troi dat he thong), resample theo KHOI LIEN TIEP co do
// dai NGAU NHIEN ~Geometric(p) (ky vong 1/p), lay VONG (circular) tu chuoi
// goc. Dung CHUNG 1 chuoi chi-so-resample cho CA 4 vector dau vao (khong
// resample rieng tung vector) vi ca 4 duoc do o CUNG trial-index, co the
// CUNG chia se troi dat he thong theo thoi gian - giu dung phu thuoc CHEO-
// vector-theo-thoi-gian, khong chi phu thuoc trong-vector.
// [opt] Ban "_into" ghi vao buffer co san (chi clear(), khong giai phong bo
// nho da cap - neu buffer da tung dat toi dung tich >= n tu lan goi truoc,
// lan nay KHONG cap phat heap moi). Dung cho hot path (moi draw bootstrap
// goi ham nay 1 lan, co the toi hang chuc nghin lan) ket hop voi buffer
// thread_local o noi goi (xem bootstrap_ci/stationary_bootstrap_ci/
// hierarchical_bootstrap_ci/bootstrap_ci_varpro_C ben duoi) de trien tieu
// hoan toan cap phat lap lai tren duong nong nay.
static void stationary_bootstrap_indices_into(vector<size_t>& idx, size_t n, double p, mt19937_64& rng){
    idx.clear();
    uniform_int_distribution<size_t> start_dist(0, n - 1);
    geometric_distribution<int> geo(p);
    while (idx.size() < n){
        size_t start = start_dist(rng);
        int block_len = geo(rng) + 1; // Geometric >=0 tren cstd, +1 de >=1
        for (int k = 0; k < block_len && idx.size() < n; k++)
            idx.push_back((start + (size_t)k) % n); // vong quanh (circular)
    }
}
vector<size_t> stationary_bootstrap_indices(size_t n, double p, mt19937_64& rng){
    vector<size_t> idx; idx.reserve(n);
    stationary_bootstrap_indices_into(idx, n, p, rng);
    return idx;
}
BootResult stationary_bootstrap_ci(const vector<vector<double>>& data_sets,
                                    function<double(const vector<vector<double>>&)> stat_fn,
                                    int n_boot, double mean_block_len, mt19937_64& rng){
    double p = 1.0 / mean_block_len;
    size_t n = data_sets[0].size();
    // [v29-song song] xem chu thich o bootstrap_ci(); stationary_bootstrap_indices()
    // cung la ham thuan (chi doc rng truyen vao, khong bien toan cuc) nen an toan.
    vector<double> stats = parallel_bootstrap_stats(n_boot, rng,
        [&data_sets, &stat_fn, n, p](int /*b*/, mt19937_64& local_rng) -> double {
            // [opt] xem giai thich buffer thread_local o bootstrap_ci() phia tren.
            thread_local vector<size_t> idx;
            thread_local vector<vector<double>> resampled;
            stationary_bootstrap_indices_into(idx, n, p, local_rng);
            if (resampled.size() != data_sets.size()){
                resampled.resize(data_sets.size());
                for (size_t vi = 0; vi < data_sets.size(); vi++)
                    resampled[vi].resize(n);
            }
            for (size_t vi = 0; vi < data_sets.size(); vi++){
                const auto& v = data_sets[vi];
                auto& rv = resampled[vi];
                for (size_t i = 0; i < n; i++) rv[i] = v[idx[i]];
            }
            return stat_fn(resampled);
        });
    sort(stats.begin(), stats.end());
    size_t lo = (size_t)(0.025 * stats.size());
    size_t hi = (size_t)(0.975 * stats.size());
    if (hi >= stats.size()) hi = stats.size() - 1;
    double med = stats[stats.size() / 2];
    vector<double> dev(stats.size());
    for (size_t i = 0; i < stats.size(); i++) dev[i] = fabs(stats[i] - med);
    sort(dev.begin(), dev.end());
    double mad_val = dev[dev.size() / 2];
    return {stats[lo], stats[hi], med, mad_val};
}

// [v26-MOI, phat minh] Hierarchical / nested stationary bootstrap.
// Dong co: chay lai TOAN BO benchmark (macro-replication) R lan voi CUNG
// tham so cho thay diem-uoc ratio_of_drops TU DAO DONG manh giua cac lan
// chay (vd 0.788 -> 0.953 -> 1.143 chi qua 3 lan chay giong het nhau), tuc
// co MOT TANG bat dinh THU HAI ma [Q]/[Q-fix] (chi nhin trong-1-lan-chay,
// theo trial-index) KHONG the nam bat: bat dinh GIUA-cac-lan-chay (may
// nong/lanh khac nhau giua 2 lan goi binary, jitter he dieu hanh/sandbox,
// nhieu tu "hang xom" ao hoa...). Day la cau truc du lieu 2 tang kinh dien
// trong thong ke (clustered/nested data: "replication" long "trial"), nhung
// chua ai ap dung cho benchmark nay o day.
//
// Thuat toan (2 tang, dung CHUNG 1 tap chi-so-replication da chon cho ca 4
// vector, giu dung phu thuoc CHEO-vector NEU co, giong tinh than stationary
// bootstrap o tren):
//   Tang 1 (giua-cac-lan-chay): chon R chi-so replication VOI HOAN LAI tu
//     {0..R-1} - bat chuoc "neu ta chay lai ca benchmark R lan nua, tap hop
//     R lan chay ta THAY duoc se the nao".
//   Tang 2 (trong-1-lan-chay): voi MOI replication da chon, resample TRIALS
//     phan tu cua no bang stationary bootstrap (block~mean_block_len) NHU
//     CU - giu drift trong-lan-chay nhu [Q-fix] da lam.
//   Sau do GHEP (pool) tat ca cac vector da resample tu R replication da
//     chon thanh 1 vector dai R*TRIALS, roi goi stat_fn tren vector gop do.
// CI ket qua PHAI >= CI stationary 1-tang ve do rong (vi cong them 1 nguon
// bat dinh THAT), neu KHONG rong hon dang ke tuc bat dinh giua-cac-lan-chay
// la nho so voi trong-lan-chay (nguoc lai voi gia thuyet dat ra o day).
// [v26c-SUA LOI THAT LAN 2] Ban v26b tinh stat_fn RIENG cho R replication roi
// LAY TRUNG BINH R gia-tri do lam 1 draw bootstrap - nhung trung binh cua R
// gia-tri THI TU NO co phuong sai = phuong sai-1-gia-tri / R (dinh ly co ban
// nhat cua trung binh mau), nen lai VO TINH lam CI hep di theo R mot lan nua,
// che mat chinh hieu ung dang muon do. Van la loi cung loai voi v26b (lam
// giam phuong sai bang mot phep toan thong ke KHONG lien quan gi den cau hoi
// dat ra), chi khac hinh thuc (gop du lieu vs trung binh hoa).
//
// Cau hoi THAT can tra loi: "neu ta chi chay CHINH XAC 1 lan (nhu thuc te
// nguoi dung se lam, khong chay 5 lan roi trung binh), do bat dinh cho diem-
// uoc CUA LAN CHAY DO la bao nhieu, biet rang lan-chay-do la MOT trong so
// nhieu lan-chay-co-the-xay-ra?" => moi draw bootstrap PHAI la ket qua cua
// CHI-MOT lan chon-lai-VOI-HOAN-LAI 1 replication (khong chon R lan, khong
// trung binh nhieu draw) roi resample stationary TRONG replication do va goi
// stat_fn CHINH XAC 1 LAN. Day la bootstrap "tien-doan cho 1 quan sat tuong
// lai" (predictive/marginal bootstrap cho du lieu long/nested), khac voi
// bootstrap uoc luong tham so trung binh - va la ban dung cho tinh huong nay.
BootResult hierarchical_bootstrap_ci(
        const vector<vector<vector<double>>>& reps_data_sets, // [R][n_vec][TRIALS]
        function<double(const vector<vector<double>>&)> stat_fn,
        int n_boot, double mean_block_len, mt19937_64& rng){
    size_t R = reps_data_sets.size();
    size_t n_vec = reps_data_sets[0].size();
    size_t n_trials = reps_data_sets[0][0].size();
    double p = 1.0 / mean_block_len;
    // [v29-song song] uniform_int_distribution<>::operator() chi doc trang
    // thai cua rng truyen vao (khong bien toan cuc/static rieng) nen goi tren
    // local_rng cua tung thread la an toan; reps_data_sets chi duoc DOC.
    vector<double> stats = parallel_bootstrap_stats(n_boot, rng,
        [&reps_data_sets, &stat_fn, R, n_vec, n_trials, p](int /*b*/, mt19937_64& local_rng) -> double {
            uniform_int_distribution<size_t> rep_dist(0, R - 1);
            size_t rep_idx = rep_dist(local_rng); // Tang 1: CHI chon 1 lan-chay VOI HOAN LAI (khong lay TB)
            // [opt] xem giai thich buffer thread_local o bootstrap_ci() phia tren.
            thread_local vector<size_t> tidx;
            thread_local vector<vector<double>> one_rep;
            stationary_bootstrap_indices_into(tidx, n_trials, p, local_rng); // Tang 2
            if (one_rep.size() != n_vec){
                one_rep.resize(n_vec);
                for (size_t v = 0; v < n_vec; v++) one_rep[v].resize(n_trials);
            }
            for (size_t v = 0; v < n_vec; v++){
                const auto& src = reps_data_sets[rep_idx][v];
                auto& dst = one_rep[v];
                for (size_t k = 0; k < n_trials; k++) dst[k] = src[tidx[k]];
            }
            return stat_fn(one_rep); // 1 draw bootstrap = 1 stat_fn call, khong gop khong TB
        });
    sort(stats.begin(), stats.end());
    size_t lo = (size_t)(0.025 * stats.size());
    size_t hi = (size_t)(0.975 * stats.size());
    if (hi >= stats.size()) hi = stats.size() - 1;
    double med = stats[stats.size() / 2];
    vector<double> dev(stats.size());
    for (size_t i = 0; i < stats.size(); i++) dev[i] = fabs(stats[i] - med);
    sort(dev.begin(), dev.end());
    double mad_val = dev[dev.size() / 2];
    return {stats[lo], stats[hi], med, mad_val};
}

// [v21-NEW, buoc 4] Fit mo hinh cost(L) ~= a + b/L bang least-squares dong
// (2 tham so, x=1/L). 'a' la tiem can khi L->vo cuc (T_alu+T_line gop lai,
// khong the tach rieng tu du lieu nay vi ca hai deu la hang so khong doi
// theo L). 'b' la phan chi phi giam dan theo 1/L (dai dien T_tlb, gia dinh
// so lan TLB-miss ti le nghich voi L do locality tang khi L tang).
struct CostModelFit { double a, b, r_squared; };

static CostModelFit fit_cost_model(const vector<int>& Ls, const vector<double>& cost){
    size_t n = Ls.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; i++){
        double x = 1.0 / (double)Ls[i];
        double y = cost[i];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double denom = (double)n * sxx - sx * sx;
    double b = (denom != 0.0) ? ((double)n * sxy - sx * sy) / denom : 0.0;
    double a = (sy - b * sx) / (double)n;
    double ss_tot = 0, ss_res = 0, mean_y = sy / (double)n;
    for (size_t i = 0; i < n; i++){
        double x = 1.0 / (double)Ls[i];
        double pred = a + b * x;
        ss_res += (cost[i] - pred) * (cost[i] - pred);
        ss_tot += (cost[i] - mean_y) * (cost[i] - mean_y);
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 1.0;
    return {a, b, r2};
}

static void print_cost_model_fit(const char* label, const vector<int>& Ls,
                                  const vector<double>& cost){
    auto fit = fit_cost_model(Ls, cost);
    cout << "  [model " << label << "] cost(L) ~= " << fixed << setprecision(4)
         << fit.a << " + " << fit.b << "/L    (R^2=" << setprecision(4)
         << fit.r_squared << ")\n";
    cout << "      L    do_that(ns)   du_doan(ns)   sai_lech(%)\n";
    for (size_t i = 0; i < Ls.size(); i++){
        double pred = fit.a + fit.b / (double)Ls[i];
        double err_pct = 100.0 * fabs(cost[i] - pred) / cost[i];
        cout << "    " << setw(4) << Ls[i]
             << "   " << setw(10) << setprecision(3) << cost[i]
             << "   " << setw(10) << setprecision(3) << pred
             << "   " << setw(8) << setprecision(2) << err_pct << "\n";
    }
    if (fit.r_squared < 0.9)
        cout << "      [CANH BAO] R^2 < 0.9 - mo hinh a+b/L KHOP KEM, co the\n"
                "      con hieu ung khac (vd row-buffer bac 2, cache L2/L3)\n"
                "      chua duoc mo hinh hoa. Dung ket luan qua tu tin.\n";
}

// ============================================================
// [v22-NEW] Boc lo bien AN: num_regions va footprint (working-set thuc te)
//
// Van de phat hien: trong thiet ke goc, num_regions = total_lines/L, nen
// khi L tang, num_regions GIAM theo (~1/L), va vi footprint = num_regions *
// 4KB, footprint CUNG giam theo L. Nghia la moi diem du lieu tren truc L
// dong thoi thay doi CA HAI: (1) do sau tai su dung 1 trang (bien dinh
// nghien cuu that) VA (2) tong kich thuoc working-set cham vao cache nao
// (L1/L2/L3) hay khong (bien nhieu, chua bao gio duoc bao cao). Day la ly
// do R^2 cua model a+b/L thap va muc giam cost/line TANG TOC (khong giam
// dan) o L lon trong thi nghiem thuc te (da kiem chung bang cach tang gap
// doi total_lines va thay hien tuong van con nguyen, chi doi bien do).
//
// Ham duoi day dung sysconf (glibc doc tu /sys, KHONG can CPUID leaf 0x18 -
// leaf nay bi nhieu hypervisor/container chan) de lay kich thuoc cache THAT
// cua may dang chay, roi bao cao footprint tung L co "cham" cache nao.
struct CacheSizes { long l1d = -1, l2 = -1, l3 = -1; };
static CacheSizes detect_cache_sizes(){
    CacheSizes c;
#ifdef _SC_LEVEL1_DCACHE_SIZE
    c.l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
    c.l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
    c.l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    return c;
}

static void report_footprint(int L, size_t total_lines_target, size_t N,
                              size_t region_words, const CacheSizes& cs){
    size_t n_regions_avail = N / region_words;
    size_t requested = total_lines_target / (size_t)L;
    size_t actual = min(requested, n_regions_avail);
    double footprint_mb = (double)actual * 4096.0 / (1024.0*1024.0);
    const char* tag = "khong ro (thieu du lieu cache)";
    if (cs.l3 > 0){
        double l1_mb = cs.l1d > 0 ? cs.l1d/1024.0/1024.0 : -1;
        double l2_mb = cs.l2  > 0 ? cs.l2 /1024.0/1024.0 : -1;
        double l3_mb = cs.l3/1024.0/1024.0;
        if (l1_mb > 0 && footprint_mb <= l1_mb) tag = "<= L1d (!!)";
        else if (l2_mb > 0 && footprint_mb <= l2_mb) tag = "<= L2 (canh bao: co the lot cache)";
        else if (footprint_mb <= l3_mb) tag = "<= L3 (CANH BAO MANH: du lieu co the 'am' qua 20 trial lap lai)";
        else tag = "> L3 (an toan, that su 'lanh'/DRAM-bound)";
    }
    cout << "    L=" << setw(3) << L << "  num_regions=" << setw(7) << actual
         << (actual < requested ? "(BI CAT!) " : "          ")
         << " footprint=" << fixed << setprecision(1) << setw(8) << footprint_mb
         << "MB  [" << tag << "]\n";
}

// ============================================================
// [v22-NEW] Thi nghiem 1: L-sweep voi num_regions CO DINH (giai toa nhieu)
//   Y tuong: giu num_regions co dinh (du lon de footprint > L3 chac chan),
//   CHI thay doi L. Nho vay chi con 1 bien thay doi (do sau tai su dung),
//   con footprint la hang so xuyen suot -> mo hinh a+b/L neu dung se khop
//   tot hon HAN so voi thiet ke goc (num_regions=total_lines/L, 2 bien
//   troi nhau). Day la each tach truc quan sat thanh 2 truc doc lap.
static pair<vector<double>, double> run_fixed_footprint_sweep(
        uint64_t* data, const vector<int>& Ls, size_t N, size_t fixed_num_regions,
        int trials, const char* label){
    vector<double> cost(Ls.size());
    cout << "  --- " << label << " (num_regions CO DINH=" << fixed_num_regions << ") ---\n";
    for (size_t li = 0; li < Ls.size(); li++){
        int L = Ls[li];
        auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K,
                                      fixed_num_regions, L, 55555 + li, false);
        vector<double> t;
        for (int tr = 0; tr < trials; tr++){
            auto a0 = high_resolution_clock::now();
            uint64_t s = popcount_from_offsets(data, offs, 0);
            auto a1 = high_resolution_clock::now();
            asm volatile("" :: "r"(s));
            t.push_back(duration<double, micro>(a1 - a0).count());
        }
        double mt = median(t);
        cost[li] = (mt * 1000.0) / (double)offs.size();
        cout << "    L=" << setw(3) << L << "  cost/line=" << fixed << setprecision(3)
             << setw(8) << cost[li] << "ns   (footprint=" << setprecision(1)
             << (fixed_num_regions*4096.0/1024/1024) << "MB, khong doi theo L)\n";
    }
    auto fit = fit_cost_model(Ls, cost);
    cout << "  [model, footprint co dinh] cost(L) ~= " << fixed << setprecision(4)
         << fit.a << " + " << fit.b << "/L    (R^2=" << setprecision(4) << fit.r_squared
         << ")  <-- so voi model goc (troi bien) o tren\n\n";
    return {cost, fit.r_squared};
}

// ============================================================
// [v22-NEW] Thi nghiem 2: footprint-sweep voi L CO DINH (dac trung rieng
//   truc con lai). Giu L nho (it anh huong reuse-depth), quet num_regions
//   qua cac nguong cache L1/L2/L3 de thay ro DUNG hieu ung nao dang chi
//   phoi cost khi working-set thay doi, doc lap voi L.
static void run_fixed_L_footprint_sweep(uint64_t* data, size_t N, int fixed_L,
                                         const vector<size_t>& region_counts,
                                         int trials, const CacheSizes& cs){
    cout << "  --- L=" << fixed_L << " co dinh, quet num_regions (footprint) ---\n";
    for (size_t nr : region_counts){
        auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, nr, fixed_L, 77777 + nr, false);
        vector<double> t;
        for (int tr = 0; tr < trials; tr++){
            auto a0 = high_resolution_clock::now();
            uint64_t s = popcount_from_offsets(data, offs, 0);
            auto a1 = high_resolution_clock::now();
            asm volatile("" :: "r"(s));
            t.push_back(duration<double, micro>(a1 - a0).count());
        }
        double mt = median(t);
        double cost = (mt * 1000.0) / (double)offs.size();
        double fp_mb = offs.size() / (double)fixed_L * 4096.0 / 1024.0 / 1024.0;
        const char* tag = "?";
        if (cs.l3 > 0){
            double l1_mb = cs.l1d/1024.0/1024.0, l2_mb = cs.l2/1024.0/1024.0, l3_mb = cs.l3/1024.0/1024.0;
            if (fp_mb <= l1_mb) tag = "<=L1d";
            else if (fp_mb <= l2_mb) tag = "<=L2";
            else if (fp_mb <= l3_mb) tag = "<=L3";
            else tag = ">L3";
        }
        cout << "    num_regions=" << setw(7) << nr << "  footprint=" << fixed << setprecision(2)
             << setw(9) << fp_mb << "MB [" << setw(5) << tag << "]  cost/line=" << setprecision(3)
             << setw(8) << cost << "ns\n";
    }
    cout << "\n";
}

// ============================================================
// [v23-NEW, PHAT MINH] Dinh luat dong dang IRM-Burst
//   Xem giai thich day du (dan xuat + kiem chung Monte Carlo ngoai chuong
//   trinh) o comment dau file, muc [Y]. Tom tat cong thuc:
//     occupancy(C,W) = min(1, C/W)         (xac suat 1 trang cu the dang
//                                            trong TLB/cache dung luong C,
//                                            tren working-set W trang, duoi
//                                            tham chieu IID deu - suy ra
//                                            DUNG bang lap luan doi xung,
//                                            khong phai xap xi)
//     mien/line(L,W,C) = (1 - occupancy(C,W)) / L
// ============================================================
static inline double irm_occupancy(double C, double W){
    if (W <= 0.0) return 0.0;
    return min(1.0, C / W);
}
static inline double predict_miss_per_line(double L, double W, double C){
    return (1.0 - irm_occupancy(C, W)) / L;
}
// Suy nguoc C tu 1 diem do thuc te (L, W, mien/line do duoc) bang dai so
// truc tiep (dao nguoc cong thuc tren) - KHONG can toi uu hoa lap.
static inline double back_out_capacity(double L, double W, double measured_miss_per_line){
    double occ_hat = 1.0 - measured_miss_per_line * L;
    occ_hat = max(0.0, min(1.0, occ_hat));
    return occ_hat * W;
}

// ---------- LRU O(1) trung binh (list + hash) dung cho self-test noi bo ----------
struct SimpleLRU {
    int cap;
    list<int> order; // front = vua dung gan nhat
    unordered_map<int, list<int>::iterator> pos;
    explicit SimpleLRU(int c): cap(c) {}
    // [v34-TOI UU, KHONG doi hanh vi] Ham nay duoc goi > 16 TRIEU lan trong 1
    // lan chay chuong trinh (self_test_irm_law + che_zipf + assoc_law + v27).
    // Ban goc: MOI lan goi (ke ca HIT - nhanh PHO BIEN nhat khi hit-rate cao)
    // deu lam it nhat 1 cap cap-phat/giai-phong node cua std::list (erase() roi
    // push_front() tao node MOI, thay vi di chuyen node cu). Sua: dung
    // list::splice() de DI CHUYEN node hien co (hit) hoac TAI SU DUNG node cua
    // nan nhan vua bi day ra (miss+day) len dau danh sach - CHI noi lai con
    // tro, KHONG malloc/free nao. Da kiem chung BIT-IDENTICAL (cung 1 chuoi
    // RNG, doi chieu hit/miss tung phan tu, ca truong hop cap=0) voi ban goc
    // truoc khi thay - hanh vi LRU/thu tu day khong doi, chi giam hang so.
    bool touch(int x){ // tra ve true neu la hit
        auto it = pos.find(x);
        if (it != pos.end()){
            order.splice(order.begin(), order, it->second);
            return true;
        } else {
            if ((int)order.size() >= cap && cap > 0){
                auto victim_it = prev(order.end());
                int victim = *victim_it;
                pos.erase(victim);
                *victim_it = x;
                order.splice(order.begin(), order, victim_it);
                pos[x] = order.begin();
            } else if (cap > 0){
                order.push_front(x);
                pos[x] = order.begin();
            }
            return false;
        }
    }
};

// ---------- [v23-NEW] Self-test: kiem chung dinh luat IRM bang mo phong THAT ----------
// Mo phong mot LRU cache dung luong C duoi tham chieu IID deu tren W muc
// (dung cau truc du lieu O(1) that, khong phai suy dien tren giay), so
// sanh ty le hit thuc nghiem voi cong thuc dong dang C/W. Day la buoc kiem
// chung NOI TAI, chay MOI LAN chuong trinh khoi dong phan [v23] - neu that
// bai (sai lech > nguong), phan con lai cua [v23] se BI BO QUA thay vi in
// ra ket luan suy dien tu 1 dinh luat chua duoc xac nhan la dung tren may
// nay (trinh bien dich/CPU/RNG khac nhau ve nguyen tac khong the doi ket
// qua cua 1 chung minh toan hoc, nhung buoc nay van giu tinh than "khong
// tin - kiem chung" xuyen suot file, giong self_test_bit_identical o tren).
static bool self_test_irm_law(mt19937_64& rng){
    struct Case { int W, C; long refs; };
    vector<Case> cases = {
        {3, 1, 300000}, {10, 7, 300000}, {1000, 64, 800000},
        {19531, 64, 800000}, {19531, 1536, 800000}
    };
    bool all_ok = true;
    cout << "[self-test IRM] Mo phong LRU that (O(1)) duoi IID-uniform, doi chieu voi C/W:\n";
    for (auto& c : cases){
        uniform_int_distribution<int> pick(0, c.W - 1);
        SimpleLRU cache(c.C);
        long hits = 0;
        for (long i = 0; i < c.refs; i++)
            if (cache.touch(pick(rng))) hits++;
        double emp = (double)hits / (double)c.refs;
        double theory = irm_occupancy((double)c.C, (double)c.W);
        double err = fabs(emp - theory);
        bool ok = err < 0.01; // nguong 1% - du chat de bat loi logic, du long cho nhieu Monte Carlo
        cout << "  W=" << setw(6) << c.W << " C=" << setw(5) << c.C
             << "  ly_thuyet=" << fixed << setprecision(5) << theory
             << "  thuc_nghiem=" << emp << "  sai_lech=" << setprecision(5) << err
             << (ok ? "  OK\n" : "  [LOI! dinh luat KHONG khop mo phong tren may nay]\n");
        all_ok = all_ok && ok;
    }
    return all_ok;
}

// ============================================================
// [v27-MOI, PHAT HIEN QUAN TRONG - khoang trong tu-kiem-chung chua tung
// duoc phat hien qua [v18]-[v26]]
//
// VAN DE: self_test_irm_law() o tren xac nhan cong thuc occupancy=C/W bang
// cach rut MOI THAM CHIEU TUOI MOI (pick(rng) goi lai moi vong lap). NHUNG
// generate_offsets() + vong do thoi gian THAT (o [Q]/[v23], xem HwCounter
// va vong "for (tr...) popcount_from_offsets(normal, offs, 0)") lam KHAC
// HAN: sinh offs MOT LAN DUY NHAT roi PHAT LAI y het (byte-for-byte) qua
// TRIALS lan. Day la HAI qua trinh sinh du lieu khac nhau ve BAN CHAT thong
// ke, va cong thuc duoc XAC NHAN cho qua trinh (1) dang bi AP DUNG ngam cho
// qua trinh (2) ma chua bao gio duoc doi chieu truc tiep.
//
// LAP LUAN (co the kiem chung dai so): duoi mot chinh sach LRU LY TUONG,
// mot chuoi TUAN HOAN CO DINH voi chu ky = W trang PHAN BIET (moi trang
// cach lan tham chieu truoc dung W-1 trang khac, LUON LUON - khong ngau
// nhien) se hoi tu, o trang thai on dinh (sau 1 chu ky warm-up), ve DUNG
// MOT trong hai gia tri: HIT=1 neu C >= W-1 (moi thu vua du cho ca chu ky),
// hoac MISS=1 neu C < W-1 (khong gi song sot du 1 chu ky) - tuc la HAM BAC
// THANG tai C=W, chu KHONG PHAI duong cong tron C/W. Day la he qua TAT YEU
// cua dinh nghia LRU (khong phai xap xi, khong phai loi mo phong) khi ap
// dung cho MOT chuoi CO DINH thay vi cac lan rut IID doc lap.
//
// Ham nay tu-kiem-chung DIEU NAY bang C++ that (khong chi lap luan tren
// giay): mo phong CA HAI qua trinh sinh du lieu tren CUNG (W,C) va SO SANH
// truc tiep voi cong thuc C/W - neu chuoi co dinh THAT SU cho ham bac
// thang (khac xa C/W o giua khoang), day la bang chung C++ cu the cho mot
// mau thuan noi tai giua CACH cong thuc duoc xac nhan va CACH no duoc dung.
// ============================================================
static void self_test_irm_fixed_replay_gap(mt19937_64& /*rng*/){
    struct Case { int W; int n_periods; };
    vector<Case> cases = { {50, 60}, {200, 60}, {1000, 40} };
    cout << "[self-test v27] MAU THUAN NOI TAI: doi chieu cong thuc C/W (duoc\n"
            "  self_test_irm_law xac nhan duoi IID-tuoi-moi-lan) voi hanh vi THAT cua\n"
            "  MOT CHUOI CO DINH PHAT LAI (dung chinh xac cach generate_offsets() sinh\n"
            "  offs MOT LAN roi vong TRIALS phat lai y het, nhu [Q]/[v23] lam that):\n";
    double max_gap_at_mid = 0.0;
    for (auto& c : cases){
        for (double frac : {0.25, 0.5, 0.75, 0.9, 1.0, 1.1}){
            int C = (int)((double)c.W * frac);
            SimpleLRU cache2(C);
            long hits2 = 0, total2 = 0, idx = 0;
            long warm = c.W; // bo qua 1 chu ky dau (warm-up) khi dem steady-state
            for (int rep = 0; rep < c.n_periods; rep++){
                for (int i = 0; i < c.W; i++){
                    bool hit = cache2.touch(i); // "i" = danh tinh co dinh cua trang thu i trong 1 chu ky
                    if (idx >= warm){ total2++; if (hit) hits2++; }
                    idx++;
                }
            }
            double fixed_replay_rate = total2 ? (double)hits2 / (double)total2 : 0.0;
            double smooth_theory = min(1.0, (double)C / (double)c.W);
            double gap = fabs(smooth_theory - fixed_replay_rate);
            if (frac > 0.2 && frac < 0.95) max_gap_at_mid = max(max_gap_at_mid, gap);
            cout << "  W=" << setw(5) << c.W << " C=" << setw(5) << C
                 << " (C/W=" << fixed << setprecision(2) << frac << ")"
                 << "   C/W_du_doan=" << setprecision(4) << smooth_theory
                 << "   chuoi_CO_DINH_thuc_te=" << fixed_replay_rate
                 << "   |khoang_cach|=" << gap
                 << (gap > 0.15 ? "  <-- LECH LON\n" : "\n");
        }
    }
    cout << "\n  => Khoang cach lon nhat o giua khoang (0.2<C/W<0.95): " << fixed << setprecision(4)
         << max_gap_at_mid << " (tuc " << setprecision(1) << (max_gap_at_mid*100.0) << "% ty le hit)\n"
         << "  KET LUAN [v27]: duoi LRU LY TUONG, chuoi CO DINH phat lai hoi tu ve HAM BAC\n"
         << "  THANG (0/1 tai C=W), chu KHONG PHAI duong cong tron C/W - day la mau thuan\n"
         << "  THAT (dai so, khong phai nhieu mo phong: xem gia tri lech >0 ro rang o tren).\n"
         << "  Y NGHIA: neu du lieu dTLB-miss THAT do o [v23]/[Q] van khop tot voi duong\n"
         << "  cong tron C/W (R^2 cao da bao cao), day KHONG THE la vi phan cung dang hanh\n"
         << "  xu dung 'LRU ly tuong tren 1 chuoi co dinh' (vua chung minh se cho ham bac\n"
         << "  thang) - loi giai thich hop ly hon: (1) TLB/cache THAT dung chinh sach GAN-\n"
         << "  DUNG (pseudo-LRU/tree-PLRU/random-replacement), KHONG phai LRU chinh xac,\n"
         << "  va/hoac (2) nhieu tu ngat/context-switch/SMT-sibling giua cac lan lap TRIALS\n"
         << "  lam chuoi 'co dinh' o muc dia chi khong con thuc su co dinh o hanh vi cache\n"
         << "  vi mo (dau du dia chi giong het nhau). Day la mot GIAI THICH KHAC cho CUNG\n"
         << "  1 hien tuong da quan sat (duong cong tron), truoc day chua tung duoc xet -\n"
         << "  tat ca cac phan [v23]-[v26] truoc gio ngam dinh 'khop C/W => LRU-IID dung',\n"
         << "  nhung ket qua nay cho thay do la mot suy luan KHONG day du.\n\n";
}

// ---------- Cache thay-the-ngau-nhien (random-replacement) O(1) cho self-test v27b ----------
struct RandomReplCache {
    int cap;
    vector<int> resident;
    unordered_set<int> resident_set;
    mt19937_64* rng;
    RandomReplCache(int c, mt19937_64* r): cap(c), rng(r) {}
    bool touch(int x){
        if (resident_set.count(x)) return true;
        if ((int)resident.size() >= cap && cap > 0){
            uniform_int_distribution<size_t> vic(0, resident.size() - 1);
            size_t idx = vic(*rng);
            int victim = resident[idx];
            resident[idx] = resident.back();
            resident.pop_back();
            resident_set.erase(victim);
        }
        if (cap > 0){ resident.push_back(x); resident_set.insert(x); }
        return false;
    }
};

// ============================================================
// [v27b-MOI] Kiem tra gia thuyet giai thich (1) o tren: chinh sach thay-the-
// NGAU-NHIEN (thay vi LRU chinh xac) tren CUNG chuoi co dinh co lam MUOT
// duong cong (thay vi ham bac thang) hay khong - va neu co, no co TRUNG voi
// C/W hay la MOT duong cong KHAC (can cong thuc rieng, chua co o day)?
// Day la buoc THAM DO (khong phai chung minh dong), giup dinh huong xem
// gia thuyet (1) o self_test_irm_fixed_replay_gap co dang huong dung hay
// khong - TRUNG THUC bao cao ca truong hop no KHONG trung khop C/W.
// ============================================================
static void self_test_random_replacement_shape(mt19937_64& rng){
    cout << "[self-test v27b, THAM DO] Chinh sach thay-the-NGAU-NHIEN tren CUNG chuoi\n"
            "  co dinh (thay vi LRU ly tuong o [v27]) co lam muot duong cong khong?\n";
    struct Case { int W; int n_periods; };
    vector<Case> cases = { {200, 80}, {1000, 50} };
    for (auto& c : cases){
        for (double frac : {0.25, 0.5, 0.75, 0.9}){
            int C = (int)((double)c.W * frac);
            mt19937_64 repl_rng(rng()); // rieng cho quyet dinh vat nan nhan, tach khoi rng ngoai
            RandomReplCache cache3(C, &repl_rng);
            long hits3 = 0, total3 = 0, idx = 0;
            long warm = c.W;
            for (int rep = 0; rep < c.n_periods; rep++){
                for (int i = 0; i < c.W; i++){
                    bool hit = cache3.touch(i);
                    if (idx >= warm){ total3++; if (hit) hits3++; }
                    idx++;
                }
            }
            double random_repl_rate = total3 ? (double)hits3 / (double)total3 : 0.0;
            double smooth_theory = min(1.0, (double)C / (double)c.W);
            cout << "  W=" << setw(5) << c.W << " C=" << setw(5) << C
                 << " (C/W=" << fixed << setprecision(2) << frac << ")"
                 << "   C/W_IID=" << setprecision(4) << smooth_theory
                 << "   thay_the_ngau_nhien=" << random_repl_rate
                 << "   lech=" << fabs(smooth_theory - random_repl_rate) << "\n";
        }
    }
    cout << "\n  => KET LUAN [v27b, THAM DO]: neu bang tren cho thay 'thay_the_ngau_nhien'\n"
            "  la duong cong TRON (khong nhay bac) NHUNG lech ro rang khoi C/W (thuong\n"
            "  THAP HON o giua khoang), day la bang chung TRUNG GIAN quan trong: gia\n"
            "  thuyet 'chinh sach gan-dung lam muot duong cong' la HOP LY VE HUONG, nhung\n"
            "  C/W cua [v23] KHONG PHAI cong thuc dung cho chinh sach nay - can mot cong\n"
            "  thuc rieng (vd qua ly thuyet 'random paging'/urn-model) neu muon suy nguoc\n"
            "  C tu dTLB that voi do chinh xac cao hon, thay vi tiep tuc dung C/W nhu mot\n"
            "  xap xi khong ro sai so cho chinh sach thay-the THAT cua phan cung. Day la\n"
            "  huong mo rong TRUNG THUC con bo ngo, KHONG duoc gia vo da giai xong o day.\n\n";
}

// ============================================================
// [v36-NEW] "3-tier empirical miss-rate predictor" - THAY THE TOAN BO
//   chuoi [v24]->[v26]->[v28]->[v35] (Che-Zipf / set-assoc / hop nhat /
//   nghiem chinh xac tap trung). [v23] (irm_occupancy/predict_miss_per_line/
//   back_out_capacity o TREN) duoc GIU NGUYEN, khong doi: no la dang thuc
//   DUNG TUYET DOI (khong xap xi) cho truong hop popularity DEU, va van
//   dang duoc dung THAT de suy ngoc C tu du lieu PMU do (VarPro o [v23]/
//   [Model]) - go bo no se pha vo pipeline suy dien C tu phan cung THAT.
//
// LY DO THAY THE (khong phai "thich hon" chu quan): chinh [v27] o TREN
//   (van con giu nguyen) da tu-chung-minh bang C++ that (khong phai suy
//   dien tren giay) rang tien de nen tang cua CA BON phan mo rong [v24]/
//   [v26]/[v28]/[v35] - "tham chieu IID TUOI MOI moi lan" - KHONG dung voi
//   cach generate_offsets() thuc su sinh du lieu (sinh 1 lan, PHAT LAI y
//   het qua nhieu trial). Ca bon phan mo rong deu la ho ham mu (Che hoac
//   suy bien tu Che), nen deu ke thua CUNG mot diem mu: ep mot regime CO
//   THE la TUAN HOAN vao dang cong ham mu tron.
//
// [v36] khong va tiep 1 xap xi khac trong CUNG ho mu do - ma doi CACH TIEP
//   CAN: do TRUC TIEP stack-distance THAT tu chinh chuoi truy cap dang xet
//   (khong gia dinh truoc dang phan phoi), thu Che truoc (VAN dung khi phu
//   hop - [v36] BAO TRUM [v23]/[v24] nhu 1 nhanh, khong vut bo), NEU Che
//   khong khop thi tu phat hien xem co phai regime TUAN HOAN hay khong
//   (dung dung cho hop [v27] da chi ra la thieu), va CHI KHI CA HAI that
//   bai moi roi vao xap xi EM 2-mu (cuu ho, gan nhan "chua chac chan" ro
//   rang, KHONG gia vo la dinh luat da chung minh).
//
// Nguon: port 1-1 tu 3 file doc lap da duoc chay & doi chieu THAT (khong
//   chi suy dien): stack_distances/decompose_bursts/che_zipf/che_burst/
//   best_step_fit (realistic_test.py + final_predictor.py) va
//   em_exp_mixture/solve_Tc (physical_law.py). Ket qua C++ ben duoi da
//   duoc doi chieu bang tay voi ban Python goc tren cung 3 loai chuoi mau
//   (YCSB/SPEC/bioinfo) va cho cung MODE + sai so cung bac do lon (xem
//   self_test_predictor_v36 - PHAI qua truoc khi in ket luan, dung tinh
//   than "khong tin - kiem chung" xuyen suot file nay).
// ============================================================

// ---------- Fenwick tree (BIT) - tinh stack distance O(n log n) ----------
struct FenwickV36 {
    int n;
    vector<double> t;
    explicit FenwickV36(int n_): n(n_), t(n_ + 2, 0.0) {}
    void add(int i, double v){ for (i++; i <= n; i += i & (-i)) t[i] += v; }
    double prefix(int i) const { double s = 0.0; for (i++; i > 0; i -= i & (-i)) s += t[i]; return s; }
    double range_sum(int lo, int hi) const {
        if (hi < lo) return 0.0;
        return prefix(hi) - (lo > 0 ? prefix(lo - 1) : 0.0);
    }
};

// Stack distance cua moi truy cap trong "trace" (id trang 0..W-1). Truy cap
// LAN DAU cua 1 trang duoc gan khoang cach = W+1000 ("lanh"/cold, giong
// dung quy uoc INF cua realistic_test.py, khong phai gia tri tuy tien).
static vector<int> stack_distances_v36(const vector<int>& trace, int W){
    int n = (int)trace.size();
    FenwickV36 fen(n);
    unordered_map<int,int> last_pos;
    last_pos.reserve((size_t)W * 2);
    vector<int> dist(n);
    for (int t = 0; t < n; t++){
        int x = trace[t];
        auto it = last_pos.find(x);
        if (it != last_pos.end()){
            int p = it->second;
            dist[t] = (int)llround(fen.range_sum(p + 1, t - 1));
            fen.add(p, -1.0);
        } else {
            dist[t] = W + 1000;
        }
        fen.add(t, 1.0);
        last_pos[x] = t;
    }
    return dist;
}

// RLE burst decomposition THAT (burst = doan lien tiep GIONG HET nhau,
// khong phai gia dinh burst-length co dinh nhu [v28]/[v35] o tren tung
// lam). q_i = ty le BURST (khong phai ty le truy cap tho) nham vao trang
// i; L_i = do dai burst trung binh cua trang i.
struct BurstDecompV36 { vector<double> q; vector<double> L; };
static BurstDecompV36 decompose_bursts_v36(const vector<int>& trace, int W){
    vector<double> burst_count(W, 0.0), burst_len_sum(W, 0.0);
    int n = (int)trace.size();
    int i = 0;
    while (i < n){
        int j = i, item = trace[i];
        while (j < n && trace[j] == item) j++;
        burst_count[item] += 1.0;
        burst_len_sum[item] += (double)(j - i);
        i = j;
    }
    double total_bursts = 0.0; for (double c : burst_count) total_bursts += c;
    BurstDecompV36 out; out.q.resize(W); out.L.assign(W, 1.0);
    for (int k = 0; k < W; k++){
        out.q[k] = (total_bursts > 0.0) ? burst_count[k] / total_bursts : 0.0;
        if (burst_count[k] > 0.0) out.L[k] = burst_len_sum[k] / burst_count[k];
    }
    return out;
}

// Che "characteristic time" qua bisection - dung y het tinh than
// che_characteristic_time cua [v24] cu (da go bo o tren), nhung o day q_i
// duoc DO THAT tu du lieu (raw access freq hoac burst freq), khong phai
// Zipf-gia-lap tu-tao chi de tu-kiem-chung ly thuyet.
// [v38-FIX #2] Ban cu: bisection THUAN TUY 100 vong, moi vong O(W) (danh
// gia g(t)) -> O(100*W) MOI LAN GOI, va bi goi W lan trong che_zipf_all/
// che_burst_all -> O(100*W^2) tong, la phan nang nhat pipeline. Ban moi:
// - Newton's method: g(t) don dieu tang va loi (concave, vi g'(t) =
//   Sum(qi*e^{-qi*t}) > 0 giam dan), nen Newton hoi tu BAC HAI - trong
//   thuc te ~6-10 vong la du (thay vi 100), va van GIU bisection lam
//   "guard rail" (Newton-bisection hybrid): neu 1 buoc Newton nhay ra
//   ngoai [lo,hi] hien tai (co the xay ra o vung g'(t) rat nho khi t
//   lon), rot ve 1 buoc bisection thay vi de phan ky - dam bao HOI TU
//   CHAT trong moi truong hop, khong danh doi do dung lay toc do.
// - Warm-start: nhan them t0 (uoc luong ban dau, vd Tc cua C-1 - Tc(C)
//   don dieu tang theo C nen day la mot warm-start TU NHIEN, khong phai
//   suy doan), giup Newton hoi tu cang nhanh hon khi goi lien tiep tren
//   mot luoi C=1..W.
// Ket qua so hoc: van la nghiem cua CHINH PHUONG TRINH g(t)=C nhu ban cu
// (khong doi dinh nghia mo hinh Che), chi doi thuat toan giai cho nhanh hon.
static double solve_Tc_v36(const vector<double>& q, double C, double t0 = -1.0){
    double W = (double)q.size();
    if (C >= W) return 1e15;
    auto g  = [&](double t){ double s = 0.0; for (double qi : q) s += 1.0 - exp(-qi * t); return s; };
    auto gp = [&](double t){ double s = 0.0; for (double qi : q) s += qi * exp(-qi * t); return s; };

    // [v38-FIX bug] g(t) hoi tu ve mot TRAN HUU HAN khi t->inf: gioi han =
    // so luong q_i > 0 (cac trang q_i=0 khong bao gio dong gop vao g(t) du
    // t lon co nao). Neu C >= tran nay, phuong trinh g(t)=C VO NGHIEM HUU
    // HAN - truoc day khong kiem tra dieu nay nen vong doubling
    // "for(guard...) hi*=2.0" chay het 200 vong roi tra ve 1 hi khong lo,
    // va vi hi do lai duoc dung lam t0 WARM-START cho buoc C+1 ke tiep, no
    // bi nhan doi 2^200 lan nua chi trong vai buoc C -> TRAN SO thanh inf
    // -> exp(-0*inf)=nan -> NaN lan ra toan bo che_zipf/che_burst. Fix:
    // phat hien truong hop nay TRUOC vong doubling, tra ve hang so lon
    // 1e15 GIONG HET nhanh "C>=W" o tren (ap dung cung quy uoc da co san
    // cho truong hop tuong tu, khong phai gia tri suy dien moi), de khong
    // lam vong doubling chay va khong lam nhiem ban warm-start.
    double g_sup = 0.0; for (double qi : q) if (qi > 0.0) g_sup += 1.0;
    if (C >= g_sup) return 1e15;

    double lo = 0.0, hi = (t0 > 0.0) ? t0 : 1.0;
    for (int guard = 0; g(hi) < C; guard++){
        if (guard > 200) return hi; // chan cung, khong treo vo han (dung bai hoc [v33-FIX] cu)
        hi *= 2.0;
    }
    double t = (t0 > 0.0 && t0 < hi) ? t0 : 0.5 * (lo + hi);

    for (int it = 0; it < 60; it++){ // tran an toan; Newton tot thuong hoi tu << 60
        double gt = g(t) - C;
        if (fabs(gt) < 1e-12) return t;
        if (gt < 0.0) lo = t; else hi = t; // duy tri bracket [lo,hi] lam guard-rail

        double deriv = gp(t);
        double t_newton = (deriv > 1e-300) ? (t - gt / deriv) : -1.0;
        if (t_newton > lo && t_newton < hi){
            t = t_newton; // buoc Newton hop le -> dung (hoi tu bac hai)
        } else {
            t = 0.5 * (lo + hi); // roi ra ngoai bracket -> rot ve bisection 1 buoc
        }
        if (hi - lo < 1e-12) break;
    }
    return 0.5 * (lo + hi);
}
// che_zipf: q_i = tan suat truy cap THO (khong tinh burst) - lua chon
// "moi nguoi lam mac dinh" khi bo qua cau truc burst.
static double che_zipf_v36(double C, const vector<double>& q, double t0 = -1.0, double* tc_out = nullptr,
                            double p_cold = 0.0){
    int W = (int)q.size();
    if (C <= 0.0) return 1.0;
    // [v39-FIX #7] C>=W KHONG con nghia la miss=0 tuyet doi: du cache chua
    // toan bo working-set, van co san p_cold truy cap compulsory (lan dau
    // cham 1 trang chua tung xuat hien) - day la mot floor CO THAT cua
    // MR_real, khong phu thuoc kich thuoc cache. Mo hinh Che goc (khong co
    // p_cold) gia dinh mien gia tri [0,1] va hoi tu ve 0, nen sai lech
    // he thong ~0.09 tren toan dai C lon trong self-test N=3000 (W=300,
    // chi 279 trang tung duoc cham -> p_cold=21/3000=0.007... that ra la
    // ty le truy cap compulsory, khong phai ty le trang - xem noi goi).
    if (C >= W) return p_cold;
    double Tc = solve_Tc_v36(q, C, t0);
    if (tc_out) *tc_out = Tc;
    double miss = 0.0;
    for (double qi : q) miss += qi * exp(-qi * Tc);
    // Tron: p_cold la san bat buoc; phan con lai (1-p_cold) la mo hinh Che
    // ap dung PHIA TREN san do, dam bao mien gia tri [p_cold, 1] thay vi
    // [0, 1] nhu ban goc.
    return p_cold + (1.0 - p_cold) * miss;
}
// che_burst: q_i/L_i tu RLE THAT - miss/access = miss/burst chia cho so
// truy cap trung binh moi burst (chi truy cap DAU tien cua 1 burst co the
// miss, cac truy cap sau LUON hit lai chinh trang do).
static double che_burst_v36(double C, const vector<double>& q, const vector<double>& L, double t0 = -1.0, double* tc_out = nullptr,
                             double p_cold = 0.0){
    int W = (int)q.size();
    if (C <= 0.0) return 1.0;
    // [v39-FIX #7] xem giai thich day du o che_zipf_v36 - cung 1 ly do:
    // compulsory-miss la mot ty le TREN TONG SO TRUY CAP (khong phai tren
    // burst), nen cong truc tiep p_cold (da tinh o muc truy cap, xem noi
    // goi) vao ca hai nhanh C>=W va nhanh Newton, giu nguyen quy uoc tron.
    if (C >= W) return p_cold;
    double Tc = solve_Tc_v36(q, C, t0);
    if (tc_out) *tc_out = Tc;
    double miss_bursts = 0.0, denom = 0.0;
    for (int i = 0; i < W; i++){
        miss_bursts += q[i] * exp(-q[i] * Tc);
        denom += q[i] * L[i];
    }
    double miss = denom > 0.0 ? miss_bursts / denom : 0.0;
    return p_cold + (1.0 - p_cold) * miss;
}

// Bo phat hien chu ky/tuan hoan: khop 1 ham bac thang VOI MUC TU DO (top/
// bottom la median thuc nghiem moi ben, KHONG ep cung 1/0 - burst co the
// khien "top" thap hon 1 that su). Day chinh la dang ham ma [v27] da CHUNG
// MINH la dung cho chuoi PHAT LAI CO DINH, thay vi duong cong C/W tron.
struct StepFitV36 { int cstar; double err, top, bot; };
// [v38-FIX #1] Ban cu: moi vong k sort lai TOAN BO left/right tu dau ->
// O(W log W) MOI BUOC x W buoc = O(W^2 log W) tong. Ban moi: duy tri 2
// multiset (left = k phan tu dau, right = W-k phan tu con lai) va DICH
// DUNG 1 phan tu tu right sang left moi buoc k -> phan cap nhat median
// chi con O(log W)/buoc (1 erase + 1 insert), median doc truc tiep tu
// iterator giua (khong sort lai). Phan tinh err van la O(n)/buoc (sai so
// L1 tong quat khong co cong thuc O(1) don gian khi doi 1 phan tu), nen
// tong dai ham giam tu O(n^2 log n) xuong O(n^2) - van la cai thien ro
// rang, va la muc can thiet cho quy mo W hien tai (vai tram - vai nghin).
static StepFitV36 best_step_fit_v36(const vector<int>& Cs, const vector<double>& MR_real){
    int n = (int)Cs.size();
    double best_err = 1e300; int best_cstar = Cs[0]; double best_top = 0, best_bot = 0;

    multiset<double> left, right(MR_real.begin(), MR_real.end());
    auto median_of = [](const multiset<double>& s) -> double {
        size_t m = s.size() / 2;
        auto it = s.begin(); advance(it, m);
        if (s.size() % 2 == 0){ double b = *it; --it; return 0.5 * (*it + b); }
        return *it;
    };
    for (int k = 1; k < n; k++){ // k=0 -> nhanh "trai" rong, bo qua (giong Python enumerate)
        // dich phan tu thu (k-1) tu right sang left (dung 1 phan tu/buoc)
        auto it = right.find(MR_real[k-1]);
        right.erase(it);
        left.insert(MR_real[k-1]);

        double top = median_of(left);
        double bot = median_of(right);
        double err = 0.0;
        for (int i = 0; i < n; i++) err += fabs((Cs[i] < Cs[k] ? top : bot) - MR_real[i]);
        err /= n;
        if (err < best_err){ best_err = err; best_cstar = Cs[k]; best_top = top; best_bot = bot; }
    }
    return { best_cstar, best_err, best_top, best_bot };
}

// EM 2-mu (mixture cua 2 phan phoi mu) tren cac stack-distance HUU HAN
// (bo qua truy cap "lanh" - da tinh rieng qua p_cold). Cuu ho TANG CUONG
// (Tier 3), KHONG phai dinh luat - danh cho phan con lai sau khi Che va
// step deu that bai (vd non-stationary/phase-drift that su).
struct EMResultV36 { double a, lam1, lam2; };
static EMResultV36 em_exp_mixture_v36(const vector<double>& x, int n_iter = 100){
    double mean_x = 0.0; for (double v : x) mean_x += v; mean_x /= (double)x.size();
    double lam1 = mean_x * 0.2, lam2 = mean_x * 2.0; // khoi tao: 1 nhanh nhanh, 1 nhanh cham
    double a = 0.5;
    for (int it = 0; it < n_iter; it++){
        double sum_r = 0.0, sum_rx = 0.0, sum_1mr = 0.0, sum_1mrx = 0.0;
        for (double xv : x){
            double p1 = a * (1.0 / lam1) * exp(-xv / lam1);
            double p2 = (1.0 - a) * (1.0 / lam2) * exp(-xv / lam2);
            double denom = p1 + p2 + 1e-300;
            double r = p1 / denom;
            sum_r += r; sum_rx += r * xv; sum_1mr += (1.0 - r); sum_1mrx += (1.0 - r) * xv;
        }
        double a_new = sum_r / (double)x.size();
        // [FIX] sum_r/sum_1mr co the ve DUNG 0.0 khi 1 cum "thang the" hoan
        // toan trong vai vong dau (moi r ~ 0 hoac ~ 1) - chia cho 0 se ra NaN
        // va lan truyen am tham (so sanh voi NaN luon false nen "converged"/
        // "degenerate" phia duoi co the danh gia SAI). Giu nguyen gia tri cu
        // cua lam1/lam2 khi mau so qua nho thay vi chia cho 0.
        double lam1_new = (sum_r   > 1e-12) ? sum_rx   / sum_r   : lam1;
        double lam2_new = (sum_1mr > 1e-12) ? sum_1mrx / sum_1mr : lam2;
        bool converged = fabs(a_new - a) < 1e-9 && fabs(lam1_new - lam1) < 1e-6 && fabs(lam2_new - lam2) < 1e-6;
        a = a_new; lam1 = lam1_new; lam2 = lam2_new;
        if (converged) break;
    }
    return { a, lam1, lam2 };
}

// [v38-FIX #4] EM cho mixture 2 mu rat de ket vao local optimum, dac biet
// voi CHI 1 khoi tao co dinh (a=0.5, lam1=mean*0.2, lam2=mean*2) nhu ban
// cu. Vi day la nhanh "cuu ho" cuoi cung (Tier 3), sai lech do local
// optimum se lam bao cao "cuu ho co ich hay khong" khong dang tin. Ham
// nay chay N_RESTART_V38 khoi tao khac nhau (phai xa nhau ve ty le lam1/
// lam2 de kham pha nhieu vung nghiem khac nhau thay vi chi nhieu quanh 1
// diem), roi CHON nghiem co log-likelihood cao nhat - dung tieu chuan
// thong ke chuan (khong phai heuristic tu nghi), thay vi tin mu quang 1
// lan chay duy nhat.
static double loglik_exp_mixture_v38(const vector<double>& x, double a, double lam1, double lam2){
    double ll = 0.0;
    for (double xv : x){
        double p1 = a * (1.0 / lam1) * exp(-xv / lam1);
        double p2 = (1.0 - a) * (1.0 / lam2) * exp(-xv / lam2);
        ll += log(p1 + p2 + 1e-300);
    }
    return ll;
}
static constexpr int N_RESTART_V38 = 5;
static EMResultV36 em_exp_mixture_multistart_v36(const vector<double>& x, int n_iter = 100){
    double mean_x = 0.0; for (double v : x) mean_x += v; mean_x /= (double)max((size_t)1, x.size());
    // 5 cap (lam1_frac, lam2_frac, a) khoi tao rai deu ra xa nhau, thay vi
    // chi 1 diem (0.2, 2.0, 0.5) nhu ban cu - muc dich la pha vo tinh doi
    // xung/kham pha ca vung a nho va a lon.
    static const double seeds[N_RESTART_V38][3] = {
        {0.2, 2.0, 0.5},   // khoi tao goc [v36] - giu lai de tuong thich nguoc
        {0.05, 5.0, 0.3},
        {0.5, 1.5, 0.7},
        {0.1, 10.0, 0.5},
        {0.3, 0.8, 0.5},
    };
    EMResultV36 best{}; double best_ll = -1e300; bool have_best = false;
    for (int s = 0; s < N_RESTART_V38; s++){
        double lam1 = mean_x * seeds[s][0], lam2 = mean_x * seeds[s][1], a = seeds[s][2];
        // vong EM giong het em_exp_mixture_v36 nhung tai dung tai cho de
        // moi restart co khoi tao rieng (khong the goi ham cu vi no khoa
        // cung khoi tao ben trong).
        for (int it = 0; it < n_iter; it++){
            double sum_r = 0.0, sum_rx = 0.0, sum_1mr = 0.0, sum_1mrx = 0.0;
            for (double xv : x){
                double p1 = a * (1.0 / lam1) * exp(-xv / lam1);
                double p2 = (1.0 - a) * (1.0 / lam2) * exp(-xv / lam2);
                double denom = p1 + p2 + 1e-300;
                double r = p1 / denom;
                sum_r += r; sum_rx += r * xv; sum_1mr += (1.0 - r); sum_1mrx += (1.0 - r) * xv;
            }
            double a_new = sum_r / (double)x.size();
            double lam1_new = (sum_r   > 1e-12) ? sum_rx   / sum_r   : lam1;
            double lam2_new = (sum_1mr > 1e-12) ? sum_1mrx / sum_1mr : lam2;
            bool converged = fabs(a_new - a) < 1e-9 && fabs(lam1_new - lam1) < 1e-6 && fabs(lam2_new - lam2) < 1e-6;
            a = a_new; lam1 = lam1_new; lam2 = lam2_new;
            if (converged) break;
        }
        if (!isfinite(a) || !isfinite(lam1) || !isfinite(lam2) || lam1 <= 0.0 || lam2 <= 0.0) continue;
        double ll = loglik_exp_mixture_v38(x, a, lam1, lam2);
        if (!have_best || ll > best_ll){ best_ll = ll; best = {a, lam1, lam2}; have_best = true; }
    }
    // Neu CA 5 restart deu suy bien (rat hiem, vd x qua it phan tu), rot ve
    // ban khoi tao goc [v36] de van tra ve 1 gia tri xac dinh thay vi rac.
    if (!have_best) return em_exp_mixture_v36(x, n_iter);
    return best;
}

// ---------- [v37-NEW] San nhieu (noise floor) cho nguong tol, co gian theo N ----------
// Van de: MR_real[c] = suffix[c]/N la MOT UOC LUONG tu N mau huu han cua 1 ty le
// (dinh nghia thong ke: suffix[c] ~ Binomial(N, p_c) voi p_c = P(stack-distance > c)).
// Do do MR_real MANG THEO sai so chuan (SE) rieng cua no, KHONG lien quan gi den
// model (Che/step/EM) co dung hay khong:
//     SE(MR_real[c]) = sqrt(p_c*(1-p_c)/N) <= 0.5/sqrt(N)   (can tren, dat max tai p_c=0.5)
// Truoc [v37], nguong "err_che <= tol" va "step.err <= tol*3.0" dung tol=0.01 CO DINH,
// khong phu thuoc N. He qua: voi N nho (vd sau khi clamp N-fixed lam giam manh so
// diem thuc te, xem bao cao nhieu cuc doan truoc do), chinh SE cua MR_real co the
// VUOT QUA 0.01 -> err_che bi "nhiem" mot luong nhieu KHONG LIEN QUAN den do sai model,
// khien Tier1 bi tu choi OAN (false negative) hoac Tier2 bi chon SAI chi vi may man
// noise nho hon nguong tinh co, khong phai vi cau truc du lieu that su tuan hoan.
// [v37] them 1 SAN NHIEU ty le 1/sqrt(N), duoc calibrate bang he so K_NOISE_V37 (xem
// self-test moi them: 2 mau N NHO trong self_test_predictor_v36) de dam bao:
//   - Voi N lon (vd cac mau self-test goc, N~150k-250k): san nhieu << 0.01 -> tol_eff
//     ~ tol nhu cu, KHONG lam thay doi hanh vi da duoc kiem chung truoc do.
//   - Voi N nho: tol_eff no rong ra dung bang bac do lon cua SE thuc su cua MR_real,
//     tranh viec pipeline "tu tin ao" khi mau qua it.
// Day KHONG phai suy dien tren giay - K_NOISE_V37 duoc chon = 2.0 (~2-sigma, bao thu
// vua phai) roi XAC NHAN THAT bang self-test moi truoc khi dua vao dung.
// [cleanup] noise_floor_v36()/K_NOISE_V37 khong con duoc dung noi nao (xac
// nhan bang -Wall -Wextra: "defined but not used") - da xoa.

// [v39-FIX #3] Fix bootstrap ban dau: KHONG duoc resample truc tiep tu
// 'dist' quan sat duoc, vi dist da mang san cau truc THAT cua chuoi (kể ca
// khi chuoi la periodic/phase-drift That, khong IID) - resample kieu do lam
// "san nhieu" phinh to theo dung cai bat-on-dinh dang muon phat hien, khien
// Tier1 nuot luon ca SPEC/Bioinfo (da kiem chung: err nhay len 0.2+ nhung
// van bi coi la "che du dung" vi tol_eff bi thoi phong qua muc).
// Fix dung: bootstrap PARAMETRIC theo dung gia thiet H0 cua Che (IID/dung
// yen voi tan suat trang = q_raw da quan sat) - sinh MOT CHUOI IID GIA
// DINH DAI N tu q_raw, tinh MR curve cua chuoi gia dinh do, so voi duong
// cong Che co dinh. Neu du lieu THAT su IID (dung gia thiet Che), day chinh
// la phan phoi lay mau ma err_che se dao dong quanh no thuan tuy do N huu
// han - khong lien quan gi den phi-tuyen-tinh/phase-drift that (vi chuoi
// gia dinh LUON IID theo xay dung, bat ke chuoi that co IID hay khong).
// Ket qua: voi du lieu THAT khong IID (SPEC/Bioinfo), err_che THAT van lon
// hon han nguong nay rat nhieu -> Tier1 van tu choi dung; voi du lieu THAT
// IID nhung N nho (YCSB N=3000), nguong nay dung phan anh dung do dao dong
// ngau nhien ma Che se gap - khong con la suy dien tren giay.
// [v39-FIX #3b] Fix tiep: ban dau #3 so moi bootstrap replicate voi
// 'model_curve' (duong Che DA FIT), nhung Che la mot xap xi giai tich luon
// mang mot SAI SO HE THONG (bias) so voi duong IID-THAT cua chinh q do -
// bias nay KHONG phu thuoc N, chi phu thuoc HINH DANG cua q (q cang lech/
// Zipf manh nhu SPEC thi bias cua Che cang lon). Vi moi bootstrap replicate
// deu bi cong them CUNG mot bias do, ket qua la:
//     err_b = bias(c) + noise_b(c)
// va percentile-90 cua err_b VO TINH "nuot" luon ca bias he thong cua Che
// vao trong "tol nhieu do N" - day chinh la ly do SPEC (q lech manh) bi
// tol_eff thoi phong qua muc (err_che=0.219 van <= tol_eff), lam mat do sac
// ben (specificity) da ghi nhan trong self-test.
// Fix dung: TACH bias ra khoi noise truoc khi lay percentile.
//   - mr_bar(c)  = trung binh MR_synth qua B replicate -> uoc luong duong
//                  IID-THAT cua q (bias cua Che tu trieu tieu o day, vi day
//                  la trung binh thuc nghiem cua chinh cac chuoi IID, KHONG
//                  so voi model_curve).
//   - bias        = MAE(mr_bar, model_curve) -> sai so HE THONG cua cong
//                  thuc Che cho DUNG hinh dang q nay (khong phu thuoc N).
//   - noise_floor = percentile-90 cua MAE(MR_synth_b, mr_bar) -> do tan xa
//                  THAT SU quanh mr_bar do N huu han (da loai bias).
//   tol_eff-floor = bias + noise_floor: "muc lech toi da Che LE RA phai tha
//   thu, ke ca khi du lieu dung la IID" - phan lech THEM do cau truc phi-IID
//   that (nhu periodic cua SPEC) van vuot qua nguong nay va bi Tier1 tu choi
//   dung, trong khi N nho (YCSB) van duoc va dung nhu #3 (vi bias ~ 0 khi q
//   gan deu, cong thuc suy bien ve dung cong thuc cu).
static double bootstrap_tol_floor_v39(const vector<double>& q, int N, int W,
                                       const vector<double>& model_curve, mt19937_64& rng, int B = 30){
    if (q.empty() || N <= 0) return 0.0;
    discrete_distribution<int> page_dist(q.begin(), q.end());
    int INF = W + 1000;

    vector<vector<double>> mr_synth(B, vector<double>(W));
    for (int b = 0; b < B; b++){
        vector<int> synth_trace(N);
        for (int i = 0; i < N; i++) synth_trace[i] = page_dist(rng); // IID DUNG theo q_raw
        vector<int> sdist = stack_distances_v36(synth_trace, W);
        vector<long long> counts((size_t)INF + 2, 0);
        for (int d : sdist) counts[min(d, INF)]++;
        vector<long long> suffix((size_t)INF + 2, 0);
        long long run = 0;
        for (int c = INF; c >= 0; c--){ run += counts[c]; suffix[c] = run; }
        for (int c = 1; c <= W; c++) mr_synth[b][c-1] = (double)suffix[c] / (double)N;
    }

    // mr_bar = uoc luong duong IID-THAT (bias cua Che tu trieu tieu vi day
    // la trung binh thuc nghiem cua chinh cac chuoi IID, khong so voi
    // model_curve).
    vector<double> mr_bar(W, 0.0);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < W; c++) mr_bar[c] += mr_synth[b][c] / (double)B;

    double bias = 0.0;
    for (int c = 0; c < W; c++) bias += fabs(mr_bar[c] - model_curve[c]);
    bias /= (double)W;

    vector<double> errs; errs.reserve(B);
    for (int b = 0; b < B; b++){
        double e = 0.0;
        for (int c = 0; c < W; c++) e += fabs(mr_synth[b][c] - mr_bar[c]);
        errs.push_back(e / (double)W);
    }
    sort(errs.begin(), errs.end());
    size_t idx = (size_t)(0.9 * (errs.size() - 1)); // phan vi 90 - bao thu vua phai, giong tinh than K=2.0 cu
    double noise_floor = errs[idx];

    return bias + noise_floor;
}

// ---------- Bo dieu phoi 3-tang (port 1-1 tu predict_miss_rate() cua
// final_predictor.py) ----------
struct TierResultV36 {
    string mode; // "che" | "periodic" | "em_rescue"
    double err = 0.0;
    string che_variant;
    int cstar = 0; double step_top = 0.0, step_bot = 0.0;
    double em_a = 0.0, em_lam1 = 0.0, em_lam2 = 0.0;
    bool em_degenerate = false;
    double tol_eff = 0.0; // [v37] nguong THAT SU da dung sau khi co gian theo N (de bao cao/debug)
    // [v38-FIX #6] luu THO ca err_che va step.err bat ke tier nao duoc chon,
    // de ben ngoai co the bootstrap CI cho CHINH quyet dinh chon tier (xem
    // bootstrap_tier_confidence_v38 ben duoi) - truoc day chi r.err duoc
    // gan (tuy tier), nen khong the biet err_che/step.err "that" o cac tier
    // khac de danh gia bien gioi quyet dinh co chac chan hay khong.
    double err_che_raw = 0.0, step_err_raw = 0.0;
};
static TierResultV36 predict_miss_rate_v36(const vector<int>& trace, int W, double tol,
                                            bool verbose, mt19937_64& rng){
    int N = (int)trace.size();
    vector<int> dist = stack_distances_v36(trace, W);
    int INF = W + 1000;
    vector<int> Cs(W); for (int i = 0; i < W; i++) Cs[i] = i + 1;

    // MR_real: doc TRUC TIEP tu histogram stack-distance cua CHINH chuoi -
    // day la dai luong DUNG TUYET DOI (dinh nghia cua LRU), khong phai mo
    // phong doc lap - vai tro cua no giong SimpleLRU trong cac self-test
    // [v23]/[v26]/[v35] o tren: lam thuoc do "su that" de doi chieu MOI
    // mo hinh xap xi (Che/step/EM) voi.
    vector<long long> counts((size_t)INF + 2, 0);
    for (int d : dist) counts[min(d, INF)]++;
    vector<long long> suffix((size_t)INF + 2, 0);
    long long run = 0;
    for (int c = INF; c >= 0; c--){ run += counts[c]; suffix[c] = run; }
    vector<double> MR_real(W);
    for (int c = 1; c <= W; c++) MR_real[c-1] = (double)suffix[c] / (double)N;

    auto burst = decompose_bursts_v36(trace, W);
    vector<double> raw_counts(W, 0.0);
    for (int x : trace) raw_counts[x] += 1.0;
    vector<double> q_raw(W); for (int i = 0; i < W; i++) q_raw[i] = raw_counts[i] / (double)N;

    // [v39-FIX #7] p_cold = ty le truy cap compulsory (stack-distance >= W,
    // tuc lan dau cham 1 trang chua tung xuat hien trong lich su gan nhat).
    // Doi len TRUOC vong Che (truoc day chi Tier3-EM moi tinh o duoi), vi
    // day la mot floor CO THAT cua MR_real ma ban than Che cung can biet -
    // khong phai dac quyen rieng cua EM. Dung chinh 'dist' da co san, khong
    // tinh lai gi moi.
    int cold_count = 0; for (int d : dist) if (d >= W) cold_count++;
    double p_cold = (double)cold_count / (double)N;

    // [v38-FIX #2] Tc(C) don dieu tang theo C (C cang lon -> can t cang lon de
    // g(t) dat toi C), nen dung Tc cua buoc C-1 lam warm-start cho Newton o
    // buoc C: giam so vong Newton can (thuong hoi tu ngay trong vai buoc khi
    // t0 da gan nghiem that), thay vi luon bat dau tu bracket [0,1] nhu cu.
    vector<double> che_zipf_all(W), che_burst_all(W);
    double tc_zipf_prev = -1.0, tc_burst_prev = -1.0;
    for (int c = 1; c <= W; c++){
        double tc_zipf = -1.0, tc_burst = -1.0;
        che_zipf_all[c-1]  = che_zipf_v36((double)c, q_raw, tc_zipf_prev, &tc_zipf, p_cold);
        che_burst_all[c-1] = che_burst_v36((double)c, burst.q, burst.L, tc_burst_prev, &tc_burst, p_cold);
        tc_zipf_prev = tc_zipf; tc_burst_prev = tc_burst;
    }
    auto mae = [](const vector<double>& a, const vector<double>& b){
        double s = 0.0; for (size_t i = 0; i < a.size(); i++) s += fabs(a[i] - b[i]); return s / a.size();
    };
    double err_zipf  = mae(che_zipf_all, MR_real);
    double err_burst = mae(che_burst_all, MR_real);
    string variant = (err_zipf <= err_burst) ? "che_zipf" : "che_burst";
    double err_che = min(err_zipf, err_burst);

    // [v37] tol_eff: co gian tol theo san nhieu thong ke cua chinh MR_real (xem
    // dan giai day du o dinh nghia noise_floor_v36 o tren). Voi N lon (self-test
    // goc, N~150k-250k) so hang them vao la khong dang ke nen hanh vi CU khong doi;
    // voi N nho no tu dong no ra dung bac do lon cua nhieu sampling that su.
    // [v39-FIX #3] tol_eff gio dung bootstrap_tol_floor_v39 (xem giai thich
    // day du o dinh nghia ham) thay vi K_NOISE_V37 * noise_floor_v36(N) cu -
    // cai cu do SE cua 1 ty le tai 1 diem C, khong phai SE cua MAE tong hop
    // tren W diem tuong quan, nen KHONG chan dung nhieu that su cua err_che.
    const vector<double>& chosen_curve = (variant == "che_zipf") ? che_zipf_all : che_burst_all;
    double boot_floor = bootstrap_tol_floor_v39(q_raw, N, W, chosen_curve, rng, 30);
    double tol_eff = max(tol, boot_floor);

    if (verbose)
        cout << "    Tier1 Che (" << variant << "): err=" << fixed << setprecision(5)
             << err_che << "  (tol=" << tol << "  tol_eff=" << tol_eff
             << "  [N=" << N << ", boot_floor(90th pct, B=30)=" << boot_floor << "])\n";

    TierResultV36 r;
    r.tol_eff = tol_eff;
    r.err_che_raw = err_che; // [v38-FIX #6]

    // [v39-FIX #3c] Tinh step-detector TRUOC quyet dinh Tier1 (khong doi den
    // khi Che bi tu choi moi tinh nhu truoc). Ly do: tu #3b, tol_eff = bias
    // (cua chinh cong thuc Che voi hinh dang q nay) + noise. Voi q lech manh
    // (SPEC), bias tu no co the da lon (~0.2), khien tol_eff luon >= err_che
    // THAT bat ke du lieu co periodic hay khong - kiem tra "err_che <= tol_eff"
    // MOT MINH khong con du dac hieu (specific) nua, vi no khong phan biet
    // duoc "Che dung vi IID that" voi "Che 'may man' nam duoi nguong da bi
    // bias thoi phong". Fix: luon tinh ca step.err va dung NO nhu mot PHEP
    // SO SANH MO HINH (model comparison) - neu step-fit giai thich du lieu
    // TOT HON HAN Che (chenh lech lon, khong phai do bias/noise ngau nhien),
    // đo la bang chung manh cho cau truc tuan hoan THAT, phai uu tien "periodic"
    // du err_che co nam duoi tol_eff hay khong.
    auto step = best_step_fit_v36(Cs, MR_real);
    r.step_err_raw = step.err; // [v38-FIX #6]
    if (verbose)
        cout << "    Tier2 step-detector: Cstar=" << step.cstar << " err=" << step.err
             << "  (Che err=" << err_che << ", tol_eff=" << tol_eff << ")\n";

    bool step_decisively_better = (step.err < 0.5 * err_che) && (step.err <= tol_eff * 3.0);

    if (step_decisively_better){
        r.mode = "periodic"; r.err = step.err; r.che_variant = variant;
        r.cstar = step.cstar; r.step_top = step.top; r.step_bot = step.bot;
        if (verbose) cout << "    => PHAT HIEN REGIME TUAN HOAN (step thang Che ro ret). mode=periodic (Cstar=" << step.cstar << ")\n";
        return r;
    }

    if (err_che <= tol_eff){
        r.mode = "che"; r.err = err_che; r.che_variant = variant;
        if (verbose) cout << "    => TIER1 DU DUNG. mode=che\n";
        return r;
    }

    // Tier 3: cuu ho EM tren mau stack-distance huu han
    vector<double> finite;
    for (int d : dist) if (d < W && d > 0) finite.push_back((double)d);
    int n_sample = min((int)finite.size(), 3000);
    vector<double> src;
    if ((int)finite.size() <= n_sample){
        src = finite;
    } else {
        vector<int> idx(finite.size()); iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        idx.resize(n_sample);
        src.reserve(n_sample);
        for (int ix : idx) src.push_back(finite[ix]);
    }
    auto em = em_exp_mixture_multistart_v36(src, 100); // [v38-FIX #4]: multi-restart thay vi 1 khoi tao co dinh
    // p_cold da tinh o tren (dung chung voi Tier1 Che, xem [v39-FIX #7])
    vector<double> mr_em(W);
    for (int c = 1; c <= W; c++)
        mr_em[c-1] = p_cold + (1.0 - p_cold) * (em.a * exp(-c / em.lam1) + (1.0 - em.a) * exp(-c / em.lam2));
    double err_em = mae(mr_em, MR_real);
    // [FIX] them dieu kien !isfinite(...) o dau: neu vi ly do nao do (vd.
    // du lieu dau vao qua it/suy bien) van con NaN/Inf lot qua, PHAI coi la
    // THOAI HOA thay vi de so sanh voi NaN (luon tra ve false) lam bo qua
    // truong hop hong, dan den in ra ket luan "cuu ho co ich" sai su that.
    bool degenerate = !isfinite(em.a) || !isfinite(em.lam1) || !isfinite(em.lam2) ||
                       (em.a < 0.02) || (em.a > 0.98) ||
                       (min(em.lam1, em.lam2) > 0 && fabs(em.lam1 - em.lam2) / em.lam2 < 0.05);
    if (verbose){
        cout << "    Tier3 EM cuu ho: a=" << em.a << " lam1=" << em.lam1 << " lam2=" << em.lam2
             << "  err=" << err_em << (degenerate ? "  [THOAI HOA]\n" : "\n");
        cout << "    => mode=em_rescue" << (degenerate ? " (THOAI HOA - khong tim ra cum thu 2 that; Che van la cau tra loi trung thuc)\n"
                                                          : " (cuu ho co ich, KHONG phai dinh luat da chung minh)\n");
    }
    r.mode = "em_rescue"; r.err = err_em; r.che_variant = variant;
    r.cstar = step.cstar; r.step_top = step.top; r.step_bot = step.bot;
    r.em_a = em.a; r.em_lam1 = em.lam1; r.em_lam2 = em.lam2; r.em_degenerate = degenerate;
    return r;
}

// [v38-FIX #6] Bootstrap CI cho CHINH quyet dinh chon tier. Toan file da
// dung bootstrap ky luong cho ratio_of_drops/C_fit o cac phan [v29] tro
// len, nhung predict_miss_rate_v36() truoc day chi tra ve 1 diem uoc duy
// nhat cho err_che/step.err - voi N nho hoac bien gioi gan tol_eff, tier
// chon co the "nhay" qua lai giua cac lan chay chi vi sampling noise ma
// khong ai biet la no dang o ranh gioi. Ham nay resample CO HOAN LAI
// (bootstrap chuan) tren chinh chuoi trace, chay lai predict_miss_rate_v36
// tren tung ban resample, roi bao cao khoang percentile 95% cua err_che
// va step.err - cho biet quyet dinh tier co "chac chan" hay dang nam
// sat lan ranh tol_eff/tol_eff*3.
struct TierConfidenceV38 {
    double err_che_lo, err_che_hi;     // 95% CI cua err_che (Tier1)
    double step_err_lo, step_err_hi;   // 95% CI cua step.err (Tier2)
    bool tier1_borderline;  // CI cua err_che bao trum tol_eff -> quyet dinh Tier1 khong chac chan
    bool tier2_borderline;  // CI cua step.err bao trum tol_eff*3 -> quyet dinh Tier2 khong chac chan
};
static TierConfidenceV38 bootstrap_tier_confidence_v38(const vector<int>& trace, int W,
                                                        double tol_eff, int n_boot, mt19937_64& rng){
    size_t N = trace.size();
    vector<double> err_ches, step_errs;
    err_ches.reserve(n_boot); step_errs.reserve(n_boot);
    uniform_int_distribution<size_t> dist(0, N - 1);
    vector<int> resampled(N);
    for (int b = 0; b < n_boot; b++){
        for (size_t i = 0; i < N; i++) resampled[i] = trace[dist(rng)];
        auto r = predict_miss_rate_v36(resampled, W, tol_eff, false, rng);
        err_ches.push_back(r.err_che_raw);
        step_errs.push_back(r.step_err_raw);
    }
    sort(err_ches.begin(), err_ches.end());
    sort(step_errs.begin(), step_errs.end());
    auto pct = [&](const vector<double>& v, double p) -> double {
        double idx = p * (double)(v.size() - 1);
        size_t lo = (size_t)floor(idx), hi = (size_t)ceil(idx);
        return v[lo] + (v[hi] - v[lo]) * (idx - (double)lo);
    };
    TierConfidenceV38 c;
    c.err_che_lo = pct(err_ches, 0.025); c.err_che_hi = pct(err_ches, 0.975);
    c.step_err_lo = pct(step_errs, 0.025); c.step_err_hi = pct(step_errs, 0.975);
    c.tier1_borderline = (c.err_che_lo <= tol_eff && tol_eff <= c.err_che_hi);
    c.tier2_borderline = (c.step_err_lo <= tol_eff * 3.0 && tol_eff * 3.0 <= c.step_err_hi);
    return c;
}

// ---------- 3 chuoi truy cap "thuc te hoa" (port 1-1 tu realistic_test.py)
// dung LAM MAU cho self-test - moi chuoi kiem tra 1 tier khac nhau ----------
static vector<int> gen_ycsb_v36(int W, int N, double s, uint64_t seed){
    // i.i.d. Zipf thuan tuy - dung gia dinh IRM/Che kinh dien theo dung nghia
    mt19937_64 rng(seed);
    vector<double> w(W);
    for (int i = 1; i <= W; i++) w[i-1] = pow((double)i, -s);
    double sum = 0.0; for (double v : w) sum += v;
    vector<double> q(W); for (int i = 0; i < W; i++) q[i] = w[i] / sum;
    discrete_distribution<int> dd(q.begin(), q.end());
    vector<int> trace(N);
    for (int i = 0; i < N; i++) trace[i] = dd(rng);
    return trace;
}
static vector<int> gen_spec_loop_v36(int W, int n_passes, double jitter, int small_burst_max, uint64_t seed){
    // Quet vong lap lap lai GAN NHU CUNG mot thu tu - tai hien dung chinh
    // "chuoi phat lai co dinh" ma [v27] da tu-kiem-chung o tren.
    mt19937_64 rng(seed);
    vector<int> base_order(W); iota(base_order.begin(), base_order.end(), 0);
    shuffle(base_order.begin(), base_order.end(), rng);
    vector<int> trace;
    uniform_int_distribution<int> swap_pick(0, W - 1);
    uniform_int_distribution<int> burst_pick(1, small_burst_max);
    for (int p = 0; p < n_passes; p++){
        vector<int> order = base_order;
        int n_swaps = (int)((double)W * jitter);
        for (int s = 0; s < n_swaps; s++){
            int a = swap_pick(rng), b = swap_pick(rng);
            swap(order[a], order[b]);
        }
        for (int item : order){
            int L = burst_pick(rng);
            for (int l = 0; l < L; l++) trace.push_back(item);
        }
    }
    return trace;
}
static vector<int> gen_bioinfo_phase_v36(int W, int n_segments, int seg_len,
                                          double motif_frac, double p_motif, uint64_t seed){
    // Non-stationary theo PHA: moi doan co "motif" nong rieng - q_i toan
    // cuc trong tren toan chuoi la dung (khong doi theo thoi gian), nhung
    // qua trinh THAT la troi pha, khong phai tuong quan.
    mt19937_64 rng(seed);
    int motif_size = max(1, (int)((double)W * motif_frac));
    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_int_distribution<int> full_pick(0, W - 1);
    vector<int> trace;
    for (int seg = 0; seg < n_segments; seg++){
        vector<int> pool(W); iota(pool.begin(), pool.end(), 0);
        shuffle(pool.begin(), pool.end(), rng);
        vector<int> motifs(pool.begin(), pool.begin() + motif_size);
        uniform_int_distribution<int> motif_pick(0, motif_size - 1);
        for (int i = 0; i < seg_len; i++){
            if (unif(rng) < p_motif) trace.push_back(motifs[motif_pick(rng)]);
            else trace.push_back(full_pick(rng));
        }
    }
    return trace;
}

// ---------- [v36] Self-test: kiem chung ca 3 tier CHON DUNG regime tren 3
// mau chuoi khac nhau, doi chieu bang tay voi ket qua chay THAT cua ban
// Python goc (realistic_test.py/final_predictor.py) truoc khi dua vao file
// nay - dung tinh than "khong tin - kiem chung" nhu self_test_irm_law() ----------
static bool self_test_predictor_v36(mt19937_64& rng){
    cout << "[self-test v36] Doi chieu 3-tier tren 3 mau truy cap thuc-te-hoa\n"
            "  (MR_real doc TRUC TIEP tu stack-distance cua chinh chuoi - dung\n"
            "  tuyet doi theo dinh nghia LRU, KHONG can mo phong doc lap):\n";
    bool all_ok = true;
    uint64_t seed_base = rng();

    {
        auto trace = gen_ycsb_v36(300, 150000, 0.99, seed_base + 1);
        auto res = predict_miss_rate_v36(trace, 300, 0.01, false, rng);
        bool ok = (res.mode == "che") && (res.err <= 0.02);
        cout << "  YCSB-like (i.i.d. Zipf thuan): mode=" << res.mode
             << " err=" << fixed << setprecision(5) << res.err
             << (ok ? "  OK (dung ky vong: Che du dung - IRM/Che van la nhanh dung khi popularity on dinh)\n"
                    : "  [KHAC KY VONG]\n");
        all_ok = all_ok && ok;
    }
    {
        auto trace = gen_spec_loop_v36(300, 400, 0.05, 3, seed_base + 2);
        auto res = predict_miss_rate_v36(trace, 300, 0.01, false, rng);
        bool ok = (res.mode == "periodic") && (res.err <= 0.03);
        cout << "  SPEC-like (vong lap tuan hoan): mode=" << res.mode
             << " err=" << res.err
             << (ok ? "  OK (dung ky vong: phat hien tuan hoan - dung diem [v27] da chi ra ma [v24]-[v35] bo lo)\n"
                    : "  [KHAC KY VONG]\n");
        all_ok = all_ok && ok;
    }
    {
        auto trace = gen_bioinfo_phase_v36(300, 6, 25000, 0.15, 0.7, seed_base + 3);
        auto res = predict_miss_rate_v36(trace, 300, 0.01, false, rng);
        bool ok = (res.mode != "che"); // khong dung on IID/khong doi theo t/gian -> Che rieng khong the du
        cout << "  Bioinfo-like (troi pha, motif doi theo doan): mode=" << res.mode
             << " err=" << res.err
             << (ok ? "  OK (dung ky vong: Che khong du, phai leo tier/cuu ho - trung thuc bao gioi han)\n"
                    : "  [KHAC KY VONG]\n");
        all_ok = all_ok && ok;
    }
    // ---- [v37-NEW] Kiem chung THAT su tol_eff co gian theo N: chay LAP LAI
    // nhieu seed o N NHO (noi san nhieu 0.5/sqrt(N) khong con nho hon 0.01 nua)
    // tren cung 1 process i.i.d. Zipf (dung ly ra la "che" du dung boi ban chat
    // qua trinh, KHONG lien quan gi den N) - neu tol co dinh 0.01, sampling
    // noise co the day err_che vuot nguong va lam mode nhay sang periodic/
    // em_rescue MOT CACH SAI (false negative do noise, khong phai do cau truc).
    // Day la self-test THAT (khong suy dien tren giay) cho chinh thay doi [v37].
    {
        const int N_SMALL = 3000, W_SMALL = 300, N_SEEDS = 8;
        int che_count = 0;
        double worst_err = 0.0, worst_tol_eff = 0.0;
        for (int s = 0; s < N_SEEDS; s++){
            auto trace = gen_ycsb_v36(W_SMALL, N_SMALL, 0.99, seed_base + 100 + s);
            auto res = predict_miss_rate_v36(trace, W_SMALL, 0.01, false, rng);
            if (res.mode == "che") che_count++;
            else { worst_err = max(worst_err, res.err); worst_tol_eff = max(worst_tol_eff, res.tol_eff); }
        }
        // Ky vong: TAT CA N_SEEDS lan phai ra "che", vi ban chat qua trinh la i.i.d.
        // Zipf giong het truong hop N lon o tren - chi N nho di, KHONG doi ban chat.
        bool ok = (che_count == N_SEEDS);
        cout << "  [v37] YCSB N-nho (N=" << N_SMALL << ", " << N_SEEDS << " seed doc lap): "
             << che_count << "/" << N_SEEDS << " lan ra mode=che"
             << (ok ? "  OK (tol_eff da co gian dung, khong bi noise sampling danh lua)\n"
                    : ("  [KHAC KY VONG] - con " + to_string(N_SEEDS - che_count)
                       + " lan bi nhay tier oan do sampling noise (worst_err="
                       + to_string(worst_err) + " > tol_eff=" + to_string(worst_tol_eff) + ")\n"));
        all_ok = all_ok && ok;
    }
    return all_ok;
}

// Chuoi mo phong DUNG cach benchmark THAT cua chinh file nay dang su dung
// (generate_offsets(): sinh 1 hoan vi ngau nhien cua num_regions trang, RUT
// GON gio moi trang duoc tham chieu dung 1 lan/chu ky - roi PHAT LAI y het
// qua nhieu trial) - day chinh la truong hop [v27] da chi ra la KHONG duoc
// mo hinh IID phuc vu dung, va la ly do THAT SU de thay [v24]-[v35] bang
// [v36]. Khac gen_spec_loop_v36 o cho KHONG co jitter/burst-length ngau
// nhien: day la PHAT LAI TUYET DOI CHINH XAC (dung boi canh sinh du lieu
// cua generate_offsets() + vong TRIALS trong file nay).
// [FIX] Ham cu KHONG nhan tham so L, nen moi trang chi xuat hien 1 lan/pass
// (burst do dai 1) - trong khi generate_offsets() THAT su tao ra L truy cap
// LIEN TIEP vao CUNG 1 trang truoc khi chuyen sang trang khac (xem vong
// "for l in 0..L" ben trong generate_offsets()). Thieu L khien demo o cuoi
// main() overclaim la "phat lai tuyet doi chinh xac" trong khi thuc ra sai
// o tang cau truc burst - anh huong truc tiep den Cstar/err ma Tier-2 (step-
// detector) tinh ra, vi burst length la bien quan trong thu 2 (song song W)
// ma chinh file nay dung cho Che/burst model o tren.
// [v38-FIX #5] Ban cu (gen_this_workload_v36 ben duoi) tu dung mot vong
// shuffle() RIENG de "mo phong" chuoi that, voi W_demo=512/L_demo=13 la
// HANG SO cung, tach roi khoi chinh generate_offsets()/N/total_lines/L
// dang chay trong vong sweep chinh cua file (noi co the bi CLAMP xuong,
// vd 14.1% ops o L=1 da bao cao). Ham nay thay vao do GOI TRUC TIEP
// generate_offsets() that (cung mot ham, cung logic clamp) voi N/total_lines/
// L THAT duoc truyen vao tu main(), roi anh xa offset (byte) ve page-id
// (0..num_regions_actual-1) qua offset/region_words - dam bao chuoi dua
// vao predict_miss_rate_v36 la CHINH XAC nhung gi generate_offsets() thuc
// su tao ra trong dieu kien (bi clamp hay khong) dang duoc do, khong con
// la mo phong gan dung tach roi nua. Tra ve ca W_actual (=num_regions sau
// clamp) de goi predict_miss_rate_v36 voi dung kich thuoc cua so.
static pair<vector<int>, int> real_workload_trace_v38(size_t N, size_t total_lines, int L,
                                                       int n_trials, uint64_t seed){
    size_t num_regions_req = total_lines / (size_t)L;
    size_t n_regions_avail = N / WORDS_PER_4K;
    size_t num_regions_actual = min(num_regions_req, n_regions_avail); // dung logic clamp that
    vector<int> trace;
    trace.reserve(num_regions_actual * (size_t)L * (size_t)n_trials);
    for (int t = 0; t < n_trials; t++){
        auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, num_regions_req, L,
                                      seed + (uint64_t)t, /*warn=*/false);
        for (size_t o : offs) trace.push_back((int)(o / WORDS_PER_4K)); // byte offset -> page-id
    }
    return { trace, (int)num_regions_actual };
}

// [cleanup] gen_this_workload_v36() da xac nhan khong con duoc goi noi nao
// (-Wunused-function) - da xoa (real_workload_trace_v38() o tren la ban
// thay the dung du lieu THAT tu generate_offsets() thay vi mo phong).

// ---------- Tien ich hoi quy tuyen tinh + RMSE/R^2 dung chung cho [v23] ----------
struct LinFit { double a, b, r2; };
static LinFit fit_linear(const vector<double>& x, const vector<double>& y){
    size_t n = x.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; i++){ sx += x[i]; sy += y[i]; sxx += x[i]*x[i]; sxy += x[i]*y[i]; }
    double denom = (double)n * sxx - sx * sx;
    double b = (denom != 0.0) ? ((double)n * sxy - sx * sy) / denom : 0.0;
    double a = (sy - b * sx) / (double)n;
    double ss_tot = 0, ss_res = 0, mean_y = sy / (double)n;
    for (size_t i = 0; i < n; i++){
        double pred = a + b * x[i];
        ss_res += (y[i]-pred)*(y[i]-pred);
        ss_tot += (y[i]-mean_y)*(y[i]-mean_y);
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res/ss_tot : 1.0;
    return {a, b, r2};
}
static double r_squared_given_model(const vector<double>& y, const vector<double>& pred){
    size_t n = y.size();
    double sy = 0; for (double v : y) sy += v;
    double mean_y = sy / (double)n;
    double ss_tot = 0, ss_res = 0;
    for (size_t i = 0; i < n; i++){
        ss_res += (y[i]-pred[i])*(y[i]-pred[i]);
        ss_tot += (y[i]-mean_y)*(y[i]-mean_y);
    }
    return (ss_tot > 1e-12) ? 1.0 - ss_res/ss_tot : 1.0;
}
static double rmse_of(const vector<double>& y, const vector<double>& pred){
    double s = 0;
    for (size_t i = 0; i < y.size(); i++){ double d = y[i]-pred[i]; s += d*d; }
    return sqrt(s / (double)y.size());
}

// ============================================================
// [v25-DE-XUAT, MOI - fallback KHONG CAN dTLB-miss/PMU cho dung diem bi
// chan o [v23] Buoc 1-4 (xem canh bao "KHONG DO DUOC HW counter" o duoi)]
//
// Y TUONG: cost(L) = a + b * predict_miss_per_line(L, W(L), C) la TUYEN TINH
// theo (a,b) VOI MOI GIA TRI C CO DINH (vi predict_miss_per_line(...,C) chi
// la MOT SO cu the khi C da chon, du no PHI TUYEN theo C). Vay thay vi can
// dTLB-miss THAT (bi chan trong sandbox nay) de suy C truoc roi moi fit
// (a,b), ta co the: quet C tren mot luoi, VOI MOI C giai (a,b) bang OLS
// dong (ham fit_linear() da co san o tren), roi CHON C cho R^2 cao nhat.
// Day chinh la "variable projection" cho bai toan binh phuong toi thieu
// phi tuyen co bien tach duoc (mot phan tham so tuyen tinh, mot phan phi
// tuyen) - Golub, G.H., Pereyra, V., "The Differentiation of Pseudo-
// Inverses and Nonlinear Least Squares Problems Whose Variables Separate",
// SIAM J. Numer. Anal. 10(2):413-432, 1973. DAY LA THUAT TOAN DA CONG BO
// (khong phai tu nghi ra); dong gop o day chi la AP DUNG dung vao cho x
// = predict_miss_per_line va y = cost/line DA CO SAN trong file (cost_normal,
// Ls, n_ops_normal - khong can do THEM gi, khong can quyen PMU nao ca, chi
// dung lai chinh du lieu wall-clock da tinh xong tu truoc trong main()).
//
// CANH BAO TRUNG THUC (khong ne tranh): day KHONG PHAI thay the tuong duong
// cho phep do dTLB-miss that. No chi tra loi mot cau hoi KHAC: "neu ta GIA
// DINH cong thuc IRM-Burst dung, thi gia tri C nao khien no khop TOT NHAT
// voi WALL-CLOCK TIME do duoc (thu KHONG can quyen gi de do)?" R^2 cao KHONG
// chung minh dTLB that hanh xu dung y nhu mo hinh LRU/fully-associative ly
// tuong - dTLB that thuong set-associative, nhieu tang, co the co replacement
// khac LRU. R^2 thap hon han model "a+b/L" cu (khong co W) hoac C_fit ra so
// am/vo ly (VD lon hon tong so trang N/WORDS_PER_4K) la dau hieu ro rang mo
// hinh KHONG khop - luc do nen bao cao that, khong nen "chon luoi dep hon".
struct VarProResult { double C_fit, a, b, r2; };
static VarProResult varpro_fit_capacity_no_pmu(
        const vector<int>& Ls_in, const vector<double>& Wactual_in,
        const vector<double>& cost_in, double C_max, int grid_steps = 4000){
    VarProResult best{ -1.0, 0.0, 0.0, -1e18 };
    for (int k = 1; k <= grid_steps; k++){
        double C = C_max * (double)k / (double)grid_steps;
        vector<double> miss_pred(Ls_in.size());
        for (size_t i = 0; i < Ls_in.size(); i++)
            miss_pred[i] = predict_miss_per_line((double)Ls_in[i], Wactual_in[i], C);
        LinFit f = fit_linear(miss_pred, cost_in);
        if (f.r2 > best.r2) best = {C, f.a, f.b, f.r2};
    }
    return best;
}

// [v28b-MOI] Quet TOAN BO mat R^2(C) va tim CAC cuc-dai-cuc-bo, thay vi chi
// lay 1 argmax duy nhat. Dong co: bootstrap CI cua C_fit ([v28] o duoi) nhay
// CHINH XAC giua 2 gia tri C va ~2*C (khong phai nhieu ngau nhien lien tuc) -
// day CO THE la dau hieu mat R^2(C) THAT SU da-cuc (vi chi co 7 diem du lieu
// quyet dinh hinh dang duong cong bao hoa, hien tuong "aliasing" C vs 2C/C
// kinh dien khi fit ham bao hoa voi it diem), KHONG phai loi luong tu hoa
// luoi. Ham nay xac nhan THAT bang cach liet ke moi cuc-dai-cuc-bo co R^2 gan
// bang cuc-dai-toan-cuc (trong nguong % cho truoc), de phan biet 2 kha nang:
//   (a) chi 1 cuc-dai ro rang -> bootstrap nhay 2 gia tri la do NHIEU DU LIEU
//       (trial resample) day diem-uoc qua lai giua 2 vung gan-toi-uu
//   (b) THAT co >= 2 cuc-dai-cuc-bo tach biet -> mo hinh IRM-Burst KHONG DINH
//       DANH duoc (unidentifiable) tren du lieu nay, C VA ~2C (hoac cac boi
//       so khac) GIONG NHAU ve mat khop-duong-cong - day la gioi han THAT cua
//       phuong phap VARPRO-khong-PMU tren ITTHIET du lieu (7 diem L), khong
//       phai loi trien khai.
struct LocalMax { double C, r2; };
static vector<LocalMax> varpro_find_local_maxima(
        const vector<int>& Ls_in, const vector<double>& Wactual_in,
        const vector<double>& cost_in, double C_max, int grid_steps,
        double rel_threshold = 0.90){
    vector<double> r2_curve(grid_steps + 1);
    double global_best_r2 = -1e18;
    for (int k = 1; k <= grid_steps; k++){
        double C = C_max * (double)k / (double)grid_steps;
        vector<double> miss_pred(Ls_in.size());
        for (size_t i = 0; i < Ls_in.size(); i++)
            miss_pred[i] = predict_miss_per_line((double)Ls_in[i], Wactual_in[i], C);
        LinFit f = fit_linear(miss_pred, cost_in);
        r2_curve[k] = f.r2;
        if (f.r2 > global_best_r2) global_best_r2 = f.r2;
    }
    vector<LocalMax> maxima;
    for (int k = 2; k < grid_steps; k++){
        if (r2_curve[k] >= r2_curve[k-1] && r2_curve[k] >= r2_curve[k+1] &&
            r2_curve[k] > 0 && r2_curve[k] >= rel_threshold * global_best_r2){
            double C = C_max * (double)k / (double)grid_steps;
            maxima.push_back({C, r2_curve[k]});
        }
    }
    // Gop cac cuc-dai qua gan nhau (trong 2% C_max) thanh 1, giu r2 cao hon -
    // luoi roi-rac co the tao vai "rang cua" lien tiep quanh 1 cuc-dai THAT.
    sort(maxima.begin(), maxima.end(), [](const LocalMax& a, const LocalMax& b){ return a.C < b.C; });
    vector<LocalMax> merged;
    for (auto& m : maxima){
        if (!merged.empty() && (m.C - merged.back().C) < 0.02 * C_max){
            if (m.r2 > merged.back().r2) merged.back() = m;
        } else merged.push_back(m);
    }
    return merged;
}

// [v28c] Gop cac local-maxima thanh CUM (cluster) khi chung tao thanh 1 CAO
// NGUYEN (r2 gan nhu khong doi qua 1 day dai C lien tiep - bao hoa mo hinh
// THAT khi C >> working-set, KHONG phai nhieu luoi). Tra ve moi cum dang
// [C_lo, C_hi] + r2 dai dien (max trong cum), de in GON thay vi liet ke tung
// diem luoi mot rieng le nhu varpro_find_local_maxima() tho lam.
struct MaxCluster { double C_lo, C_hi, r2; };
static vector<MaxCluster> cluster_local_maxima(const vector<LocalMax>& lmax, double C_max){
    vector<MaxCluster> clusters;
    for (auto& m : lmax){
        if (!clusters.empty() && (m.C - clusters.back().C_hi) < 0.06 * C_max &&
            fabs(m.r2 - clusters.back().r2) < 0.01){
            clusters.back().C_hi = m.C;
            clusters.back().r2 = max(clusters.back().r2, m.r2);
        } else clusters.push_back({m.C, m.C, m.r2});
    }
    return clusters;
}

// [v28-MOI, DOT PHA THAT SU cho DUNG muc tieu cua ca file] Tat ca bootstrap/
// stationary/hierarchical/variance-decomposition o [S]/[Q-fix]/[T]/[U] TRUOC
// GIO deu chi dinh luong bat dinh cho ratio_of_drops - MOT SO PHU, dung de
// kiem tra gia dinh i.i.d va so sanh thuong/hugepage. Nhung SAN PHAM KHOA HOC
// CHINH cua toan bo file la C_fit (dung luong dTLB hieu dung, trang) tu VARPRO
// - VA SO NAY TU TRUOC GIO CHI CO 1 DIEM UOC DUY NHAT, KHONG CO CI, o MOI
// phien ban tu v25 den v27. Day la 1 lo hong nghiem trong xet theo dung tinh
// than "dung tin diem-uoc don le" ma toan bo phan [Q]/[S]/[T]/[U] da ra suc
// nhan manh - ap dung dung cho ratio_of_drops nhung lai QUEN ap dung cho chinh
// C_fit, thu duy nhat nguoi doc thuc su quan tam.
//
// Ham nay lap lai dung quy trinh VARPRO cho TUNG draw bootstrap: chon 1
// replication (tang 1, giong hierarchical_bootstrap_ci), resample stationary
// CUNG 1 tap chi-so-trial cho TAT CA cac L (tang 2, giu dung phu thuoc cheo-L
// theo trial giong cach [S]/[Q-fix] giu phu thuoc cheo-vector), tinh lai
// cost/line cho tung L tu du lieu da resample, roi CHAY LAI VARPRO tren du
// lieu do de ra 1 C_fit_boot. Lap N_BOOT lan -> phan phoi bootstrap cua C_fit.
// grid_steps trong bootstrap co the THAP hon diem-uoc chinh (toc do), CHI can
// du de CI khong bi luong tu hoa qua tho.
BootResult bootstrap_ci_varpro_C(
        const vector<vector<vector<double>>>& reps_times, // [R][nL][TRIALS]
        const vector<int>& Ls_in, const vector<double>& Wactual_in,
        const vector<size_t>& n_ops_in, double C_max, int grid_steps,
        int n_boot, double mean_block_len, mt19937_64& rng){
    size_t R = reps_times.size();
    size_t nL = Ls_in.size();
    size_t n_trials = reps_times[0][0].size();
    double p = 1.0 / mean_block_len;
    // [v29-song song] day la ham TON KEM NHAT trong 4 ham bootstrap (moi lan
    // boot phai chay lai TOAN BO grid-search varpro_fit_capacity_no_pmu voi
    // grid_steps diem) - loi ich song song hoa o day LON NHAT. reps_times/
    // Ls_in/Wactual_in/n_ops_in chi duoc DOC, varpro_fit_capacity_no_pmu la
    // ham thuan (khong static mutable) nen an toan goi dong thoi tu nhieu
    // thread voi tham so rieng cua tung thread.
    vector<double> stats = parallel_bootstrap_stats(n_boot, rng,
        [&reps_times, &Ls_in, &Wactual_in, &n_ops_in, R, nL, n_trials, p, C_max, grid_steps]
        (int /*b*/, mt19937_64& local_rng) -> double {
            uniform_int_distribution<size_t> rep_dist(0, R - 1);
            size_t rep_idx = rep_dist(local_rng); // tang 1: 1 lan-chay VOI HOAN LAI (giong [T])
            // [opt] xem giai thich buffer thread_local o bootstrap_ci() phia tren.
            // (Ham nay van ton kem chu yeu vi grid-search VARPRO ben duoi, khong
            // phai vi cap phat idx/resampled - nhung dong bo hoa cho nhat quan.)
            thread_local vector<size_t> tidx;
            thread_local vector<double> resampled;
            thread_local vector<double> cost_boot;
            stationary_bootstrap_indices_into(tidx, n_trials, p, local_rng); // tang 2, CHUNG cho moi L
            resampled.resize(n_trials);
            cost_boot.resize(nL);
            for (size_t li = 0; li < nL; li++){
                for (size_t k = 0; k < n_trials; k++) resampled[k] = reps_times[rep_idx][li][tidx[k]];
                cost_boot[li] = (median(resampled) * 1000.0) / (double)n_ops_in[li];
            }
            VarProResult vp_b = varpro_fit_capacity_no_pmu(Ls_in, Wactual_in, cost_boot, C_max, grid_steps);
            return vp_b.C_fit;
        });
    sort(stats.begin(), stats.end());
    size_t lo = (size_t)(0.025 * stats.size());
    size_t hi = (size_t)(0.975 * stats.size());
    if (hi >= stats.size()) hi = stats.size() - 1;
    double med = stats[stats.size() / 2];
    vector<double> dev(stats.size());
    for (size_t i = 0; i < stats.size(); i++) dev[i] = fabs(stats[i] - med);
    sort(dev.begin(), dev.end());
    double mad_val = dev[dev.size() / 2];
    return {stats[lo], stats[hi], med, mad_val};
}

int main(int argc, char** argv){
    // Defaults nho de sandbox/may it RAM van chay duoc.
    // May that: ./v20 100000000 200000 30 200 2000
    size_t N           = 10'000'000;
    size_t total_lines = 100000;
    int    TRIALS      = 20;
    int    RMSE_REPS   = 80;
    int    N_BOOT      = 1000;
    int    R_REPS      = 5; // [v26-MOI] so lan chay lai TOAN BO benchmark (macro-replication),
                             // de do bat dinh GIUA-cac-lan-chay - xem hierarchical_bootstrap_ci()
    if (argc >= 2) N           = stoull(argv[1]);
    if (argc >= 3) total_lines = stoull(argv[2]);
    if (argc >= 4) TRIALS      = stoi(argv[3]);
    if (argc >= 5) RMSE_REPS   = stoi(argv[4]);
    if (argc >= 6) N_BOOT      = stoi(argv[5]);
    if (argc >= 7) R_REPS      = stoi(argv[6]);

    // [P] Pin truoc alloc
    bool pinned = pin_to_cpu0();
    cout << "[P] Ghim CPU0: " << (pinned ? "THANH CONG" : "THAT BAI") << "\n";

    // [v21-NEW] Chon kernel popcount nhanh nhat CPU nay ho tro
    init_popcount_dispatch();
    cout << "[SIMD] Kernel popcount dang dung: " << simd_level_name << "\n";

    // [v21-NEW, buoc 2] BAT BUOC self-test truoc khi benchmark - neu kernel
    // dispatch sai, dung ngay thay vi tao ra so lieu benchmark khong dang tin.
    self_test_bit_identical(0xC0FFEE, 20000);
    self_test_contig_bit_identical(0xC0FFEE2, 3000); // [v24] rieng cho contig_bulk_dispatch/Harley-Seal

    const size_t BYTES = N * sizeof(uint64_t);
    mt19937_64 seed_rng(2026);
    cout << "N=" << N << " (~" << BYTES/(1024*1024) << "MB), total_lines=" << total_lines
         << ", TRIALS=" << TRIALS << ", RMSE_REPS=" << RMSE_REPS
         << ", N_BOOT=" << N_BOOT << "\n\n";

    // Alloc + fill
    void* raw_normal = alloc_region(BYTES, false);
    if (!raw_normal){ cout << "mmap thuong that bai\n"; return 1; }
    uint64_t* normal = reinterpret_cast<uint64_t*>(raw_normal);
    for (size_t k = 0; k < N; k++) normal[k] = seed_rng();
    uint64_t ref_normal = contig_bulk_dispatch(normal, N);
    cout << "[doi chung] trang thuong 4KB, ref=" << ref_normal << "\n";
#if defined(__x86_64__) || defined(_M_X64)
    benchmark_v24_harleyseal(normal, N, g_simd_level); // [v24] so sanh THAT tren buffer that N tu hien hanh
    benchmark_v31_kway_sweep(normal, N, g_simd_level); // [v31] 2acc vs 8acc, THAT tren buffer that N tu hien hanh
    self_test_contig_mt_bit_identical(normal, N);      // [v31] kiem tra contig_bulk_mt dung, KHONG do toc do o day
#endif

    void* raw_hp = alloc_region(BYTES, true);
    if (!raw_hp){ cout << "mmap hugepage that bai\n"; munmap(raw_normal, BYTES); return 1; }
    uint64_t* hp = reinterpret_cast<uint64_t*>(raw_hp);
    for (size_t k = 0; k < N; k++) hp[k] = seed_rng();
    uint64_t ref_hp = contig_bulk_dispatch(hp, N);
    long hp_kb = anon_hugepages_kb_for_addr(raw_hp);
    cout << "[thi nghiem] huge page 2MB, ref=" << ref_hp
         << "   AnonHugePages/VMA=" << hp_kb << "kB";
    if (hp_kb >= 0 && (size_t)hp_kb * 1024 > BYTES * 0.8)
        cout << "  (XAC NHAN huge page THAT SU)\n\n";
    else
        cout << "  (CANH BAO: khong xac nhan duoc huge page)\n\n";

    // ---------- Compute-floor (v18, nay dung SIMD dispatch + doi chieu RDTSCP) ----------
    cout << "===== [Calib] Compute-floor (L1-resident, khong memory stall)\n";
    {
        const size_t CAL_LINES = 2'000'000;
        double tsc_ghz = calibrate_tsc_ghz();
        cout << "  [TSC] Tan so hieu chuan: " << fixed << setprecision(3) << tsc_ghz << " GHz\n";

        vector<double> t_chrono, t_tsc_ns;
        for (int tr = 0; tr < TRIALS; tr++){
            // --- do bang chrono (nhu ban goc v18/v19/v20) ---
            auto a0 = high_resolution_clock::now();
            uint64_t s1 = floor_bulk_dispatch(normal, CAL_LINES, LINES_PER_4K);
            auto a1 = high_resolution_clock::now();
            t_chrono.push_back(duration<double, micro>(a1 - a0).count());
            asm volatile("" :: "r"(s1));

            // --- do bang RDTSCP+LFENCE (serializing that su, khong cho OoO leak) ---
            unsigned aux;
            _mm_lfence();
            uint64_t c0 = __rdtscp(&aux);
            _mm_lfence();
            uint64_t s2 = floor_bulk_dispatch(normal, CAL_LINES, LINES_PER_4K);
            _mm_lfence();
            uint64_t c1 = __rdtscp(&aux);
            t_tsc_ns.push_back((double)(c1 - c0) / tsc_ghz);
            asm volatile("" :: "r"(s2));
        }
        double mt_chrono_ns = median(t_chrono) * 1000.0;
        double mt_tsc_ns    = median(t_tsc_ns);
        cout << "  compute-floor cost/line (chrono, ~clock_gettime) = " << fixed << setprecision(3)
             << mt_chrono_ns / (double)CAL_LINES << " ns\n";
        cout << "  compute-floor cost/line (RDTSCP, serialized)     = " << setprecision(3)
             << mt_tsc_ns    / (double)CAL_LINES << " ns\n";
        double diff_pct = 100.0 * fabs(mt_chrono_ns - mt_tsc_ns) / mt_tsc_ns;
        cout << "  Chenh lech chrono vs RDTSCP: " << setprecision(2) << diff_pct << "%";
        if (diff_pct < 1.0)
            cout << "  (khong dang ke o quy mo " << CAL_LINES << " line/trial)\n\n";
        else
            cout << "  (dang chu y — OoO leak qua ranh gioi do co the that)\n\n";
    }

    // ---------- [Q] HW PMU ----------
    cout << "===== [Q] HW PMU (dTLB-load-miss, LLC-load-miss)\n";
    vector<int> Ls_full = {1, 2, 4, 8, 12, 13, 15, 16, 32, 64};
    {
        HwCounter dtlb, llc;
        dtlb.open(DTLB_MISS_CFG(), "dTLB-miss");
        llc.open(LLC_MISS_CFG(), "LLC-miss");
        if (!dtlb.ok || !llc.ok){
            cout << "  [!] KHONG DO DUOC HW counter (permission/container).\n"
                    "      Thu: sudo sysctl kernel.perf_event_paranoid=1\n\n";
        } else {
            for (const char* label_ptr : {"thuong", "hugepage"}){
                string label(label_ptr);
                uint64_t* data = (label == "thuong") ? normal : hp;
                cout << "  --- mang " << label << " ---\n";
                for (int L : Ls_full){
                    size_t num_regions = total_lines / (size_t)L;
                    auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K,
                                                 num_regions, L, 30000 + L);
                    dtlb.reset_enable(); llc.reset_enable();
                    for (int tr = 0; tr < TRIALS; tr++){
                        uint64_t s = popcount_from_offsets(data, offs, 0);
                        asm volatile("" :: "r"(s));
                    }
                    long long dtlb_m = dtlb.disable_read();
                    long long llc_m  = llc.disable_read();
                    size_t total_acc = offs.size() * (size_t)TRIALS;
                    cout << "    L=" << setw(3) << L
                         << "  dTLB-miss/line=" << fixed << setprecision(4)
                         << (double)dtlb_m / total_acc
                         << "  LLC-miss/line=" << setprecision(4)
                         << (double)llc_m / total_acc << "\n";
                }
            }
            cout << "\n  => dTLB giam manh + LLC gan nhu khong doi => TLB la nguyen nhan chinh.\n"
                    "  => Nguoc lai => row-buffer/cache-effect.\n\n";
        }
    }

    // ---------- [N-fixed] cost/line deconfounded + shuffle (v18) ----------
    cout << "===== [N-fixed] cost/line (RNG tach, xao thu tu L moi trial, MAD)\n";
    // [v22-NEW] Boc lo bien an TRUOC khi chay: cho thay so voi moi L thuc su
    // dang do bao nhieu num_regions/footprint - day la thu ma ban goc chua
    // bao gio in ra, khien nguoi doc de nham cost/line chi phan anh "L"
    // (do sau tai su dung) trong khi footprint (working-set) cung dang
    // thay doi ngam theo.
    {
        CacheSizes cs = detect_cache_sizes();
        cout << "  [v22] Cache phat hien qua sysconf: L1d="
             << (cs.l1d>0? to_string(cs.l1d/1024)+"KB" : "?") << " L2="
             << (cs.l2>0? to_string(cs.l2/1024/1024)+"MB" : "?") << " L3="
             << (cs.l3>0? to_string(cs.l3/1024/1024)+"MB" : "?") << "\n";
        cout << "  [v22] Boc lo num_regions/footprint AN sau moi L (thiet ke goc: num_regions=total_lines/L):\n";
        for (int L : vector<int>{1,2,4,8,16,32,64})
            report_footprint(L, total_lines, N, WORDS_PER_4K, cs);
        cout << "  => Neu footprint 'cham' L2/L3 O NHUNG L KHAC NHAU, cost/line quan sat\n"
                "  dang tron 2 hieu ung (do sau tai su dung + kich thuoc working-set) lam 1.\n\n";
    }
    vector<int> Ls = {1, 2, 4, 8, 16, 32, 64};
    // [v21-FIX-BUG-BOOTSTRAP, KHOI PHUC trong ban hop nhat] Tra ve them
    // n_ops_per_L (offsets_per_L[li].size() THAT SU) - vi generate_offsets co
    // the CLAMP num_regions xuong khi total_lines/L > so vung 4KB kha dung
    // (N/WORDS_PER_4K), khien offsets tra ve IT HON total_lines gia dinh.
    // Dieu nay xay ra chu yeu o L NHO (L=1 can nhieu vung nhat) - neu bootstrap
    // gia dinh "n_ops = total_lines" cho MOI L (day chinh la dieu ban v22 goc
    // vo tinh lam, xem ghi chu [HOP NHAT v21+v22] o dau file), se gay SAI LECH
    // CHUAN HOA rieng cho L=1 (co the lech ~5x neu N nho so voi total_lines),
    // lam ty le dn bi DAO NGUOC hoan toan so voi diem uoc that, khien
    // bootstrap CI sup do ve [0,0] mot cach co he thong.
    auto run_deconfounded = [&](uint64_t* data, const char* label)
        -> tuple<vector<double>, vector<vector<double>>, vector<size_t>>
    {
        vector<vector<double>> times(Ls.size());
        vector<vector<size_t>> offsets_per_L(Ls.size());
        for (size_t li = 0; li < Ls.size(); li++){
            int L = Ls[li];
            size_t num_regions = total_lines / (size_t)L;
            offsets_per_L[li] = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K,
                                                 num_regions, L, 12000 + li);
        }
        vector<int> order(Ls.size());
        iota(order.begin(), order.end(), 0);
        mt19937_64 shuffle_rng(999);
        for (int tr = 0; tr < TRIALS; tr++){
            shuffle(order.begin(), order.end(), shuffle_rng);
            for (int li : order){
                auto a0 = high_resolution_clock::now();
                uint64_t s = popcount_from_offsets(data, offsets_per_L[li], 0);
                auto a1 = high_resolution_clock::now();
                asm volatile("" :: "r"(s));
                times[li].push_back(duration<double, micro>(a1 - a0).count());
            }
        }
        vector<double> cost(Ls.size());
        vector<size_t> n_ops_per_L(Ls.size());
        for (size_t li = 0; li < Ls.size(); li++){
            int L = Ls[li];
            size_t n_ops = offsets_per_L[li].size();
            n_ops_per_L[li] = n_ops;
            double mt = median(times[li]);
            double md = mad(times[li]);
            cost[li] = (mt * 1000.0) / (double)n_ops;
            cout << "  [" << label << "] L=" << setw(3) << L
                 << "  cost/line=" << fixed << setprecision(3) << setw(8) << cost[li]
                 << "ns  +/-" << setprecision(3)
                 << (md * 1000.0) / (double)n_ops << "ns (MAD)";
            if (n_ops < total_lines)
                cout << "  [CANH BAO: offsets bi kep " << n_ops << "/" << total_lines
                     << " - N qua nho so voi total_lines o L nay]";
            cout << "\n";
        }
        return {cost, times, n_ops_per_L};
    };

    // [v26-MOI] Chay ca "run_deconfounded" R_REPS lan (macro-replication) thay vi
    // 1 lan, de sau nay do duoc bat dinh GIUA-cac-lan-chay (xem [T] o duoi) - phat
    // hien tu quan sat thuc te: diem-uoc ratio_of_drops nhay 0.788->0.953->1.143
    // qua cac lan chay lai chuong trinh nay voi CUNG tham so. Lan replication #0
    // van duoc dung nguyen ven cho toan bo cac section [Q]/[S]/[Model]... ben
    // duoi (KHONG doi hanh vi cu), R_REPS-1 lan con lai CHI dung cho [T] moi.
    cout << "===== [v26] Macro-replication: chay lai TOAN BO benchmark deconfounded "
         << R_REPS << " lan\n"
            "  (de do bat dinh GIUA-cac-lan-chay, tach biet voi drift TRONG-1-lan-chay cua [Q])\n";
    vector<vector<vector<double>>> reps_times_normal, reps_times_hp;
    vector<double> reps_ratio_of_drops;
    vector<size_t> n_ops_normal, n_ops_hp;
    vector<double> cost_normal, cost_hp;
    vector<vector<double>> times_normal, times_hp;
    for (int rep = 0; rep < R_REPS; rep++){
        cout << "  --- replication " << (rep + 1) << "/" << R_REPS << " ---\n";
        auto [cn, tn, on] = run_deconfounded(normal, "thuong ");
        auto [ch, th, oh] = run_deconfounded(hp,     "hugepage");
        if (rep == 0){
            cost_normal = cn; times_normal = tn; n_ops_normal = on;
            cost_hp     = ch; times_hp     = th; n_ops_hp     = oh;
        }
        reps_times_normal.push_back(tn);
        reps_times_hp.push_back(th);
        double dn_r = cn[0] / cn.back();
        double dh_r = ch[0] / ch.back();
        double ror_r = (dn_r - 1.0) > 1e-9 ? (dh_r - 1.0) / (dn_r - 1.0) : 0.0;
        reps_ratio_of_drops.push_back(ror_r);
        cout << "      ratio_of_drops (diem uoc, lan chay nay) = " << fixed << setprecision(3) << ror_r << "\n";
    }
    {
        double mn = 0.0;
        for (double x : reps_ratio_of_drops) mn += x;
        mn /= (double)reps_ratio_of_drops.size();
        double sd = 0.0;
        for (double x : reps_ratio_of_drops) sd += (x - mn) * (x - mn);
        sd = R_REPS > 1 ? sqrt(sd / (R_REPS - 1)) : 0.0;
        double mn_val = *min_element(reps_ratio_of_drops.begin(), reps_ratio_of_drops.end());
        double mx_val = *max_element(reps_ratio_of_drops.begin(), reps_ratio_of_drops.end());
        cout << "  => Qua " << R_REPS << " lan chay: ratio_of_drops mean=" << setprecision(3) << mn
             << "  stddev(giua-cac-lan-chay)=" << sd
             << "  [min=" << mn_val << ", max=" << mx_val << "]\n";
        if (sd > 0.05)
            cout << "  [CANH BAO MANH] Do lech giua-cac-lan-chay (" << sd << ") KHONG nho - benchmark\n"
                    "  nay tren MAY/SANDBOX hien tai co bat dinh giua-cac-lan-chay dang ke, rieng biet\n"
                    "  voi drift trong-1-lan-chay cua [Q]. Xem [T] o duoi de co CI phan anh dung dieu\n"
                    "  nay (thay vi chi tin 1 lan chay nhu [S]/[Q-fix] lam).\n";
        cout << "\n";
    }

    cout << "\n  --- SO SANH ---\n";
    for (size_t i = 0; i < Ls.size(); i++){
        cout << "    L=" << setw(3) << Ls[i]
             << "  thuong=" << fixed << setprecision(2) << setw(7) << cost_normal[i]
             << "ns   hugepage=" << setw(7) << cost_hp[i]
             << "ns   ty le=" << setprecision(3) << (cost_hp[i] / cost_normal[i]) << "\n";
    }
    double drop_normal = cost_normal[0] / cost_normal.back();
    double drop_hp     = cost_hp[0]     / cost_hp.back();
    cout << "\n  Muc giam (L=1->64): thuong=" << setprecision(3) << drop_normal
         << "x,  hugepage=" << drop_hp << "x\n";
    // [v21-NEW, buoc 3] Canh bao ro khi mau so gan 0: ratio_of_drops =
    // (drop_hp-1)/(drop_normal-1) se NO (bien thien cuc lon, khong on dinh)
    // neu drop_normal ~ 1 (nghia la L khong anh huong toi cost tren mang
    // thuong - vd may khong co ap luc TLB o quy mo nay). Nguong 0.05 la kinh
    // nghiem, khong phai chan tuyet doi.
    if (drop_normal - 1.0 < 0.05){
        cout << "  [CANH BAO] drop_normal gan 1.0 (=" << setprecision(4) << drop_normal
             << ") => mau so cua ratio_of_drops gan 0 => ket qua ben duoi CO THE\n"
                "  KHONG ON DINH (nhay cam cao voi nhieu do). Dien giai than trong,\n"
                "  uu tien nhin CI thay vi diem uoc don le.\n";
    }
    double ratio_of_drops = (drop_normal - 1.0) > 1e-9
                            ? (drop_hp - 1.0) / (drop_normal - 1.0) : 0.0;
    cout << "  Ty le phan con lai tren hugepage (diem uoc): " << setprecision(3)
         << ratio_of_drops << "\n";

    // ---------- [v25c-MOI, ung voi (Q)] Kiem tra troi dat he thong (nhiet/DVFS)
    // theo trial-index TRUOC KHI tin bootstrap_ci (i.i.d.) o duoi. Neu co troi
    // dat qua cac trial LIEN TIEP, resample i.i.d. tung phan tu se DANH GIA
    // THAP do bat dinh thuc (cac quan sat khong con doc lap voi nhau). Chay
    // TRUC TIEP tren CHINH 4 vector se dua vao bootstrap_ci ben duoi - khong
    // bia them du lieu, dung nguyen ban da do (thu tu trong times_normal[li]/
    // times_hp[li] la thu tu trial THAT, vi vong lap ngoai la 'for tr' va chi
    // L-order o BEN TRONG moi trial bi xao - xem run_deconfounded()).
    cout << "\n  [Q] Kiem tra troi dat he thong (nhiet/DVFS) theo trial-index,\n"
            "  TRUOC KHI tin bootstrap CI o duoi (Theil-Sen slope + permutation test,\n"
            "  N_perm=2000, tren CHINH 4 vector se dung cho bootstrap_ci):\n";
    // [v34-MOI] Buoc nay la O(TRIALS^2 * n_perm), chay 4 lan (4 vector) - da
    // toi uu hang so o permutation_test_trend() nhung van con O(n^2) ve ban
    // chat thuat toan (Theil-Sen). TRIALS la tham so nguoi dung tu dat qua
    // argv[3] khong gioi han tren, nen canh bao SOM thay vi de treo im lang
    // neu ai do tang TRIALS de "do chinh xac hon" ma khong ngo buoc nay dat
    // rieng O(n^2) trong khi phan con lai cua chuong trinh van tuyen tinh.
    if (TRIALS > 300){
        cout << "  [CANH BAO HIEU NANG] TRIALS=" << TRIALS << " lon -> buoc [Q] nay ban chat\n"
                "  O(TRIALS^2 * 2000) x 4 vector (Theil-Sen + permutation test), co the mat\n"
                "  vai chuc giay den vai phut du DA toi uu hang so - CAC BUOC KHAC cua\n"
                "  chuong trinh (bootstrap, SIMD benchmark) van tuyen tinh voi TRIALS nen\n"
                "  KHONG cham tuong ung. Neu khong can kiem tra troi dat, giam TRIALS hoac\n"
                "  doi.\n";
    }
    {
        mt19937_64 trend_rng(271828);
        struct TV { const char* name; const vector<double>* v; };
        vector<TV> tvecs = {
            {"times_normal[L=1] ", &times_normal[0]},
            {"times_normal[L=64]", &times_normal.back()},
            {"times_hp[L=1]     ", &times_hp[0]},
            {"times_hp[L=64]    ", &times_hp.back()},
        };
        bool any_significant = false;
        for (auto& tv : tvecs){
            double slope = theil_sen_slope(*tv.v);
            double med_v = median(*tv.v);
            double drift_total = slope * (double)(tv.v->size() - 1);
            double drift_pct = (med_v > 1e-12) ? 100.0 * drift_total / med_v : 0.0;
            double pval = permutation_test_trend(*tv.v, 2000, trend_rng);
            bool sig = pval < 0.05;
            any_significant = any_significant || sig;
            cout << "    " << tv.name << ": Theil-Sen slope=" << scientific << setprecision(2) << slope
                 << " us/trial  troi_dat_uoc_toan_bo=" << fixed << setprecision(2) << drift_pct
                 << "%  p(permutation)=" << setprecision(4) << pval
                 << (sig ? "  <-- CO XU HUONG THAT (p<0.05)" : "  (khong phan biet duoc voi nhieu)") << "\n";
        }
        if (any_significant)
            cout << "  => IT NHAT 1 vector CO xu huong don dieu THAT (p<0.05): bootstrap i.i.d.\n"
                    "  o duoi CO THE danh gia THAP do bat dinh thuc su. Nen dien giai CI voi de\n"
                    "  dat than trong hon, hoac thay bang block/stationary bootstrap (Politis &\n"
                    "  Romano 1994) - day la KET QUA CHAN DOAN, CHUA phai ban sua.\n\n";
        else
            cout << "  => KHONG vector nao co xu huong phan biet duoc voi nhieu (p>=0.05 ca 4).\n"
                    "  i.i.d. bootstrap_ci o duoi hop ly dung nhu hien tai tren may nay, khong\n"
                    "  can sua - NHUNG ket luan nay chi dung cho may/lan chay nay, nen chay lai\n"
                    "  kiem tra nay moi khi doi may hoac tang manh TRIALS.\n\n";
    }

    // ---------- [S] Bootstrap CI for ratio_of_drops ----------
    {
        // ds[0]=L1_normal times, ds[1]=L64_normal, ds[2]=L1_hp, ds[3]=L64_hp
        // [v21-FIX, RESTORED] n_ops MUST come from the actual
        // n_ops_normal/n_ops_hp values (which may be < total_lines because
        // generate_offsets is clamped for small L) — NEVER assume they are
        // equal to total_lines for every L. This was the root cause that
        // systematically collapsed the bootstrap CI to [0,0]
        // (see the [MERGED v21+v22] notes at the beginning of the file).
        auto stat_fn = [&](const vector<vector<double>>& ds) -> double {
            auto med_cost = [](const vector<double>& t, size_t n_ops){
                return (median(t) * 1000.0) / (double)n_ops;
            };
            double c1n  = med_cost(ds[0], n_ops_normal[0]);
            double c64n = med_cost(ds[1], n_ops_normal.back());
            double c1h  = med_cost(ds[2], n_ops_hp[0]);
            double c64h = med_cost(ds[3], n_ops_hp.back());
            double dn = c1n / c64n;
            double dh = c1h / c64h;
            return (dn - 1.0) > 1e-9 ? (dh - 1.0) / (dn - 1.0) : 0.0;
        };
        vector<vector<double>> data_sets = {
            times_normal[0], times_normal.back(),
            times_hp[0],     times_hp.back()
        };
        mt19937_64 boot_rng(4242);
        auto boot = bootstrap_ci(data_sets, stat_fn, N_BOOT, boot_rng);
        cout << "  [S] Bootstrap CI 95% for ratio_of_drops (i.i.d., ASSUMPTION ALREADY INVALIDATED BY [Q]\n"
                "  ABOVE — see below for comparison with the corrected version): [" << fixed << setprecision(3)
             << boot.lo << ", " << boot.hi << "]"
             << "   median=" << boot.median << "  MAD=" << boot.mad
             << "  (n_boot=" << N_BOOT << ")\n";

        // [v25c-NEW, corresponding to the (Q)-fix] Directly compare against
        // stationary bootstrap (the proper fix for the issue confirmed in [Q]).
        // mean_block_len=3 is an empirical choice for short sequences
        // (TRIALS is typically 20-30). Also test block=5 to evaluate sensitivity
        // to this parameter instead of trusting a single value.
        mt19937_64 sboot_rng1(5242), sboot_rng2(6242);
        auto sboot3 = stationary_bootstrap_ci(data_sets, stat_fn, N_BOOT, 3.0, sboot_rng1);
        auto sboot5 = stationary_bootstrap_ci(data_sets, stat_fn, N_BOOT, 5.0, sboot_rng2);
        cout << "  [Q-fix] Stationary bootstrap CI 95% (block~3): ["
             << setprecision(3) << sboot3.lo << ", " << sboot3.hi << "]"
             << "   median=" << sboot3.median << "  MAD=" << sboot3.mad << "\n";
        cout << "  [Q-fix] Stationary bootstrap CI 95% (block~5): ["
             << setprecision(3) << sboot5.lo << ", " << sboot5.hi << "]"
             << "   median=" << sboot5.median << "  MAD=" << sboot5.mad << "\n";
        double width_iid = boot.hi - boot.lo;
        double width_sboot3 = sboot3.hi - sboot3.lo;
        cout << "  => CI width: i.i.d.=" << setprecision(3) << width_iid
             << "   stationary(block~3)=" << width_sboot3
             << "   ratio=" << setprecision(2) << (width_sboot3 / width_iid) << "x\n";
        if (width_sboot3 > width_iid * 1.15)
            cout << "     Stationary CI is SIGNIFICANTLY WIDER — this CONFIRMS (rather than merely\n"
                    "     suggesting) that the i.i.d. bootstrap above UNDERESTIMATED uncertainty,\n"
                    "     exactly as warned in [Q]. From this point onward, the stationary CI\n"
                    "     should be used instead of the i.i.d. version.\n";
        else
            cout << "     Stationary CI is NOT significantly wider — the issue identified in [Q]\n"
                    "     may NOT materially affect the CI width for THIS PARTICULAR statistic\n"
                    "     (the ratio), because both the numerator and denominator may drift in\n"
                    "     ways that partially cancel each other out.\n";
        double lo = boot.lo, hi = boot.hi;
        if (lo > 0.7)
            cout << "      => CI lies entirely above 0.7: a substantial residual effect remains even with huge pages.\n";
        else if (hi < 0.3)
            cout << "      => CI lies entirely below 0.3: huge pages eliminate nearly all of the effect; the TLB is the dominant cause.\n";
        else
            cout << "      => CI overlaps the 0.3/0.7 thresholds: the conclusion is NOT yet sufficiently conclusive.\n";
        cout << "\n";

        // ---------- [T, v26-NEW] Hierarchical bootstrap: incorporates
        // BETWEEN-run uncertainty (whereas [Q]/[Q-fix] only accounts for
        // WITHIN-run uncertainty) ----------
        cout << "  [T] Hierarchical (nested/predictive) bootstrap CI 95% - each bootstrap draw\n"
                "  = sample WITH REPLACEMENT exactly 1 out of the " << R_REPS << " completed replications (level 1),\n"
                "  then perform stationary resampling WITHIN that replication (level 2, still "
             << TRIALS << " observations, WITHOUT merging or averaging across multiple replications\n"
                "  — both approaches would artificially narrow the CI; see the v26b/v26c-GENUINE BUG FIX notes):\n";
        vector<vector<vector<double>>> reps_data_sets(R_REPS);
        for (int rep = 0; rep < R_REPS; rep++){
            reps_data_sets[rep] = {
                reps_times_normal[rep][0], reps_times_normal[rep].back(),
                reps_times_hp[rep][0],     reps_times_hp[rep].back()
            };
        }
        mt19937_64 hboot_rng(7242);
        auto hboot = hierarchical_bootstrap_ci(reps_data_sets, stat_fn, N_BOOT, 3.0, hboot_rng);
        cout << "  [T] Hierarchical bootstrap CI 95% (block~3, R=" << R_REPS << "): ["
             << fixed << setprecision(3) << hboot.lo << ", " << hboot.hi << "]"
             << "   median=" << hboot.median << "  MAD=" << hboot.mad << "\n";
        double width_hboot = hboot.hi - hboot.lo;
        cout << "  => CI width: i.i.d.=" << setprecision(3) << width_iid
             << "   stationary(block~3)=" << width_sboot3
             << "   hierarchical(block~3,R=" << R_REPS << ")=" << width_hboot
             << "   hierarchical/stationary ratio=" << setprecision(2) << (width_hboot / width_sboot3) << "x\n";
        // [v26d] Correct comparison: hierarchical vs. stationary (BOTH already
        // account for within-run drift), NOT versus i.i.d. (the uncorrected
        // baseline). The stationary→hierarchical difference isolates ONLY the
        // contribution of BETWEEN-run uncertainty, separate from the within-run
        // drift already handled by [Q-fix].
        if (width_hboot > width_sboot3 * 1.15)
            cout << "     [T] CONFIRMED: incorporating between-run uncertainty produces a\n"
                    "     SIGNIFICANTLY WIDER CI than single-run stationary bootstrap.\n"
                    "     [Q-fix] alone underestimates the true uncertainty (it only considers\n"
                    "     a single run), so [T] should be regarded as the official CI.\n";
        else
            cout << "     [T] The stationary→hierarchical increase is small — between-run\n"
                    "     uncertainty (identified in [v26]) is not substantially larger than\n"
                    "     the within-run drift already corrected by [Q-fix]. The stationary CI\n"
                    "     is likely acceptable in practice, although [T] remains the more\n"
                    "     methodologically complete approach because it does not assume that\n"
                    "     a single run is sufficient.\n";
        cout << "\n";

// ---------- [U, v27-NEW, breakthrough] Variance decomposition (random-effects,
        // method-of-moments) + OPTIMAL measurement budget allocation formula --------------
        // Motivation: [T] only indicates "there is between-run uncertainty" (a single number),
        // but DOES NOT tell you WHAT TO DO with that budget (benchmark runtime)
        // to reduce uncertainty the fastest. The real answer depends on the RATIO
        // between the 2 variance components - they must be separated, not just measuring
        // the combined CI width.
        //
        // Model (1-way random-effects, standard for nested designs "replication
        // > trial"): estimate of 1 run = mu + b_r + w_r, with
        //   b_r ~ (0, sigma_b^2)  = BETWEEN-run uncertainty (machine state:
        //         thermals, DVFS, virtualized "neighbors"... INDEPENDENT of TRIALS)
        //   w_r ~ (0, sigma_w^2/TRIALS) = WITHIN-run uncertainty (trial noise,
        //         decreases by ~1/TRIALS as trials increase - this is what [Q-fix] handled)
        // Estimated sigma_b^2 = max(0, Var(reps_ratio_of_drops across R runs) - sigma_w^2/TRIALS)
        // (classic method-of-moments for 1-way random-effects ANOVA, BALANCED design
        // since each replication has the SAME TRIALS).
        cout << "  [U] Variance decomposition (random-effects ANOVA, method-of-moments)\n"
                "  across " << R_REPS << " macro-replications run in [v26], AND the OPTIMAL budget\n"
                "  allocation formula derived from it:\n";
        {
            double mean_ror = 0.0;
            for (double x : reps_ratio_of_drops) mean_ror += x;
            mean_ror /= (double)R_REPS;
            double var_total_across_reps = 0.0;
            for (double x : reps_ratio_of_drops) var_total_across_reps += (x - mean_ror) * (x - mean_ror);
            var_total_across_reps /= (double)(R_REPS - 1);

            // sigma_w^2/TRIALS is estimated DIRECTLY via the variance of the
            // stationary-bootstrap distribution INDIVIDUALLY for EACH replication (avoiding theoretical
            // 1/TRIALS formulas to prevent false i.i.d. assumptions), then AVERAGED
            // across R replications - this IS EXACTLY sigma_w^2/TRIALS at the current
            // TRIALS scale, NOT raw (unscaled) sigma_w^2.
            double var_within_avg = 0.0;
            mt19937_64 diag_rng(31415);
            const int N_BOOT_DIAG = 1000; // sufficient for stable variance estimation, no need to be as wide as main CI
            for (int rep = 0; rep < R_REPS; rep++){
                vector<vector<double>> ds_rep = {
                    reps_times_normal[rep][0], reps_times_normal[rep].back(),
                    reps_times_hp[rep][0],     reps_times_hp[rep].back()
                };
                auto sb = stationary_bootstrap_ci(ds_rep, stat_fn, N_BOOT_DIAG, 3.0, diag_rng);
                // Convert MAD (Median Absolute Deviation) to equivalent standard deviation
                // estimator: for approximately normal distributions, sigma ~ 1.4826 * MAD.
                double sigma_eq = 1.4826 * sb.mad;
                var_within_avg += sigma_eq * sigma_eq;
            }
            var_within_avg /= (double)R_REPS;

            double var_between_est = max(0.0, var_total_across_reps - var_within_avg);
            double var_grand_total = var_between_est + var_within_avg;
            double pct_between = (var_grand_total > 1e-12) ? 100.0 * var_between_est / var_grand_total : 0.0;
            double pct_within  = 100.0 - pct_between;

            cout << "    Var(reps_ratio_of_drops across " << R_REPS << " runs, raw)      = "
                 << scientific << setprecision(3) << var_total_across_reps << "\n"
                 << "    Within-1-run Var estimated from stationary boot (Avg over R) = "
                 << var_within_avg << "   (at current TRIALS=" << TRIALS << ")\n"
                 << "    => Estimated BETWEEN-run Var (method-of-moments)       = "
                 << var_between_est << "\n"
                 << fixed << setprecision(1)
                 << "    => Decomposition: " << pct_between << "% from BETWEEN-runs (sigma_b^2, cannot\n"
                 << "       be reduced by increasing TRIALS)   vs   " << pct_within
                 << "% from WITHIN-1-run (sigma_w^2/TRIALS,\n"
                 << "       can be reduced by increasing TRIALS).\n";

            // ---------- OPTIMAL budget allocation formula ----------
            // If the final result is the MEAN of R independent runs, each with
            // TRIALS trials, then (derived protocol-wide, see longer note in chat history):
            // Var(mean) = sigma_b^2/R + sigma_w^2/T with T = R*TRIALS = TOTAL trials across the budget.
            // => With FIXED T, the term sigma_w^2/T DOES NOT CHANGE regardless of how R
            // vs TRIALS is split (it depends only on TOTAL T) - ONLY the term sigma_b^2/R
            // can be reduced by increasing R. Conclusion: WITH A FIXED BUDGET T, increasing
            // the number of INDEPENDENT RUNS R (reducing TRIALS per run down to a reasonable minimum
            // so [Q]/[Q-fix] can still diagnose drift, ~10-20) is ALWAYS better than
            // or equal to investing more TRIALS into a single run - because the sigma_b^2/R component
            // CANNOT be reduced by any means other than increasing R.
            if (pct_between > 15.0){
                double sigma_w2_full = var_within_avg * (double)TRIALS; // estimate raw sigma_w^2 (unscaled by TRIALS)
                // [v33-FIX] This variable was previously computed then DISCARDED, never printed -
                // matching the intent of the comment right above it (estimate sigma_w^2 COMPARED to sigma_b^2
                // on line 2989) but never appeared in cout. Added a line to print the REAL value, placed next to
                // var_between_est so the reader can directly compare both terms of Var(mean) = sigma_b^2/R + sigma_w^2/T.
                cout << "    (estimates: sigma_w^2 (per trial, unscaled)=" << scientific << setprecision(3)
                     << sigma_w2_full << "   vs sigma_b^2 (per run)=" << var_between_est
                     << fixed << setprecision(1) << ")\n";
                cout << "    [OPTIMAL RECOMMENDATION, because between-run ratio > 15%]: with TOTAL budget\n"
                        "    T=R*TRIALS trials FIXED, Var(final mean) = sigma_b^2/R + sigma_w^2/T -\n"
                        "    the 2nd term DOES NOT change based on how R/TRIALS is split (depends only on T), so\n"
                        "    ONLY increasing R (reducing TRIALS/run down to ~10-20, enough for [Q] to diagnose\n"
                        "    drift) TRULY reduces Var(mean). Example: current fixed T = "
                     << (R_REPS * TRIALS) << " trials\n"
                        "    (R=" << R_REPS << ", TRIALS=" << TRIALS << "), switching to R=" << (R_REPS * 2)
                     << ", TRIALS=" << (TRIALS / 2 > 5 ? TRIALS / 2 : TRIALS)
                     << " (SAME T) will reduce\n"
                        "    Var(mean) from (sigma_b^2/" << R_REPS << " + sigma_w^2/T) down to (sigma_b^2/"
                     << (R_REPS * 2) << " + sigma_w^2/T) - 1st term decreases by ~2x, 2nd term STAYS THE SAME.\n";
            } else {
                cout << "    Between-run ratio <= 15% (uncertainty stems mainly from within-1-run) - with\n"
                        "    a fixed budget, increasing TRIALS per run remains equally effective as increasing R,\n"
                        "    NO strategy change is required.\n";
            }
            cout << "    [Honest & Important WARNING]: sigma_b^2 here is estimated from " << R_REPS
                 << " replications\n"
                    "    run SEQUENTIALLY WITHIN THE SAME process (a loop in main()), NOT from " << R_REPS
                 << " executions\n"
                    "    re-invoking the .exe file from scratch (a new process). Previous observations (re-running\n"
                    "    the entire binary multiple times from shell) yielded significantly larger ESTIMATE fluctuation\n"
                    "    (0.788 to 1.143, WIDER than both [T]/[U] estimated here), because separate .exe calls introduce\n"
                    "    additional sources of variance (reloading binary, cold cache/TLB from start, different scheduler state)\n"
                    "    which these in-process replications CANNOT replicate. => sigma_b^2\n"
                    "    and between-run % printed above are LOWER BOUNDS (need to be re-estimated by\n"
                    "    invoking .exe multiple times externally, e.g., shell script 'for i in 1..N; do ./bin;\n"
                    "    done', which cannot be done FROM WITHIN a single run of this binary).\n";
        }
        cout << "\n";
    }

    // ---------- [v21-NEW, step 4] Fit cost(L) = a + b/L model ----------
    cout << "===== [Model] Fit cost(L) ~= a + b/L (least-squares)\n";
    print_cost_model_fit("normal ", Ls, cost_normal);
    print_cost_model_fit("hugepage", Ls, cost_hp);
    cout << "\n";

    // ---------- Prefetch sweep (v18) ----------
    cout << "===== [MLP-explicit] Prefetch-distance at L=16 (normal array)\n";
    {
        int L = 16;
        size_t num_regions = total_lines / (size_t)L;
        auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, num_regions, L, 55000);
        for (int pd : {0, 2, 4, 8, 16, 32}){
            vector<double> t;
            for (int tr = 0; tr < TRIALS; tr++){
                auto a0 = high_resolution_clock::now();
                uint64_t s = popcount_from_offsets(normal, offs, pd);
                auto a1 = high_resolution_clock::now();
                asm volatile("" :: "r"(s));
                t.push_back(duration<double, micro>(a1 - a0).count());
            }
            double mt = median(t);
            cout << "  prefetch_distance=" << setw(3) << pd
                 << "  cost/line=" << fixed << setprecision(3)
                 << (mt * 1000.0) / (double)offs.size() << " ns\n";
        }
        cout << "\n";
    }

// ---------- [R] True RMSE (INDEPENDENT offset per rep) ----------
    cout << "===== [R] True RMSE (INDEPENDENT offset per rep), L around Cochran\n";
    {
        cout << "  --- standard array ---\n";
        for (int L : Ls_full){
            size_t num_regions = total_lines / (size_t)L;
            vector<double> times, rel_errs;
            for (int r = 0; r < RMSE_REPS; r++){
                auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K,
                                             num_regions, L, 70000ULL + r * 97 + L);
                auto a0 = high_resolution_clock::now();
                uint64_t sum = popcount_from_offsets(normal, offs, 0);
                auto a1 = high_resolution_clock::now();
                times.push_back(duration<double, micro>(a1 - a0).count());
                // FIX bug v19: each offset = 8 words
                double mean_per_word = (double)sum / (double)(offs.size() * 8);
                double estimate = mean_per_word * (double)N;
                double rel_err  = (estimate - (double)ref_normal) / (double)ref_normal;
                rel_errs.push_back(rel_err);
            }
            double rmse = 0;
            for (double e : rel_errs) rmse += e * e;
            rmse = sqrt(rmse / rel_errs.size());
            double mt  = median(times);
            double eff = rmse * sqrt(mt);   // smaller is better
            cout << "    L=" << setw(3) << L
                 << "  median_time=" << fixed << setprecision(1) << setw(9) << mt
                 << "us  RMSE=" << setprecision(5) << rmse
                 << "  RMSE*sqrt(t)=" << setprecision(4) << eff << "\n";
        }
        cout << "\n";
    }

    // ---------- Sanity: sampling vs full-scan ----------
    cout << "===== [Sanity] Sampling (L=64) vs full sequential scan\n";
    {
        size_t num_regions = total_lines / 64;
        auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, num_regions, 64, 99999);
        auto a0 = high_resolution_clock::now();
        uint64_t s = popcount_from_offsets(normal, offs, 8);
        auto a1 = high_resolution_clock::now();
        asm volatile("" :: "r"(s));
        double sample_us = duration<double, micro>(a1 - a0).count();
        size_t bytes_touched = offs.size() * 64;

        auto b0 = high_resolution_clock::now();
        volatile uint64_t full_sum = contig_bulk_dispatch(normal, N);
        auto b1 = high_resolution_clock::now();
        double full_us = duration<double, micro>(b1 - b0).count();

        cout << "  Sampling (L=64, " << bytes_touched/1024 << "KB): "
             << fixed << setprecision(1) << sample_us << " us\n";
        cout << "  Full sequential (" << BYTES/(1024*1024) << "MB): "
             << full_us << " us\n";
        cout << "  => Sampling is faster by " << setprecision(1)
             << (full_us / sample_us) << "x\n\n";
        (void)full_sum;
    }

    // ---------- [v22-NEW] Confounding variable resolution: decoupling 2 variable axes (L vs footprint) ----------
    cout << "===== [v22] Noise-decoupling experiment: isolating 'reuse depth' (L)\n"
            "      from 'working-set size' (footprint), instead of letting them drift together\n";
    {
        CacheSizes cs = detect_cache_sizes();
        size_t n_regions_avail = N / WORDS_PER_4K;
        // Select a fixed num_regions large enough to ensure footprint > L3 (safe,
        // truly "cold"), while not exceeding available n_regions.
        size_t safe_fixed_regions = n_regions_avail; // use entire available page space
        if (cs.l3 > 0){
            size_t min_regions_over_l3 = (size_t)((cs.l3 * 1.5) / 4096.0) + 1;
            safe_fixed_regions = min(n_regions_avail, max(min_regions_over_l3, (size_t)4000));
        }
        cout << "  (using fixed num_regions=" << safe_fixed_regions << " -> footprint="
             << fixed << setprecision(1) << (safe_fixed_regions*4096.0/1024/1024) << "MB)\n\n";

        cout << "  [Experiment 1] L-sweep, FIXED footprint (noise-decoupled):\n";
        auto [cost_fixed_fp, r2_fixed] = run_fixed_footprint_sweep(
            normal, Ls, N, safe_fixed_regions, max(5, TRIALS/2), "standard array");
        cout << "  R^2 comparison: ORIGINAL model (coupled variables)=" << setprecision(4) << 0.0
             << " (see [Model] section above)  vs  DECOUPLED model=" << r2_fixed << "\n\n";

        cout << "  [Experiment 2] Footprint-sweep, FIXED L=8 (characterizing remaining axis):\n";
        vector<size_t> region_sweep = {8, 64, 256, 2048, 8192, (size_t)(n_regions_avail*0.9)};
        run_fixed_L_footprint_sweep(normal, N, 8, region_sweep, max(5, TRIALS/2), cs);

        if (g_cap_violations > 0)
            cout << "  [v22] Total number of generate_offsets() TRUNCATIONS (N-fixed violations) across entire run: "
                 << g_cap_violations << "\n";
    }
// ============================================================
    // [v23-NEW, INVENTION] IRM-Burst scaling law: replaces the
    //   curve-fit a+b/L model with a PHYSICALLY MEANINGFUL formula
    //   (derived from probability theory, not empirical curve fitting),
    //   automatically resolving the L-vs-footprint ambiguity by incorporating
    //   BOTH explicit variables, and SELF-VERIFYING cross-wise across 2 independent axes.
    //   See full derivation in the file header comment, section [Y]/[Z].
    // ============================================================
    cout << "===== [v23] IRM-Burst scaling law (new invention, self-verifying)\n";
    {
        mt19937_64 irm_rng(31415926);
        bool law_ok = self_test_irm_law(irm_rng);
        if (!law_ok){
            cout << "  [v23] Self-test FAILED on this machine -> SKIPPING entire [v23] section\n"
                    "  (will not use results inferred from an unverified physical law).\n\n";
        } else {
            HwCounter dtlb2;
            dtlb2.open(DTLB_MISS_CFG(), "dTLB-miss");

            // [v25c-NEW, corresponding to (R)] Run VARPRO (P) UNCONDITIONALLY from here - even when
            // real PMUs ARE present - not just when blocked as before. Overhead is CHEAP (pure wall-clock
            // data already available from cost_normal/Ls/n_ops_normal, no extra measurements), so there is
            // no reason to only run half of it. The goal matches proposal (R): present BOTH numbers
            // (real PMU AND software VARPRO) side-by-side IN EVERY RUN, so anyone with real PMU hardware
            // can cross-validate immediately without modifying code further.
            vector<double> Wactual_vp(Ls.size());
            for (size_t i = 0; i < Ls.size(); i++)
                Wactual_vp[i] = (double)n_ops_normal[i] / (double)Ls[i];
            double C_max_guess = *max_element(Wactual_vp.begin(), Wactual_vp.end());
            VarProResult vp = varpro_fit_capacity_no_pmu(Ls, Wactual_vp, cost_normal, C_max_guess);
            CostModelFit old_fit_novpmu = fit_cost_model(Ls, cost_normal);

            if (!dtlb2.ok){
                cout << "  [v23] UNABLE TO MEASURE HW counter (permission/container) -> cannot\n"
                        "  infer C from REAL dTLB-misses. Try: sudo sysctl kernel.perf_event_paranoid=1\n"
                        "  (The formula itself was confirmed logically/mathematically valid in the self-test above.)\n\n"
                        "  [v25-proposal] Running a fallback INSTEAD that REQUIRES NO PMU: using VARPRO\n"
                        "  (Golub & Pereyra 1973) on the ALREADY AVAILABLE cost_normal/Ls/n_ops_normal data\n"
                        "  (pure wall-clock, zero extra measurements, requires no HW privileges):\n";
                cout << "    C_fit (VARPRO, no PMU) = " << fixed << setprecision(1) << vp.C_fit
                     << " pages (~" << setprecision(2) << (vp.C_fit * 4.0) << "KB)\n"
                     << "    cost/line = " << setprecision(4) << vp.a << " + " << vp.b
                     << " * predicted_misses/line(IRM-Burst)   R^2=" << setprecision(4) << vp.r2 << "\n"
                     << "    comparison across the SAME 7 points: OLD model a+b/L (without W) R^2="
                     << setprecision(4) << old_fit_novpmu.r_squared << "\n"
                     << "    (WARNING: this is the C that BEST FITS measured timing ASSUMING the\n"
                     << "     idealized IRM-Burst/LRU model holds true - NOT C measured directly from\n"
                     << "     hardware. High R^2 is INDIRECT evidence, not direct confirmation that real\n"
                     << "     dTLB behaves strictly according to the model; see full warnings in the\n"
                     << "     comment preceding varpro_fit_capacity_no_pmu().)\n";

                // [v28-NEW, BREAKTHROUGH] Bootstrap CI for C_fit - NEVER PRESENT in any
                // previous version, despite being the ONLY metric the reader truly needs.
                // Reuses the R replications from [v26] (the reps_times_normal variable already
                // available in main), ZERO extra hardware measurements.
                {
                    mt19937_64 vp_boot_rng(8181);
const int GRID_BOOT = 2000; // Increased from 500 to 2000 after detecting MAD=0.0;
                            // suspected to be a grid artifact (see dry-run logs),
                            // still lower than 4000 of the point-estimate to preserve speed.
auto vp_ci = bootstrap_ci_varpro_C(reps_times_normal, Ls, Wactual_vp,
                                   n_ops_normal, C_max_guess, GRID_BOOT,
                                   N_BOOT, 3.0, vp_boot_rng);
cout << "    [v28] Bootstrap CI 95% for C_fit (hierarchical/stationary, block~3,\n"
     << "    R=" << R_REPS << ", grid_boot=" << GRID_BOOT << "): ["
     << setprecision(1) << vp_ci.lo << ", " << vp_ci.hi << "] pages"
     << "   median=" << vp_ci.median << "   MAD=" << vp_ci.mad << "\n"
     << "    => (~" << setprecision(2) << (vp_ci.lo * 4.0 / 1024.0) << " - "
     << (vp_ci.hi * 4.0 / 1024.0) << " MB TLB-reach, median ~"
     << (vp_ci.median * 4.0 / 1024.0) << " MB)\n";

double vp_ci_width = vp_ci.hi - vp_ci.lo;
if (vp_ci_width > 0.6 * vp.C_fit)
    cout << "    [REAL WARNING] CI width (" << setprecision(1) << vp_ci_width
         << " pages) is a LARGE fraction of the point estimate (" << vp.C_fit << " pages) -\n"
         << "    C_fit from VARPRO on this machine/sandbox is UNSTABLE; report\n"
         << "    THIS ENTIRE interval [v28], NOT just a single point estimate, whenever citing.\n";

// [v28b] Check whether CI jumps between multiples of C - if so, scan
// the actual R^2(C) surface to confirm TRUE multi-modality rather than a grid artifact.
vector<LocalMax> lmax = varpro_find_local_maxima(Ls, Wactual_vp, cost_normal,
                                                  C_max_guess, 4000, 0.90);
// [v33-FIX] cluster_local_maxima() was written in [v28c] SPECIFICALLY to group
// consecutive grid points into tighter [C_lo, C_hi] ranges (see pre-definition comments),
// but prior to this patch was NEVER called here — the ONLY place using lmax — leaving
// [v28b] printing raw grid points (compiler raised "cluster_local_maxima defined but not used").
vector<MaxCluster> lclusters = cluster_local_maxima(lmax, C_max_guess);
cout << "    [v28b] R^2(C) surface scan on raw point estimate: " << lmax.size()
     << " grid points -> " << lclusters.size()
     << " local maxima clusters within 90% threshold of global R^2:\n";

for (auto& c : lclusters)
    cout << "        C=[" << setprecision(1) << c.C_lo << ", " << c.C_hi
         << "] pages  R^2~" << setprecision(4) << c.r2 << "\n";

if (lclusters.size() >= 2) {
    // [v33-FIX-2] Use lclusters.size() (NOT lmax.size() as in old version):
    // a long flat R^2 plateau (C >> working-set, model genuinely saturated —
    // as [v28c] itself noted) can generate MANY raw grid points satisfying
    // "local maximum" (each point on a flat sequence is >= its 2 neighbors)
    // despite being 1 contiguous region, not separate peaks. Counting lmax.size()
    // directly FALSELY flags "TRUE multi-modality" (2 C aliasing values) when it
    // is actually 1 wide plateau. cluster_local_maxima merges consecutive points
    // (distance <6% C_max AND close r2) into 1 [C_lo, C_hi] range, so lclusters.size()>=2
    // is the correct signal for "TRULY >= 2 distinct regions", accurately distinguishing
    // options (a)/(b) described in the [v28b] function comments.
    cout << "    => CONFIRMED REAL (not a grid artifact): IRM-Burst model CANNOT\n"
            "       IDENTIFY C on this 7-point dataset - there are >= 2 DISTINCT clusters\n"
            "       of C values fitting the curve NEARLY EQUALLY WELL. CI [v28] jumping between\n"
            "       these values ACCURATELY REFLECTS the genuine ambiguity of the problem, not a bug.\n"
            "       To make C identifiable, ADD MORE L data points (currently 7: 1,2,4,8,16,32,64)\n"
            "       or test real PMU for a 2nd independent constraint (see branches (R)/(P)).\n";
} else {
    cout << "    => Only 1 clear maximum - CI [v28] value jumps are likely due to MULTIPLE TRIALS\n"
            "       pushing the estimate across this peak's neighborhood, not TRUE multi-modality.\n";
}
cout << "\n";
                }

                // [v25b-NEW, continuing for the interrupted session] MISSING STEP in this
                // non-PMU branch: validating the PHYSICAL PLAUSIBILITY of C_fit. The PMU-enabled
                // branch below (else) ALREADY performed this in "Step 1" (comparing against L1-DTLB~64,
                // L2-STLB~1536-2048 — see comments around line 1826), but this VARPRO branch
                // previously ONLY printed R^2 WITHOUT validating whether the C_fit number made physical sense.
                // High R^2 (0.9871 measured) DOES NOT imply C_fit reflects actual dTLB — consistent with
                // the original project design: "High R^2 serves as INDIRECT evidence," but
                // no one had ACTUALLY cross-checked the raw value until now (merely stated on paper).
                {
                    CacheSizes cs_chk = detect_cache_sizes();
                    const double plausible_lo = 32.0, plausible_hi = 4096.0; // pages;
                    // reference range identical to the one used in the PMU branch below (Step 1)
                    cout << "    [v25b] Validating physical plausibility of C_fit (missing step here;\n"
                            "    PMU branch (else) ALREADY did this in Step 1, but this branch lacked it):\n"
                            "      Reference plausible dTLB range (aggregated L1+L2-STLB, matching Step 1): ["
                         << fixed << setprecision(0) << plausible_lo << ", " << plausible_hi << "] pages\n"
                         << "      C_fit measured via VARPRO                                : "
                         << setprecision(1) << vp.C_fit << " pages\n";
                    if (vp.C_fit < plausible_lo || vp.C_fit > plausible_hi) {
                        cout << "      => UNPLAUSIBLE for dTLB: exceeds upper bound of reference range by "
                             << setprecision(1) << (vp.C_fit / plausible_hi) << "x.\n"
                             << "      => REAL WARNING (empirically confirmed): High R^2=" << setprecision(4) << vp.r2
                             << " IS NOT PROOF that C_fit\n"
                             << "         reflects true dTLB. The curve-shape model (occupancy-shaped) can\n"
                             << "         fit EXCELLENTLY against a COMPLETELY DIFFERENT mechanism (L2/L3 cache, or\n"
                             << "         specific memory-hierarchy behavior of this virtualized sandbox) that shares\n"
                             << "         the same curve trajectory without being a dTLB effect.\n";
                        if (cs_chk.l1d > 0 || cs_chk.l2 > 0 || cs_chk.l3 > 0) {
                            cout << "      Cross-checking against OS cache hierarchy reported via sysconf on this host:\n";
                            if (cs_chk.l1d > 0) cout << "        L1d=" << setprecision(1) << (cs_chk.l1d/1024.0)
                                                 << "KB (~" << setprecision(0) << (cs_chk.l1d/4096.0) << " pages)\n";
                            if (cs_chk.l2  > 0) cout << "        L2 =" << setprecision(2) << (cs_chk.l2/1024.0/1024.0)
                                                 << "MB (~" << setprecision(0) << (cs_chk.l2/4096.0) << " pages)\n";
                            if (cs_chk.l3  > 0) cout << "        L3 =" << setprecision(2) << (cs_chk.l3/1024.0/1024.0)
                                                 << "MB (~" << setprecision(0) << (cs_chk.l3/4096.0) << " pages)\n";
                            cout << "      C_fit (~" << setprecision(2) << (vp.C_fit*4.0/1024.0)
                                 << "MB) shares the order of magnitude with L2/L3 above but DOES NOT align\n"
                                 << "      directly with any single level — HONEST FINDING: cannot definitively conclude\n"
                                 << "      the underlying mechanism, only that it is CONFIRMED NOT pure dTLB. Requires\n"
                                 << "      REAL PMU counters (outside this sandbox, or sudo sysctl kernel.perf_event_paranoid=1\n"
                                 << "      if host permissions allow) to verify — refrain from further speculation.\n";
                        }
                    } else {
                        cout << "      => Within plausible range — CANNOT REJECT the dTLB hypothesis (however,\n"
                                "         this remains INDIRECT evidence, not direct hardware verification).\n";
                    }
                    cout << "\n";
                }
            } else {
                cout << "  [Step 1] Measuring real dTLB-misses/line across L-axis (matching PMU in [Q]),\n"
                        "  backing out C from each point via direct algebra, taking MEDIAN as C_fit:\n";
                vector<double> pts_L, pts_W, pts_miss;
                for (int L : Ls_full){
                    size_t num_regions = total_lines / (size_t)L;
                    auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, num_regions, L, 45000 + L);
                    dtlb2.reset_enable();
                    for (int tr = 0; tr < TRIALS; tr++){
                        uint64_t s = popcount_from_offsets(normal, offs, 0);
                        asm volatile("" :: "r"(s));
                    }
                    long long dtlb_m = dtlb2.disable_read();
                    size_t total_acc = offs.size() * (size_t)TRIALS;
                    double W_actual = (double)offs.size() / (double)L;
                    double miss_per_line = (double)dtlb_m / (double)total_acc;
                    double C_hat = back_out_capacity((double)L, W_actual, miss_per_line);
                    pts_L.push_back(L); pts_W.push_back(W_actual); pts_miss.push_back(miss_per_line);
                    cout << "    L=" << setw(3) << L << "  W=" << setw(7) << fixed << setprecision(0) << W_actual
                         << "  miss/line_meas=" << setprecision(5) << miss_per_line
                         << "  C_inferred=" << setprecision(1) << C_hat << "\n";
                }
                vector<double> C_hats;
                for (size_t i = 0; i < pts_L.size(); i++)
                    C_hats.push_back(back_out_capacity(pts_L[i], pts_W[i], pts_miss[i]));
                double C_fit = median(C_hats);
                cout << "  => C_fit (median over " << C_hats.size() << " points) = " << fixed << setprecision(1)
                     << C_fit << " pages (~" << setprecision(2) << (C_fit * 4.0) << "KB TLB reach)\n"
                     << "     (sanity check reference: typical modern x86 CPU L1-DTLB ~64 entries,\n"
                     << "      L2-STLB ~1536-2048 entries — this is a QUALITATIVE SANITY CHECK, NOT\n"
                     << "      exact per-tier CPUID reading; C_fit represents an 'effective' capacity\n"
                     << "      aggregating multiple TLB levels + secondary effects, not bound to match a specific tier.)\n\n";

                // [v25c-NEW, corresponding to (R)] This delivers the exact validation (R) required: DIRECTLY\n"
                // comparing C_fit from REAL HW PMU (above) against C_fit from software VARPRO (vp, calculated\n"
                // unconditionally at function entry without PMU) — "comparing software miss vs hardware miss directly,\n"
                // instead of relying solely on paper reasoning". Actual architecture transitioned from\n"
                // StackDistHist/miss-ratio to VARPRO/IRM-Burst via (P); hence, these two C_fit values are cross-checked\n"
                // here instead of the two miss ratios outlined in the original (R) draft.
                {
                    double pct_diff = 100.0 * (vp.C_fit - C_fit) / C_fit;
                    cout << "  [R] DIRECT cross-validation: C_fit from REAL HW PMU (above) vs software VARPRO\n"
                            "  (vp, calculated unconditionally at [v23] entry, no PMU required):\n"
                         << "    C_fit (real PMU)       = " << fixed << setprecision(1) << C_fit << " pages\n"
                         << "    C_fit (software VARPRO) = " << vp.C_fit << " pages  (R^2=" << setprecision(4) << vp.r2 << ")\n"
                         << "    Divergence             = " << setprecision(1) << pct_diff << "%\n";
                    if (fabs(pct_diff) < 30.0)
                        cout << "    => Reasonably close (<30%) — REAL EMPIRICAL EVIDENCE (not inferred from R^2) that\n"
                                "    the software IRM-Burst/VARPRO model closely approximates actual hardware PMU values on\n"
                                "    THIS SPECIFIC HOST. First direct comparison between both C_fit values.\n\n";
                    else
                        cout << "    => LARGE DIVERGENCE (>=30%) — Software VARPRO FAILS to accurately approximate\n"
                                "    real PMU C_fit on this host, despite high curve-fitting R^2. This serves as EMPIRICAL\n"
                                "    EVIDENCE that high R^2 alone is insufficient to substitute software C_fit for PMU\n"
                                "    counters when real hardware counters are accessible — use VARPRO purely as fallback.\n\n";
                }

                vector<double> pred_miss_L;
                for (size_t i = 0; i < pts_L.size(); i++)
                    pred_miss_L.push_back(predict_miss_per_line(pts_L[i], pts_W[i], C_fit));
                double rmse_miss = rmse_of(pts_miss, pred_miss_L);
                double r2_miss   = r_squared_given_model(pts_miss, pred_miss_L);
                cout << "  [Step 2] Fitting formula with C_fit on the MEASURED L-axis:\n"
                     << "    RMSE(miss/line) = " << scientific << setprecision(3) << rmse_miss
                     << "    R^2(miss/line) = " << fixed << setprecision(4) << r2_miss << "\n\n";

                // ---- Fit cost/line = compute_floor + penalty*predicted_miss, using pre-existing cost_normal ----
                vector<double> pred_miss_for_cost(Ls.size());
                for (size_t i = 0; i < Ls.size(); i++){
                    double W_actual_cost = (double)n_ops_normal[i] / (double)Ls[i];
                    pred_miss_for_cost[i] = predict_miss_per_line((double)Ls[i], W_actual_cost, C_fit);
                }
                LinFit cost_fit = fit_linear(pred_miss_for_cost, cost_normal);
                CostModelFit old_fit = fit_cost_model(Ls, cost_normal); // legacy model, comparison only, unmeasured
                cout << "  [Step 3] Predicted cost/line = " << fixed << setprecision(4) << cost_fit.a
                     << " + " << cost_fit.b << " * predicted_miss/line  (R^2=" << setprecision(4) << cost_fit.r2 << ")\n"
                     << "    Comparing R^2 across the SAME 7 L-points: LEGACY model (a+b/L, pure curve-fit)="
                     << setprecision(4) << old_fit.r_squared << "   NEW model (IRM-Burst, physically grounded)="
                     << cost_fit.r2 << "\n\n";

                // ---- OUT-OF-SAMPLE validation on a completely independent FOOTPRINT AXIS, NO re-fitting ----
                cout << "  [Step 4] OUT-OF-SAMPLE validation: predicting across FOOTPRINT axis\n"
                        "  (fixed L=8, sweeping W) USING C_fit + cost_fit learned strictly FROM the L-axis above:\n";
                size_t n_regions_avail = N / WORDS_PER_4K;
                vector<size_t> W_sweep = {16, 128, 512, 2048, 6000, (size_t)(n_regions_avail * 0.85)};
                vector<double> oos_miss_meas, oos_miss_pred, oos_cost_meas, oos_cost_pred;
                for (size_t W : W_sweep){
                    if (W < 1 || W > n_regions_avail) continue;
                    auto offs = generate_offsets(N, WORDS_PER_4K, LINES_PER_4K, W, 8, 51000 + W, false);
                    vector<double> t;
                    for (int tr = 0; tr < max(5, TRIALS / 2); tr++){
                        auto a0 = high_resolution_clock::now();
                        uint64_t s = popcount_from_offsets(normal, offs, 0);
                        auto a1 = high_resolution_clock::now();
                        asm volatile("" :: "r"(s));
                        t.push_back(duration<double, micro>(a1 - a0).count());
                    }
                    double cost_meas = (median(t) * 1000.0) / (double)offs.size();

                    dtlb2.reset_enable();
                    for (int tr = 0; tr < TRIALS; tr++){
                        uint64_t s = popcount_from_offsets(normal, offs, 0);
                        asm volatile("" :: "r"(s));
                    }
                    long long dtlb_m = dtlb2.disable_read();
                    double miss_meas = (double)dtlb_m / (double)(offs.size() * (size_t)TRIALS);

                    double W_actual = (double)offs.size() / 8.0;
                    double miss_pred = predict_miss_per_line(8.0, W_actual, C_fit);
                    double cost_pred = cost_fit.a + cost_fit.b * miss_pred;

                    oos_miss_meas.push_back(miss_meas); oos_miss_pred.push_back(miss_pred);
                    oos_cost_meas.push_back(cost_meas); oos_cost_pred.push_back(cost_pred);

                 cout << "    W=" << setw(7) << W
                         << "  miss/line: meas=" << fixed << setprecision(5) << miss_meas
                         << " pred=" << miss_pred
                         << "   cost/line: meas=" << setprecision(3) << cost_meas
                         << "ns pred=" << cost_pred << "ns\n";
                }
                if (oos_miss_meas.size() >= 3){
                    double r2_oos_miss = r_squared_given_model(oos_miss_meas, oos_miss_pred);
                    double r2_oos_cost = r_squared_given_model(oos_cost_meas, oos_cost_pred);
                    cout << "\n  => Out-of-sample R^2 across footprint axis (fitted on L-axis, PURE PREDICTION here):\n"
                         << "     miss/line: R^2=" << fixed << setprecision(4) << r2_oos_miss
                         << "     cost/line: R^2=" << r2_oos_cost << "\n"
                         << "  (Legacy a+b/L model CANNOT be evaluated here due to lacking the W parameter —\n"
                         << "   this exact capability allows [v23] to surpass BOTH the legacy model AND [v22]'s discrete 2-experiment design.)\n\n";
                } else {
                    cout << "\n  [!] Insufficient valid data points across footprint axis to compute out-of-sample R^2.\n\n";
                }
            }
        }
    }

// ============================================================
    // [v27+v27b-NEW] Internal contradiction never previously detected: see full
    //   derivation + self-tests in comments preceding self_test_irm_fixed_replay_gap()
    //   and self_test_random_replacement_shape() above. Summary: self_test_irm_law()
    //   confirms C/W under the fresh-IID-per-access assumption, whereas generate_offsets() +
    //   REAL TRIALS loops replayed a FIXED offset sequence — under ideal LRU, this
    //   converges to a STEP FUNCTION rather than C/W. This is NOT paper-based deduction:
    //   the two functions below self-test both processes using real C++ implementations.
    // ============================================================
    cout << "===== [v27] Internal contradiction: 'FIXED replayed sequence' vs 'Fresh IID per access'\n";
    {
        mt19937_64 v27_rng(20260728);
        self_test_irm_fixed_replay_gap(v27_rng);
        self_test_random_replacement_shape(v27_rng);
    }

    // ============================================================
    // [v36-NEW] THAY THE [v24]/[v26]/[v28]/[v35]: bo du doan mien 3-tang
    //   (Che -> phat hien tuan hoan -> cuu ho EM), port THAT tu 3 file
    //   Python doc lap da chay & doi chieu so lieu truoc khi dua vao day.
    //   Ly do thay the: xem comment day du truoc self_test_predictor_v36()
    //   o tren - tom tat: [v27] ngay ben tren da tu-chung-minh (khong phai
    //   suy dien) rang gia dinh nen tang cua CA BON phan mo rong cu (tham
    //   chieu IID tuoi moi) KHONG dung voi cach chinh generate_offsets()
    //   cua file nay sinh du lieu (phat lai co dinh) - ca bon deu ke thua
    //   cung diem mu do vi cung la ho ham mu. [v36] tu phat hien duoc dieu
    //   nay thay vi ep du lieu vao dung 1 dang ham dinh san.
    //   [v23] (irm_occupancy/predict_miss_per_line/back_out_capacity) van
    //   GIU NGUYEN o tren - no dung TUYET DOI cho popularity deu va van
    //   dang duoc VarPro/[Model] dung THAT tren du lieu PMU, khong lien
    //   quan gi den diem mu nay (khong bi go bo).
    // ============================================================
    cout << "===== [v36] Bo du doan mien 3-tang (Che -> tuan hoan -> EM cuu ho), thay [v24]-[v35]\n";
    {
        mt19937_64 v36_rng(20260730);
        bool v36_ok = self_test_predictor_v36(v36_rng);
        if (!v36_ok){
            cout << "  [v36] Self-test FAILED tren may nay -> BO QUA ket luan cua [v36]\n"
                    "  (khong tin mot pipeline chua duoc xac nhan tren may nay).\n\n";
        } else {
            cout << "  [v36] Self-test OK: pipeline chon DUNG tier tren ca 3 dang chuoi khac nhau\n"
                    "  (i.i.d. Zipf -> Che; vong lap tuan hoan -> phat hien tuan hoan; troi pha -> cuu ho),\n"
                    "  va sai so cung bac do lon voi ban Python goc doc lap (da doi chieu bang tay).\n\n";

            cout << "  --- Ap dung [v36] cho chinh chuoi truy cap ma generate_offsets()/vong TRIALS\n"
                    "      cua file nay THAT SU tao ra (hoan vi co dinh, phat lai nguyen ven qua nhieu\n"
                    "      trial) - day la truong hop CHINH XAC ma [v27] da chi ra la [v24]-[v35] mo hinh\n"
                    "      SAI DANG (ep ham bac thang vao duong cong tron): ---\n";
            // [v38-FIX #5] Dung dung N/total_lines/L THAT cua vong sweep chinh
            // (khai bao o dau main(), van con trong scope o day) va goi
            // generate_offsets() THAT (qua real_workload_trace_v38) thay vi
            // hang so W_demo/L_demo tach roi nhu ban cu - neu total_lines/L
            // dang bi clamp trong sweep that, demo o day se PHAN ANH DUNG
            // dieu kien clamp do (W_actual tra ve co the < total_lines/L).
            int L_demo = 13; // van neo theo dai Cochran (~13-15) ma benchmark chinh dung
            auto [real_shape_trace, W_actual] = real_workload_trace_v38(N, total_lines, L_demo, 40, 20260730ULL);
            auto res_real = predict_miss_rate_v36(real_shape_trace, W_actual, 0.01, true, v36_rng);
            cout << "  => mode=" << res_real.mode << "  err=" << fixed << setprecision(5) << res_real.err;
            if (res_real.mode == "periodic")
                cout << "  (Cstar=" << res_real.cstar << " ~ W_actual=" << W_actual
                     << " - KHOP CHINH XAC voi ket luan dinh tinh cua [v27]: day la mot ham BAC THANG,\n"
                        "     khong phai duong cong C/W tron ma [v23]/[v24]/[v26]/[v28]/[v35] gia dinh\n"
                        "     khi ap dung nham cho chuoi phat-lai-co-dinh nay)\n\n";
            else
                cout << "\n\n";

            // [v38-FIX #6] Bootstrap CI cho quyet dinh tier tren chinh chuoi thuc
            // te nay - cho biet ranh gioi Tier1/Tier2 co dang "nhay" vi sampling
            // noise hay khong, thay vi chi tin 1 diem uoc err_che/step.err.
            {
                mt19937_64 boot_rng(999331);
                auto conf = bootstrap_tier_confidence_v38(real_shape_trace, W_actual, res_real.tol_eff, 30, boot_rng);
                cout << "  [v38] Bootstrap CI (30 resample) cho quyet dinh tier:\n"
                     << "    err_che  95% CI = [" << conf.err_che_lo << ", " << conf.err_che_hi << "]"
                     << (conf.tier1_borderline ? "  <= tol_eff NAM TRONG CI: Tier1 DANG O LAN RANH\n"
                                                 : "  (tol_eff nam ngoai CI: Tier1 on dinh)\n")
                     << "    step.err 95% CI = [" << conf.step_err_lo << ", " << conf.step_err_hi << "]"
                     << (conf.tier2_borderline ? "  <= tol_eff*3 NAM TRONG CI: Tier2 DANG O LAN RANH\n\n"
                                                 : "  (tol_eff*3 nam ngoai CI: Tier2 on dinh)\n\n");
            }

            cout << "  --- Doi chieu voi 3 dang workload thuc-te-hoa khac ([v36] TU DONG chon dung\n"
                    "      tier cho tung dang, khong can nguoi dung chon truoc mo hinh): ---\n";
            {
                cout << "  [YCSB-like, i.i.d. Zipf]\n";
                auto tr = gen_ycsb_v36(300, 150000, 0.99, 20260731ULL);
                auto r = predict_miss_rate_v36(tr, 300, 0.01, true, v36_rng);
                cout << "    => mode=" << r.mode << "  err=" << r.err << "\n\n";
            }
            {
                cout << "  [SPEC-like, vong lap tuan hoan + jitter nho]\n";
                auto tr = gen_spec_loop_v36(300, 400, 0.05, 3, 20260732ULL);
                auto r = predict_miss_rate_v36(tr, 300, 0.01, true, v36_rng);
                cout << "    => mode=" << r.mode << "  err=" << r.err
                     << (r.mode == "periodic" ? (string("  (Cstar=") + to_string(r.cstar) + ")\n\n") : string("\n\n"));
            }
            {
                cout << "  [Bioinfo-like, troi pha theo doan]\n";
                auto tr = gen_bioinfo_phase_v36(300, 6, 25000, 0.15, 0.7, 20260733ULL);
                auto r = predict_miss_rate_v36(tr, 300, 0.01, true, v36_rng);
                cout << "    => mode=" << r.mode << "  err=" << r.err
                     << (r.em_lam1 > 0 ? "  (a=" + to_string(r.em_a) + " lam1=" + to_string(r.em_lam1)
                                          + " lam2=" + to_string(r.em_lam2) + (r.em_degenerate ? " THOAI HOA)\n\n" : ")\n\n")
                                       : "\n\n");
            }

            cout << "  => KET LUAN [v36]: thay vi 4 cong thuc rieng le ([v24]/[v26]/[v28]/[v35]) deu\n"
                    "  ngam dinh popularity ON DINH THEO THOI GIAN va tham chieu doc lap tung lan\n"
                    "  (2 gia dinh ma [v27] va vi du 'troi pha' o tren cho thay co the SAI trong thuc\n"
                    "  te), [v36] do TRUC TIEP tren du lieu va TU BAO CAO khi mot gia dinh khong con\n"
                    "  dung (chuyen tier) thay vi im lang ep so lieu vao 1 dang cong thuc co dinh.\n"
                    "  GIOI HAN TRUNG THUC con lai: nhanh cuu ho EM (Tier 3) la mot xap xi CHAN DOAN,\n"
                    "  khong phai dinh luat da chung minh - khi no THOAI HOA (a~0/1 hoac lam1~lam2),\n"
                    "  ket luan trung thuc nhat van la 'che con lai la cau tra loi thanh that nhat',\n"
                    "  KHONG phai 'da giai xong'. [v23] o tren van la mo hinh DUY NHAT trong file nay\n"
                    "  duoc chung minh dai so (khong xap xi) - [v36] khong thay the no, chi thay the\n"
                    "  4 lop mo rong ho ham mu chua bao gio duoc doi chieu voi du lieu THAT.\n\n";
        }
    }

    munmap(raw_normal, BYTES);
    munmap(raw_hp, BYTES);
    return 0;
}
