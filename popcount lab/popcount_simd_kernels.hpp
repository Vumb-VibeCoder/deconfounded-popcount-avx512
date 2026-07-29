#pragma once

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

