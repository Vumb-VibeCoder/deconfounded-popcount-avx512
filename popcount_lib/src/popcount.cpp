// src/popcount.cpp — Implementation cua libpopcount.
//
// Cac kernel SIMD duoc ke thua truc tiep (khong doi logic) tu du an nghien
// cuu popcount_v39 - da qua nhieu vong self-test bit-identical o do. Phan
// "nghien cuu" (benchmark, thong ke bootstrap, mo hinh IRM/cache...) KHONG
// nam trong thu vien nay - o day chi giu lai dung phan "kernel + dispatch"
// dung duoc nhu 1 thu vien popcount thong thuong.
#include "popcount/popcount.h"

#include <immintrin.h>
#include <cpuid.h>
#include <xmmintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace popcount_detail {

using std::size_t;
using std::uint64_t;
using std::uint32_t;

// ---------------------------------------------------------------------
// Kernel scalar (baseline / fallback, luon dung duoc tren moi CPU x86-64)
// ---------------------------------------------------------------------
static inline uint64_t popcount64_hw(uint64_t x) { return __builtin_popcountll(x); }

static uint64_t popcount_array_scalar(const uint64_t* data, size_t n) {
    uint64_t t = 0;
    for (size_t i = 0; i < n; i++) t += popcount64_hw(data[i]);
    return t;
}

static uint64_t popcount_block512_scalar(const uint64_t* base) {
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) s += popcount64_hw(base[i]);
    return s;
}

static uint64_t popcount_gather_scalar(const uint64_t* data, const size_t* offsets,
                                        size_t n_offsets, int prefetch_distance) {
    uint64_t sum = 0;
    if (prefetch_distance > 0) {
        size_t pd = (size_t)prefetch_distance;
        for (size_t i = 0; i < n_offsets; i++) {
            if (i + pd < n_offsets)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            sum += popcount_block512_scalar(data + offsets[i]);
        }
    } else {
        for (size_t i = 0; i < n_offsets; i++)
            sum += popcount_block512_scalar(data + offsets[i]);
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)

// ---------------------------------------------------------------------
// Kernel AVX2 (nibble-LUT cho gather; Harley-Seal CSA cho mang lien tuc)
// ---------------------------------------------------------------------
__attribute__((target("avx2")))
static inline __m256i popcnt8_epi8_avx2(__m256i v, __m256i lut, __m256i low_mask) {
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i popcnt_lo = _mm256_shuffle_epi8(lut, lo);
    __m256i popcnt_hi = _mm256_shuffle_epi8(lut, hi);
    __m256i byte_sums = _mm256_add_epi8(popcnt_lo, popcnt_hi);
    return _mm256_sad_epu8(byte_sums, _mm256_setzero_si256());
}

__attribute__((target("avx2")))
static uint64_t popcount_block512_avx2(const uint64_t* base) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 4));
    __m256i acc = _mm256_add_epi64(popcnt8_epi8_avx2(v0, lut, low_mask),
                                    popcnt8_epi8_avx2(v1, lut, low_mask));
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

__attribute__((target("avx2")))
static uint64_t popcount_gather_avx2(const uint64_t* data, const size_t* offsets,
                                      size_t n_offsets, int prefetch_distance) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    size_t n = n_offsets;
    // [opt] tach nhanh prefetch_distance>0 (bat-bien trong ham) ra khoi than
    // vong lap nong - 2 vong lap rieng (khong dung lambda o day: lambda cuc
    // bo trong ham co __attribute__((target(...))) khong luon duoc GCC ap
    // dung dung target cho operator() sinh ra, gay loi "target specific
    // option mismatch" luc inline - da xac nhan thuc nghiem. Nhan ban than
    // vong lap la cach an toan, giong nguyen ban popcount_v39).
    if (prefetch_distance > 0) {
        size_t pd = (size_t)prefetch_distance;
        for (size_t i = 0; i < n; i++) {
            if (i + pd < n)
                _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + pd]), _MM_HINT_T0);
            const uint64_t* base = data + offsets[i];
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + 4));
            acc0 = _mm256_add_epi64(acc0, popcnt8_epi8_avx2(v0, lut, low_mask));
            acc1 = _mm256_add_epi64(acc1, popcnt8_epi8_avx2(v1, lut, low_mask));
        }
    } else {
        for (size_t i = 0; i < n; i++) {
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

__attribute__((target("avx2")))
static inline void csa256(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c) {
    __m256i u = _mm256_xor_si256(a, b);
    h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    l = _mm256_xor_si256(u, c);
}

// Harley-Seal CSA (carry-save adder) popcount cho mang uint64_t lien tuc.
// Nhanh hon nibble-LUT tung-vector tren mang lon vi giam so lan shuffle/sad
// tren moi vector (gom 16 vector 256-bit truoc khi rut gon 1 lan).
__attribute__((target("avx2")))
static uint64_t popcount_array_avx2_harleyseal(const uint64_t* data, size_t n) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i* vdata = reinterpret_cast<const __m256i*>(data);
    size_t nvec = n / 4;

    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256(), twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256(), eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

#define POPCOUNT_LD(idx) _mm256_loadu_si256(vdata + (idx))
    size_t i = 0;
    for (; i + 16 <= nvec; i += 16) {
        csa256(twosA, ones, ones, POPCOUNT_LD(i+0), POPCOUNT_LD(i+1));
        csa256(twosB, ones, ones, POPCOUNT_LD(i+2), POPCOUNT_LD(i+3));
        csa256(foursA, twos, twos, twosA, twosB);
        csa256(twosA, ones, ones, POPCOUNT_LD(i+4), POPCOUNT_LD(i+5));
        csa256(twosB, ones, ones, POPCOUNT_LD(i+6), POPCOUNT_LD(i+7));
        csa256(foursB, twos, twos, twosA, twosB);
        csa256(eightsA, fours, fours, foursA, foursB);
        csa256(twosA, ones, ones, POPCOUNT_LD(i+8), POPCOUNT_LD(i+9));
        csa256(twosB, ones, ones, POPCOUNT_LD(i+10), POPCOUNT_LD(i+11));
        csa256(foursA, twos, twos, twosA, twosB);
        csa256(twosA, ones, ones, POPCOUNT_LD(i+12), POPCOUNT_LD(i+13));
        csa256(twosB, ones, ones, POPCOUNT_LD(i+14), POPCOUNT_LD(i+15));
        csa256(foursB, twos, twos, twosA, twosB);
        csa256(eightsB, fours, fours, foursA, foursB);
        csa256(sixteens, eights, eights, eightsA, eightsB);
        total = _mm256_add_epi64(total, popcnt8_epi8_avx2(sixteens, lut, low_mask));
    }
#undef POPCOUNT_LD
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(eights, lut, low_mask), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(fours, lut, low_mask), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcnt8_epi8_avx2(twos, lut, low_mask), 1));
    total = _mm256_add_epi64(total, popcnt8_epi8_avx2(ones, lut, low_mask));

    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), total);
    uint64_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (size_t j = i; j < nvec; j++) {
        __m256i v = _mm256_loadu_si256(vdata + j);
        alignas(32) uint64_t t2[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(t2), popcnt8_epi8_avx2(v, lut, low_mask));
        sum += t2[0] + t2[1] + t2[2] + t2[3];
    }
    for (size_t k = nvec * 4; k < n; k++) sum += popcount64_hw(data[k]);
    return sum;
}

// ---------------------------------------------------------------------
// Kernel AVX-512 VPOPCNTDQ (8-accumulator cho mang lien tuc; 2-accumulator
// cho gather - rong accumulator khong giup gather vi bottleneck la do tre
// truy cap bo nho ngau nhien, khong phai compute).
// ---------------------------------------------------------------------
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_block512_avx512(const uint64_t* base) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(base));
    __m512i p = _mm512_popcnt_epi64(v);
    alignas(64) uint64_t t[8];
    _mm512_store_si512(reinterpret_cast<void*>(t), p);
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) s += t[i];
    return s;
}

__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_gather_avx512(const uint64_t* data, const size_t* offsets,
                                        size_t n_offsets, int prefetch_distance) {
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    size_t n = n_offsets;
    size_t i = 0;
    // [opt] xem giai thich (khong dung lambda) o popcount_gather_avx2() phia tren.
    if (prefetch_distance > 0) {
        size_t pd = (size_t)prefetch_distance;
        for (; i + 1 < n; i += 2) {
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
        for (; i + 1 < n; i += 2) {
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
    for (; i < n; i++) sum += popcount_block512_avx512(data + offsets[i]);
    return sum;
}

__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_array_avx512_8acc(const uint64_t* data, size_t n) {
    __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
    __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
    __m512i acc4=_mm512_setzero_si512(), acc5=_mm512_setzero_si512();
    __m512i acc6=_mm512_setzero_si512(), acc7=_mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
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
    for (; i + 8 <= n; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(data+i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int k = 0; k < 8; k++) sum += t[k];
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)((1u << (n - i)) - 1);
        __m512i v = _mm512_maskz_loadu_epi64(mask, (const void*)(data+i));
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), _mm512_popcnt_epi64(v));
        for (int k = 0; k < 8; k++) sum += t[k];
    }
    return sum;
}

// ---------------------------------------------------------------------
// Phat hien CPU luc chay (CPUID + XCR0 - kiem tra ca OS co luu/khoi phuc
// state YMM/ZMM, khong chi CPU co ho tro).
// ---------------------------------------------------------------------
__attribute__((target("xsave")))
static inline unsigned long long read_xcr0() { return _xgetbv(0); }

static popcount_simd_level_t detect_simd_level() {
    unsigned eax1, ebx1, ecx1, edx1;
    if (!__get_cpuid(1, &eax1, &ebx1, &ecx1, &edx1)) return POPCOUNT_LEVEL_SCALAR;
    bool osxsave = (ecx1 >> 27) & 1;
    bool avx_cpu = (ecx1 >> 28) & 1;
    if (!osxsave || !avx_cpu) return POPCOUNT_LEVEL_SCALAR;

    unsigned long long xcr0 = read_xcr0();
    bool os_avx    = (xcr0 & 0x6)  == 0x6;
    bool os_avx512 = (xcr0 & 0xE6) == 0xE6;

    unsigned eax7, ebx7, ecx7, edx7;
    if (!__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7)) return POPCOUNT_LEVEL_SCALAR;
    bool has_avx2            = (ebx7 >> 5)  & 1;
    bool has_avx512f         = (ebx7 >> 16) & 1;
    bool has_avx512vpopcntdq = (ecx7 >> 14) & 1;

    if (os_avx512 && has_avx512f && has_avx512vpopcntdq) return POPCOUNT_LEVEL_AVX512_VPOPCNTDQ;
    if (os_avx && has_avx2) return POPCOUNT_LEVEL_AVX2;
    return POPCOUNT_LEVEL_SCALAR;
}

#endif // __x86_64__

// ---------------------------------------------------------------------
// Bang dispatch (function pointer) - khoi tao 1 lan, thread-safe qua
// std::once_flag, sau do CHI DOC tu nhieu thread (an toan, khong can lock).
// ---------------------------------------------------------------------
using ArrayFn  = uint64_t(*)(const uint64_t*, size_t);
using Block512Fn = uint64_t(*)(const uint64_t*);
using GatherFn = uint64_t(*)(const uint64_t*, const size_t*, size_t, int);

static std::atomic<ArrayFn>  g_array_fn{popcount_array_scalar};
static std::atomic<Block512Fn> g_block512_fn{popcount_block512_scalar};
static std::atomic<GatherFn> g_gather_fn{popcount_gather_scalar};
static std::atomic<popcount_simd_level_t> g_level{POPCOUNT_LEVEL_SCALAR};
static std::string g_level_name = "scalar (POPCNT)";
static std::once_flag g_init_once;

static void init_dispatch_once() {
#if defined(__x86_64__) || defined(_M_X64)
    popcount_simd_level_t level = detect_simd_level();
    // Cho phep ep muc SIMD qua bien moi truong (huu ich cho testing/benchmark
    // so sanh cong bang tren cung 1 may). Khong set -> tu dong phat hien.
    if (const char* force = std::getenv("POPCOUNT_FORCE")) {
        std::string f(force);
        for (auto& c : f) c = (char)tolower((unsigned char)c);
        if (f == "scalar") level = POPCOUNT_LEVEL_SCALAR;
        else if (f == "avx2" && level >= POPCOUNT_LEVEL_AVX2) level = POPCOUNT_LEVEL_AVX2;
        else if (f == "avx512" && level == POPCOUNT_LEVEL_AVX512_VPOPCNTDQ) level = POPCOUNT_LEVEL_AVX512_VPOPCNTDQ;
    }
    switch (level) {
        case POPCOUNT_LEVEL_AVX512_VPOPCNTDQ:
            g_array_fn = popcount_array_avx512_8acc;
            g_block512_fn = popcount_block512_avx512;
            g_gather_fn = popcount_gather_avx512;
            g_level_name = "AVX-512 VPOPCNTDQ (array: 8-acc | gather: 2-acc)";
            break;
        case POPCOUNT_LEVEL_AVX2:
            g_array_fn = popcount_array_avx2_harleyseal;
            g_block512_fn = popcount_block512_avx2;
            g_gather_fn = popcount_gather_avx2;
            g_level_name = "AVX2 (array: Harley-Seal CSA | gather: nibble-LUT)";
            break;
        default:
            level = POPCOUNT_LEVEL_SCALAR;
            g_level_name = "scalar (POPCNT, 1 word/lenh)";
    }
    g_level = level;
#else
    g_level_name = "scalar (POPCNT, non-x86_64)";
#endif
}

static void ensure_init() { std::call_once(g_init_once, init_dispatch_once); }

static unsigned pick_n_threads(unsigned requested, size_t n) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned n_threads = (requested == 0) ? hw : std::min(requested, hw);
    if (n_threads < 1) n_threads = 1;
    // Nguong toi thieu de bu chi phi tao/dieu phoi thread.
    if (n < 4096 || n_threads == 1) return 1;
    return n_threads;
}

} // namespace popcount_detail

// ===========================================================================
// C API
// ===========================================================================
extern "C" {

void popcount_init(void) { popcount_detail::ensure_init(); }

popcount_simd_level_t popcount_active_level(void) {
    popcount_detail::ensure_init();
    return popcount_detail::g_level.load();
}

const char* popcount_active_level_name(void) {
    popcount_detail::ensure_init();
    return popcount_detail::g_level_name.c_str();
}

uint64_t popcount_block512(const uint64_t block[8]) {
    popcount_detail::ensure_init();
    return popcount_detail::g_block512_fn.load()(block);
}

uint64_t popcount_array(const uint64_t* data, size_t count) {
    popcount_detail::ensure_init();
    return popcount_detail::g_array_fn.load()(data, count);
}

uint64_t popcount_array_mt(const uint64_t* data, size_t count, unsigned n_threads_req) {
    popcount_detail::ensure_init();
    unsigned n_threads = popcount_detail::pick_n_threads(n_threads_req, count);
    if (n_threads == 1) return popcount_detail::g_array_fn.load()(data, count);

    std::vector<uint64_t> partial(n_threads, 0);
    size_t chunk = count / n_threads;
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    auto fn = popcount_detail::g_array_fn.load();
    for (unsigned t = 0; t < n_threads; t++) {
        size_t begin = (size_t)t * chunk;
        size_t end = (t == n_threads - 1) ? count : begin + chunk;
        pool.emplace_back([&partial, fn, data, begin, end, t] {
            partial[t] = fn(data + begin, end - begin);
        });
    }
    for (auto& th : pool) th.join();
    uint64_t sum = 0;
    for (uint64_t p : partial) sum += p;
    return sum;
}

uint64_t popcount_bulk_gather(const uint64_t* data, const size_t* offsets,
                               size_t n_offsets, int prefetch_distance) {
    popcount_detail::ensure_init();
    return popcount_detail::g_gather_fn.load()(data, offsets, n_offsets, prefetch_distance);
}

int popcount_self_test(uint32_t seed, size_t n_blocks) {
    popcount_detail::ensure_init();
    std::mt19937_64 rng(seed);
    alignas(64) uint64_t buf[8];
    auto active_fn = popcount_detail::g_block512_fn.load();
    for (size_t b = 0; b < n_blocks; b++) {
        for (int w = 0; w < 8; w++) buf[w] = rng();
        uint64_t expect = popcount_detail::popcount_block512_scalar(buf);
        uint64_t got = active_fn(buf);
        if (got != expect) return 0;
    }
    return 1;
}

} // extern "C"
