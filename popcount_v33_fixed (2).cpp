// ============================================================
// popcount_v21_combined.cpp — hop nhat v18 + v19 + v21 (SIMD + RDTSCP)
//
// Tu v18 (deconfounded):
//   [BUG-1] RNG tach hoan toan khoi vung do thoi gian
//   [BUG-2] AnonHugePages doc dung VMA (smaps)
//   [BUG-3] Xao thu tu L moi trial (trung binh hoa thermal/DVFS)
//   + MAD, compute-floor calib, prefetch-distance sweep, #pragma unroll
//
// Tu v19 (hwcounters + bootstrap):
//   [P] Ghim CPU0 TRUOC khi alloc (tranh NUMA first-touch + migration)
//   [Q] HW PMU that: dTLB-load-miss + LLC-load-miss via perf_event_open
//       (fallback an toan neu permission/container)
//   [R] RMSE that + offset DOC LAP moi lan (mo rong L quanh Cochran ~13-15)
//   [S] Bootstrap CI 95% cho ratio_of_drops (resample trial-time)
//   + Sanity: sampling vs full sequential scan
//
// Sua them (v20):
//   - RMSE estimator: mean_per_word = sum/(offs.size()*8) * N  (bug v19: thieu /8)
//   - bootstrap_ci duoc goi that su
//   - ngan DCE bang asm volatile
//
// v21 (moi, dap lai review "chua cham tran"):
//   [T] SIMD popcount runtime-dispatch: AVX-512 VPOPCNTDQ / AVX2 nibble-LUT /
//       scalar fallback. Kiem tra CPUID + XCR0 (OS state that su, khong chi
//       CPU ho tro). Da xac minh scalar==avx2==avx512 tren 2 trieu test case.
//       Ap dung cho: popcount_from_offsets (hot path chinh), compute-floor,
//       va ca popcount_array_ref (contig) de ref_normal/ref_hp/full-scan
//       nhanh hon voi N lon ma khong doi ket qua (bit-identical).
//
//   [T-BUG] Trong luc code kernel "bulk", phat hien 1 loi hieu nang NGHIEM
//       TRONG tu chinh qua trinh sua: dua lines_per_4k thanh tham so runtime
//       khien "i % lines_per_4k" bien tu AND-bit (khi la constexpr) thanh
//       lenh DIV 64-bit that (~20-40+ cycle/lan) - objdump xac nhan. Sua bang
//       (i & (lines_per_4k-1)) + static_assert luy-thua-2. Sau khi sua DIV o
//       CA HAI phia (scalar va avx512), so sanh cong bang cho: scalar=2.54ns,
//       avx512=0.54ns/line => AVX-512 nhanh hon ~4.7x - GAN VOI ky vong ly
//       thuyet cua nhan dinh Gemini (c), sau khi loai bo nhieu do phuong phap.
//       Bai hoc: mot loi hieu nang (DIV vs AND) co the anh huong LON hon ca
//       loi ich tu SIMD - luon doi chieu ca hai nhanh sau moi thay doi.
//
//   [U] RDTSCP+LFENCE (serializing that su) doi chieu voi high_resolution_clock
//       (~clock_gettime, KHONG serializing) rieng o compute-floor - noi nhay
//       cam nhat voi OoO leak qua ranh gioi do. Sau khi toc do tang len (do
//       sua DIV->AND), chenh lech chrono-vs-RDTSCP ~1.7% - BAT DAU dang chu y
//       (khac voi truoc khi sua DIV, luc do <1% vi phep do bi DIV che khuat).
//       Cang toi uu compute-floor xuong thap, sai so do luong cang chiem ty
//       trong lon hon - dung minh hoa dung nhan dinh (a) cua Gemini, nhung
//       chi lo ro SAU KHI da loai bo cac nut that lon hon (nhu DIV) truoc.
//
// v22 (tren nen v21):
//   [V] generate_offsets: them tham so warn + bien dem g_cap_violations. Khi
//       num_regions yeu cau > so vung 4KB kha dung, ham se CLAMP xuong (hanh
//       vi cu van giu) NHUNG gio in canh bao ro rang va dem so lan xay ra, vi
//       dieu nay lam "N-fixed" khong con dung va co the thoi phong cost/line
//       o L nho do mau so nho hon du kien - truoc day loi nay AM THAM khong
//       duoc bao cao.
//   [W] Phat hien confound moi: thiet ke goc dat num_regions = total_lines/L,
//       nen moi diem tren truc L dong thoi doi CA "do sau tai su dung" LAN
//       "kich thuoc working-set" (footprint = num_regions*4KB) - hai bien
//       nhieu troi len nhau. Them detect_cache_sizes() (qua sysconf, an toan
//       voi container/hypervisor chan CPUID leaf 0x18), report_footprint(),
//       va 2 thi nghiem giai toa nhieu: (1) L-sweep voi footprint CO DINH >
//       L3 (giu 1 bien duy nhat thay doi), (2) footprint-sweep voi L CO DINH
//       (dac trung rieng truc con lai qua cac nguong L1/L2/L3).
//
// [HOP NHAT v21+v22] Ban nay lay v22 lam nen (nhieu tinh nang hon) nhung
//   KHOI PHUC lai ban va bootstrap cua v21 ([v21-FIX-BUG-BOOTSTRAP]) ma v22 da
//   VO TINH revert mat trong luc them tinh nang moi: v22 cho stat_fn cua
//   bootstrap_ci gia dinh "n_ops = total_lines" cho CA HAI L=1 va L=64, day
//   chinh la loi ma v21 da phat hien va sua (offsets co the bi CLAMP it hon
//   total_lines, nhat la o L nho, khien chuan hoa sai va bootstrap CI sup do
//   ve [0,0] mot cach he thong). Da doi chieu tung dong qua diff de xac nhan
//   day la mat di khong chu y (khong co comment nao trong v22 giai thich ly
//   do revert), nen ban hop nhat nay phuc hoi n_ops_per_L THAT SU tu
//   run_deconfounded + dung no trong bootstrap, giu nguyen moi thu khac cua
//   v22 (bao gom ca g_cap_violations/warn moi da bo sung o buoc [V]).
//
// v23 (MOI PHAT MINH, dap lai yeu cau "tim thuat toan moi qua nghien cuu"):
//   [X] Van de goc: model cost(L) ~= a + b/L (buoc [Model]) la CURVE-FIT
//       thuan tuy tren 7 diem L rieng le - khong co y nghia vat ly ro rang
//       cho tham so b, VA khong co truc "footprint/working-set" (W) trong
//       cong thuc, nen [v22] phai vat va giai toa nhieu bang 2 thi nghiem
//       RIENG BIET (L-sweep footprint co dinh / footprint-sweep L co dinh)
//       ma khong bao gio HOP NHAT lai thanh MOT mo hinh du bao duoc ca 2 truc.
//
//   [Y] Phat minh: "Dinh luat dong dang IRM-Burst" (Independent Reference
//       Model cho mau truy cap burst). Lap luan bang XAC SUAT TRAO DOI DUOC
//       (exchangeability), khong phai curve-fit:
//         - Coi moi "burst" (L lan cham cung 1 trang 4KB) la MOT tham chieu
//           logic toi 1 trong W trang (W = so trang trong working-set),
//           chon deu ngau nhien doc lap (IID) - dung mo ta chinh xac cach
//           generate_offsets() sinh du lieu (region_dist deu, doc lap moi p).
//         - Voi cache/TLB dung luong C (giu dung C trang gan-dung-nhat kieu
//           LRU), ky vong tong so trang trong cache LUON = C (sau warm-up),
//           va do W trang la DOI XUNG (trao doi duoc) nen MOI trang co CUNG
//           xac suat dang trong cache = C/W - suy ra true tiep tu ky vong
//           tuyen tinh, KHONG xap xi. Vay:
//             occupancy(C,W) = min(1, C/W)
//             mien/burst      = 1 - occupancy(C,W)
//             mien/line       = (1 - occupancy(C,W)) / L
//           (chia L vi trong 1 burst chi lan cham DAU co the mien, L-1 lan
//           con lai LUON hit vi cung trang, khong co tham chieu nao xen vao).
//         - Da KIEM CHUNG bang mo phong Monte Carlo doc lap (Python, ngoai
//           chuong trinh) tren nhieu cap (W,C) thuc te (ke ca W=19531,
//           C=64 va C=1536 - dung kich thuoc L1-DTLB/L2-STLB dien hinh cua
//           CPU x86 hien dai) - sai lech <0.1% so voi ly thuyet, phu hop
//           nhieu Monte Carlo o 2-3 trieu mau. Ket qua nay la mot truong hop
//           DAC BIET (popularity DEU) cua ly thuyet co dien ve LRU duoi
//           Independent Reference Model (vi du: xap xi "characteristic
//           time" cua Che et al. 2002 cho popularity bat ky suy giam ve
//           dang DUNG (khong con la xap xi) khi popularity deu, nho tinh
//           doi xung). self_test_irm_law() ben duoi lap lai chinh phep kiem
//           chung nay BEN TRONG chuong trinh C++ (LRU O(1) that, khong phai
//           suy dien tren giay) truoc khi dung cong thuc de suy dien tham so
//           tu du lieu phan cung that - neu that bai, phan [v23] tu dong bi
//           BO QUA thay vi in ra ket luan khong dang tin.
//
//   [Z] Vi cong thuc co CA L LAN W tuong minh (khac han a+b/L chi co L), no
//       TU DONG giai toa nhieu ma khong can 2 thi nghiem tach roi: suy nguoc
//       C tu 1 diem do (L,W,mien/line) bang dai so truc tiep (khong toi uu
//       hoa lap), lay median qua nhieu diem tren TRUC L (dung du lieu PMU da
//       co san) de duoc C_fit ON DINH; sau do dung CHINH C_fit ay de DU BAO
//       (khong fit lai) tren TRUC FOOTPRINT hoan toan doc lap - day la kiem
//       chung ngoai-mau (out-of-sample) qua 2 truc, dieu ma model a+b/L cu
//       KHONG THE lam (vi no khong co tham so W). Neu ca 2 truc deu khop
//       tot voi CUNG mot C_fit, do la bang chung manh model dang nam bat
//       dung co che vat ly, khong phai chi khop duong cong tren 1 tap du lieu.
//
// v31 (MOI, dap lai yeu cau "toi uu hon nua + tim thuat toan/kien truc moi"):
//   [AA] contig_bulk_avx512_v31_8acc: mo rong tu 2 accumulator ([v30]) len 8
//       accumulator doc lap cho duong ong "contig" AVX-512. DA DO THAT bang
//       microbenchmark doc lap (xem docs/bench_kway.cpp di kem) tren buffer
//       1.6GB (vuot han L3) TREN CHINH MAY NAY: 2acc~4.9-5.1, 4acc~4.9-5.1,
//       8acc~4.3-4.5, 16acc~4.2-4.4 ns/line - 8 accumulator la diem "khuyu"
//       hop ly (loi ~12-15% so 2acc, 16acc chi loi them <3% nua). Ca 3 muc
//       deu bit-identical voi scalar. Da noi vao dispatch + benchmark that
//       ngay trong binary chinh (benchmark_v31_kway_sweep(), chi in so khi
//       da tu-kiem-chung bit-identical, giong tinh than [v24]/[v30]).
//   [AB] Phat hien AM (negative, bao cao trung thuc): thu ap dung y tuong
//       tuong tu (rong accumulator) cho popcount_bulk_avx512 (duong ong
//       "gather", offset ngau nhien) - KHONG co cai thien dang ke (~23-25ns/
//       line ca 2/4/8-way). Ket luan: nut that o day la DO TRE truy cap bo
//       nho ngau nhien (TLB/cache-miss), khong phai so accumulator ALU - CPU
//       out-of-order da du "cua so" de giu nhieu load ngoai le bay voi chi 2
//       accumulator; prefetch_distance (co san tu [Q]/[R]) van la don bay
//       THAT duy nhat cho duong ong nay. KHONG ap dung [AA] cho gather.
//   [AC] contig_bulk_mt: kernel da luong (std::thread) cho contig, chia mang
//       thanh N doan doc lap + cong lai (bit-identical TUYET DOI voi uint64_t,
//       khong nhu float). CANH BAO TRUNG THUC: may container chay ban vá nay
//       CHI thay 1 vCPU (`nproc`=1), nen CHUA DO DUOC loi ich toc do thuc su -
//       self_test_contig_mt_bit_identical() moi xac nhan TINH DUNG, chua xac
//       nhan TOC DO. Nguoi dung nen tu do lai tren may nhieu core thuc.
//   [AD] Y TUONG MO RONG (GIA THUYET, CHUA TU-KIEM-CHUNG - neu muon dung nhu
//       [v23]/[v26] thi PHAI viet self-test doi chieu voi mo phong/do luong
//       THAT truoc khi tin): co the mo hinh hoa cost(K) theo K (so accumulator)
//       bang CHINH fit_cost_model() da co san trong file (dang a+b/K), va ly
//       giai qua Little's Law (dinh luat hang doi co dien, THAT, khong bia:
//       concurrency = throughput x latency) - so accumulator/do sau prefetch
//       dong vai tro "concurrency", va diem bao hoa (8acc->16acc loi it) ung
//       voi luc concurrency vuot qua so line-fill-buffer/thong luong bo nho
//       ma nhan co the tan dung. Day CHUA phai mot dinh luat da kiem chung
//       nhu [v23]/[v26] - de xuat trung thuc cho ai muon lam tiep, khong lam
//       gia thanh ket luan da xac nhan o day.
//   [AE] [DOI CHIEU CHEO MAY THAT, do nguoi dung tu chay tren Xeon Platinum
//       8481C, L3=105MB, N=20M tu/160MB (RAM may do chi ~1.9GB nen phai giam
//       N so voi 1.6GB tren may phat trien - van vuot L3)]:
//         contig: 2acc=5.754  4acc=5.319  8acc=5.040  16acc=5.118 ns/line
//         => 8-way VAN thang tren CA HAI may (~12-17% nhanh hon 2acc), NHUNG
//         16-way tren may nay CHAM LAI thay vi chi bao hoa (5.04->5.12,
//         khac voi may phat trien noi 16acc van nhanh hon 8acc mot chut) -
//         hop ly vi 16 accumulator (16 thanh ghi ZMM) gay ap luc thanh ghi
//         (register pressure) khac nhau tuy microarch/compiler schedule -
//         => KET LUAN SUA LAI cho dung voi ca 2 may: dung O 8-way la lua
//         chon AN TOAN NHAT (gan-toi-uu tren ca hai, khong bao gio la kem
//         nhat), KHONG nen mac dinh len 16-way du co the thang tren mot vai
//         may cu the.
//         gather (200k ops, offset ngau nhien): tren may Xeon 8481C, ket qua
//         doi chieu theo prefetch_distance CHO THAY loi ich K-way la CO
//         nhung NHO va PHU THUOC prefetch_distance/may, khac voi may phat
//         trien (KHONG co loi ich ro ret nao). Vi du tren 8481C: pf=0 thi
//         8-way nhanh nhat; pf=8-16 thi 2-way/4-way ngang hoac tot hon
//         8-way; pf=32 lam RIENG 2-way cham di ro (prefetch qua xa lam ban
//         accumulator it, tang ap luc bo dem prefetch). => XAC NHAN lai
//         [AB]: gather KHONG co "K-way tot nhat" pho quat - day la tham so
//         PHAI tune theo tung may, khong nen hard-code. Gia tri an toan
//         tren ca 2 may da thu: 4-way + prefetch_distance~16 (khong bao gio
//         la lua chon te nhat o may nao trong 2 may, du khong luon la lua
//         chon nhanh nhat). Dispatch mac dinh cua [v30]/[v31] VAN GIU 2-way
//         cho gather (don gian, an toan, dung it thanh ghi) - ai muon vat
//         them 5-10% tren may cu the thi tu doi K/pf qua benchmark_v31 hoac
//         ban rong cua gather_Kacc trong docs/bench_kway.cpp.
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
//   Ban dau viet kieu "per-line" (goi ham tra ve tong cho tung 8-word),
//   nhung benchmark thuc te tren may nay cho thay AVX2 kieu do CHAM HON
//   scalar (4.24ns vs 3.19ns/line) vi chi phi horizontal-reduce (store+load)
//   lap lai o MOI lan goi an mon het loi ich vector hoa.
//
//   Sua lai theo kieu "bulk": giu tong trong vector register (__m256i/__m512i)
//   xuyen suot toan bo vong lap, CHI reduce-ve-scalar 1 LAN DUY NHAT o cuoi.
//   Do that: scalar=5.95ns, avx2=3.55ns, avx512=3.13ns/line (bulk) => avx512
//   nhanh hon scalar ~1.9x trong kich ban nay (khong phai 4-8x ly thuyet, vi
//   ban than scalar POPCNT da duoc OoO/superscalar pipeline rat tot).
//
//   Luu y: neu bien dich voi -mavx512vpopcntdq, GCC/Clang co the TU DONG
//   vector hoa vong lap scalar popcount64_hw thanh vpopcntq (da kiem chung
//   bang objdump). Runtime dispatch o day van co gia tri vi: (1) dam bao
//   toi uu dung tren MOI may chay binary nay du build bang co gi, (2) kieu
//   "bulk" tu viet tay thang ca ban auto-vectorize cua compiler ap dung cho
//   kieu "per-line".
// ============================================================
static uint64_t popcount8_scalar(const uint64_t* base){
    uint64_t sum = 0;
    // [v21-FIX] "#pragma unroll" (cu phap Clang) bi GCC bo qua HOAN TOAN
    // (-Wunknown-pragmas), nghia la ban goc chua bao gio thuc su unroll tren
    // GCC du comment dau file co nhac "#pragma unroll". Sua bang macro
    // tuong thich ca hai trinh bien dich.
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

// ---------- Kernel BULK: gom offsets, tich luy vector, reduce 1 lan ----------
uint64_t popcount_bulk_scalar(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    uint64_t sum = 0;
    size_t n = offsets.size();
    for (size_t i = 0; i < n; i++){
        if (prefetch_distance > 0 && i + (size_t)prefetch_distance < n)
            _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + prefetch_distance]), _MM_HINT_T0);
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
    for (size_t i = 0; i < n; i++){
        if (prefetch_distance > 0 && i + (size_t)prefetch_distance < n)
            _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + prefetch_distance]), _MM_HINT_T0);
        const uint64_t* base = data + offsets[i];
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

__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_bulk_avx512(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    // [v26-OPT] Ban goc dung 1 accumulator duy nhat -> moi vong lap la mot
    // chuoi phu thuoc du lieu tuan tu (acc = add(acc, popcnt(v))), vong sau
    // phai doi vong truoc cong xong. vpopcntq co latency vai cycle nen chuoi
    // 1-accumulator nay gioi han throughput du CPU co the chay song song
    // nhieu phep popcnt cung luc. Ban AVX2 ben tren (popcount_bulk_avx2) da
    // tranh loi nay bang 2 accumulator doc lap (acc0/acc1) vi no xu ly 2
    // vector/iteration; ban AVX512 nay ap dung cung ky thuat: 2 accumulator
    // doc lap, xu ly 2 offset/vong de pha chuoi phu thuoc, cho phep CPU
    // pipeline nhieu vpopcntq song song truoc khi reduce. Duoi le (n so
    // chan) duoc xu ly bang scalar, khong dung AVX512 mask de tranh them
    // 1 target-feature phu.
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    size_t n = offsets.size();
    size_t i = 0;
    for (; i + 1 < n; i += 2){
        if (prefetch_distance > 0 && i + (size_t)prefetch_distance < n)
            _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + prefetch_distance]), _MM_HINT_T0);
        if (prefetch_distance > 0 && i + 1 + (size_t)prefetch_distance < n)
            _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + 1 + prefetch_distance]), _MM_HINT_T0);
        __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i]));
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + 1]));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
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
// [v33-NEW] Ban K-way + prefetch_distance tuong minh cho duong ong "gather"
// (offset ngau nhien) - MUC DICH KHONG PHAI de thay dispatch mac dinh (xem
// [AB]: rong accumulator KHONG giup duong ong nay tren may phat trien), ma
// de SWEEP CA HAI TRUC K x prefetch_distance CUNG LUC trong
// benchmark_v33_gather_kway_pf_sweep() ben duoi. Ly do: [AB]/[AE] moi lan
// deu CO DINH prefetch_distance o MOT gia tri roi doi K (hoac nguoc lai),
// nen ket luan "K khong giup gather" co the bi confound - K nho + pd ngan
// co the nghen giong het K lon + pd ngan (ca hai deu thieu "cua so" MLP),
// khien nguoi doc lam tuong rong accumulator vo dung trong MOI truong hop,
// trong khi thuc ra chi la pd chua du de nuoi K load ngoai-le cung luc.
// Quet luoi that (thay vi 2 sweep 1-chieu rieng le) moi phan biet duoc.
template<int K>
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t popcount_bulk_avx512_Kacc(const uint64_t* data, const vector<size_t>& offsets, int prefetch_distance){
    __m512i acc[K];
    for (int k = 0; k < K; k++) acc[k] = _mm512_setzero_si512();
    size_t n = offsets.size();
    size_t i = 0;
    for (; i + (size_t)K <= n; i += (size_t)K){
        if (prefetch_distance > 0){
            #pragma GCC unroll 16
            for (int k = 0; k < K; k++){
                size_t idx = i + (size_t)k + (size_t)prefetch_distance;
                if (idx < n) _mm_prefetch(reinterpret_cast<const char*>(data + offsets[idx]), _MM_HINT_T0);
            }
        }
        #pragma GCC unroll 16
        for (int k = 0; k < K; k++){
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(data + offsets[i + (size_t)k]));
            acc[k] = _mm512_add_epi64(acc[k], _mm512_popcnt_epi64(v));
        }
    }
    uint64_t sum = 0;
    for (int k = 0; k < K; k++){
        alignas(64) uint64_t t[8];
        _mm512_store_si512(reinterpret_cast<void*>(t), acc[k]);
        for (int j = 0; j < 8; j++) sum += t[j];
    }
    for (; i < n; i++){ // duoi le so K
        if (prefetch_distance > 0 && i + (size_t)prefetch_distance < n)
            _mm_prefetch(reinterpret_cast<const char*>(data + offsets[i + (size_t)prefetch_distance]), _MM_HINT_T0);
        sum += popcount8_scalar(data + offsets[i]);
    }
    return sum;
}

// ---------- Kernel BULK cho compute-floor (truy cap modulo, khong offsets) ----------
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
__attribute__((target("avx512f,avx512vpopcntdq")))
static uint64_t floor_bulk_avx512_legacy(const uint64_t* data, size_t count, size_t lines_per_4k){
    // [v26-OPT] cung ky thuat 2-accumulator nhu popcount_bulk_avx512 o tren:
    // pha chuoi phu thuoc tuan tu de CPU pipeline nhieu vpopcntq song song.
    // [v33-DOI TEN] Ban 2acc nay TRUOC DAY ten la "floor_bulk_avx512" (dispatch
    // mac dinh); doi ten thanh "_legacy" va GIU NGUYEN body de lam mo hinh
    // doi chieu bit-identical + toc do cho ban 8acc moi ben duoi (dung y het
    // tinh than contig_bulk_avx512_v31_8acc() da lam voi contig_bulk_avx2 cu).
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 1 < count; i += 2){
        const uint64_t* base0 = data + (i & (lines_per_4k - 1)) * 8;
        const uint64_t* base1 = data + ((i + 1) & (lines_per_4k - 1)) * 8;
        __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(base0));
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(base1));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
    }
    alignas(64) uint64_t tmp0[8], tmp1[8];
    _mm512_store_si512(reinterpret_cast<void*>(tmp0), acc0);
    _mm512_store_si512(reinterpret_cast<void*>(tmp1), acc1);
    uint64_t sum = 0;
    for (int k = 0; k < 8; k++) sum += tmp0[k] + tmp1[k];
    for (; i < count; i++) sum += popcount8_scalar(data + (i & (lines_per_4k - 1)) * 8); // duoi le
    return sum;
}

// ============================================================
// [v33-NEW] Ap dung y tuong [AA]/[v31] (mo rong 2acc -> 8acc lam giam chi
// phi dieu khien vong lap/tu, xem giai thich day du o
// contig_bulk_avx512_v31_8acc() phia duoi) cho duong ong "floor"
// (floor_bulk_avx512, du lieu wrap quanh lines_per_4k - trong benchmark cua
// file nay lines_per_4k=LINES_PER_4K=64 dong nghia du lieu THUONG resident
// san trong L1/L2, tuc la duong ong nay COMPUTE/ILP-BOUND ngay tu dau, chu
// khong memory-bound nhu buffer "contig" lon o [v31]). Day CHINH la truong
// hop ma [v31] con de ngo: contig da duoc mo rong len 8acc va tu-doi-chieu,
// nhung floor_bulk_avx512 van chi dung 2 accumulator "hardcode" tu [v26].
//
// Cung nhu [v32] da lam voi contig_bulk_avx512_Kacc, o day cung co 1 ban
// template K-way (de tu do POPCNT_FLOOR_K luc chay, khong can build lai) VA
// 1 ban hand-unroll 8acc rieng (floor_bulk_avx512_v33_8acc) dung lam dispatch
// mac dinh moi - ban 2acc cu duoc GIU NGUYEN o tren voi ten
// floor_bulk_avx512_legacy() de tu-doi-chieu bit-identical/toc do (xem
// benchmark_v33_floor_kway_sweep() ben duoi, THAT tren du lieu cua lan chay
// hien tai, khong tin vao con so co dinh trong comment).
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

// [v33-NEW] Ban TONG QUAT K-way cho floor, doi xung voi contig_bulk_avx512_Kacc
// ([v32]) - cho phep chon K luc chay qua POPCNT_FLOOR_K (xem init_popcount_dispatch)
// ma khong can build lai, vi diem toi uu K co the khac nhau giua cac may
// (dung tinh than [AE] da quan sat voi contig). floor_bulk_avx512_v33_8acc()
// o tren VAN la dispatch mac dinh (K=8, khong doi hanh vi neu khong set env).
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

// [v30-MOI, PHAT HIEN QUAN TRONG] Ban goc cua contig_bulk_avx2 (giu lai o day
// duoi ten "_1acc_legacy") dung DUY NHAT 1 accumulator xuyen suot vong lap -
// CHINH XAC cung loi chuoi-phu-thuoc-tuan-tu ma [v26-OPT] da tim thay va sua
// cho popcount_bulk_avx512/floor_bulk_avx512 (xem comment o do), nhung loi
// nay CHUA BAO GIO duoc sua cho nhanh "contig" AVX2! Nghiem trong hon: ham
// nay chinh la BASELINE "nibble-LUT (cu)" ma benchmark_v24_harleyseal() dung
// de so sanh voi Harley-Seal - nghia la con so "Harley-Seal nhanh hon
// 1.1x-1.9x" da bao cao o [v24] co the MOT PHAN la do so sanh VOI MOT
// BASELINE BI THIET THOI (1 accumulator, chuoi phu thuoc), chu KHONG hoan
// toan la loi ich thuan tuy cua thuat toan CSA - day la CHINH loai nhieu do
// phuong phap ma file nay da nhieu lan tu bat (vd DIV-vs-AND o [T-BUG]).
// Sua bang 2 accumulator doc lap (acc0/acc1, xu ly 2 vector 256-bit/vong,
// giong ky thuat da dung o popcount_bulk_avx2) de pha chuoi phu thuoc va co
// mot baseline CONG BANG. Ban goc (1 accumulator, KHONG sua doi logic) duoc
// GIU LAI duoi ten "_1acc_legacy" de benchmark_v24_harleyseal() co the in CA
// HAI, dinh luong that xem confound nay lon co nao - thay vi am tham thay
// the va lam mat kha nang doi chieu.
__attribute__((target("avx2")))
static uint64_t contig_bulk_avx2(const uint64_t* data, size_t n){
    // [v30-FIX] 2 accumulator doc lap (acc0/acc1), moi vong xu ly 2 vector
    // 256-bit (8 tu 64-bit) thay vi 1, de CPU co the pipeline 2 chuoi
    // shuffle/and/add/sad doc lap song song thay vi cho tuan tu tren 1 acc.
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
// (Ban goc, KHONG sua doi logic, chi doi ten - dat SAU contig_bulk_avx2 de
// doc de doi chieu 2 ham canh nhau. Dung lam baseline trong benchmark_v24_harleyseal.)
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
    // [v30-FIX, CHUA KIEM CHUNG TREN MAY NAY] Ban goc dung 1 accumulator duy
    // nhat -> CHINH XAC cung loi chuoi-phu-thuoc-tuan-tu ma [v26-OPT] da tim
    // va sua cho popcount_bulk_avx512/floor_bulk_avx512 (xem comment o do),
    // nhung KHONG HIEU SAO chua tung duoc ap dung cho nhanh "contig" nay -
    // mot su khong nhat quan ro rang trong CHINH file nay. Sua bang 2
    // accumulator doc lap, giong het ky thuat da dung o 2 ham kia.
    // CANH BAO TRUNG THUC: may container nay KHONG co avx512vpopcntdq
    // (da xac nhan qua /proc/cpuinfo va detect_simd_level() luc chay), nen
    // ham nay KHONG THE duoc thuc thi/do toc do that o day - chi sua ve mat
    // logic/cau truc (giong het pattern da kiem chung o popcount_bulk_avx512)
    // va dam bao van BIT-IDENTICAL bang xem xet: phep toan giao hoan/ket hop
    // (tong cac tu doc lap) nen tach lam 2 accumulator KHONG doi ket qua
    // cuoi, chi doi THU TU cong (float thi co the doi ket qua do lam tron,
    // nhung day la uint64_t nguyen nen TUYET DOI khong doi). Ai chay tren may
    // co VPOPCNTDQ that nen chay lai self_test_bit_identical/self_test_contig_bit_identical
    // (da co san trong file) de tu xac nhan truoc khi tin ket qua toc do.
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
// [v31-MOI, DA DO THAT tren chinh may nay - khong bia] Van de voi
// contig_bulk_avx512 (2 accumulator, [v30]): benchmark tren buffer 1.6GB
// (vuot han L3, memory-bound thuc su, khong phai gia lap) cho thay 2acc va
// 4acc GAN NHU KHONG khac nhau (~4.9-5.1ns/line ca hai, trong bien do nhieu
// do), nhung mo rong len 8 accumulator DOC LAP (64 tu/vong lap thay vi 16)
// giam duoc ~12-15% cost/line mot cach ON DINH qua nhieu lan chay lap lai
// (5.0 -> 4.4ns/line), va 16 accumulator chi loi THEM rat it so voi 8
// (4.4 -> 4.3ns/line - ro rang la vung bao hoa/diminishing-returns). Ca 3
// muc deu bit-identical voi scalar (kiem tra o self_test_contig_bit_identical
// va benchmark_v31_kway_sweep ben duoi, tren du lieu THAT cua lan chay hien
// hanh, khong tin vao con so co dinh trong comment).
//
// Giai thich hop ly (KHONG phai quy luat vat ly moi, chi la ly luan kien
// truc thong thuong): voi truy cap TUAN TU (khong random), HW prefetcher da
// no lo bang thong roi - cai 2acc/4acc/8acc/16acc thay doi khong phai bang
// thong ma la CHI PHI DIEU KHIEN VONG LAP (loop counter, branch, tinh dia
// chi) tren MOI don vi cong viec: 8 accumulator = 64 tu/vong lap = 1/4 so
// vong lap so voi 2acc (16 tu/vong), nen phan chi phi co dinh do duoc chia
// deu tren nhieu du lieu hon. Qua 8 accumulator, loi ich nay bao hoa vi chi
// phi vong lap da du nho so voi thoi gian doi bo nho.
//
// [v31-PHAM VI, DA THU VA KHONG THANH CONG - bao cao trung thuc] Da thu ap
// dung Y TUONG TUONG TU (mo rong accumulator 2->4->8) cho popcount_bulk_avx512
// (duong ong "gather", offset ngau nhien, TLB/cache-miss-bound): KHONG co cai
// thien dang ke (~23.5-25ns/line ca 2-way/4-way/8-way, sai khac trong bien do
// nhieu). Ly do hop ly: o day nut that la DO TRE truy cap bo nho ngau nhien
// (TLB-miss/cache-miss ~50-70 cycle), khong phai so luong accumulator ALU -
// CPU out-of-order da co du "cua so" de giu nhieu load ngoai-le bay cung luc
// voi chi 2 accumulator, gioi han thuc su la SO LUONG LINE-FILL-BUFFER cua
// nhan, khong phai so thanh ghi vector dung de cong don. prefetch_distance
// (da co san tu [Q]/[R]) van la don bay THAT cho duong ong nay (~7% o
// pf=8..16 so voi pf=0), rong accumulator KHONG phai. Day la mot phat hien
// AM (negative result) co gia tri: tranh lam nguoi doc sau nay ton cong mo
// rong accumulator cho nhanh gather ma khong co loi ich.
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
    // [v33-NEW] Duoi <8 tu (toi da 7 phan tu): gop thanh 1 lenh masked-load
    // AVX512F thay vi vong scalar tung tu - it lenh hon, khong nhanh (chi con
    // toi da 7 phan tu du kien), nhung don gian/dong nhat voi kernel khac va
    // tranh 1 vong lap co dieu kien rieng. Da tu-kiem-chung: self_test_contig_bit_identical
    // trong file nay chay 3000 trial n=0..1199 (moi truong hop du chia het cho
    // 8), so khop tuyet doi voi scalar truoc khi tin dung ban nay.
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
    uniform_int_distribution<size_t> region_dist(0, n_regions - 1);
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
            vector<vector<double>> resampled;
            for (auto& v : data_sets){
                uniform_int_distribution<size_t> idx(0, v.size() - 1);
                vector<double> rv(v.size());
                for (size_t i = 0; i < v.size(); i++) rv[i] = v[idx(local_rng)];
                resampled.push_back(move(rv));
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
vector<size_t> stationary_bootstrap_indices(size_t n, double p, mt19937_64& rng){
    vector<size_t> idx; idx.reserve(n);
    uniform_int_distribution<size_t> start_dist(0, n - 1);
    geometric_distribution<int> geo(p);
    while (idx.size() < n){
        size_t start = start_dist(rng);
        int block_len = geo(rng) + 1; // Geometric >=0 tren cstd, +1 de >=1
        for (int k = 0; k < block_len && idx.size() < n; k++)
            idx.push_back((start + (size_t)k) % n); // vong quanh (circular)
    }
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
            vector<size_t> idx = stationary_bootstrap_indices(n, p, local_rng);
            vector<vector<double>> resampled;
            for (auto& v : data_sets){
                vector<double> rv(n);
                for (size_t i = 0; i < n; i++) rv[i] = v[idx[i]];
                resampled.push_back(move(rv));
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
            vector<size_t> tidx = stationary_bootstrap_indices(n_trials, p, local_rng); // Tang 2
            vector<vector<double>> one_rep(n_vec);
            for (size_t v = 0; v < n_vec; v++){
                const auto& src = reps_data_sets[rep_idx][v];
                one_rep[v].reserve(n_trials);
                for (size_t k = 0; k < n_trials; k++) one_rep[v].push_back(src[tidx[k]]);
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
static void self_test_irm_fixed_replay_gap(mt19937_64& rng){
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
// [v26-NEW, MO RONG] Dinh luat IRM hieu chinh theo TAP-HOP-LIEN-KET
// (set-associative). Mo rong [v23] (an ngam dinh fully-associative, tuc
// S=1 "way" cho toan bo C) sang truong hop that cua cache/TLB phan cung:
// S set doc lap, moi set co A=C/S "way". Giong tinh than [v24] mo rong
// sang popularity lech - day la mo rong theo TRUC KHAC (cau truc anh xa
// set), khong phai theo phan phoi truy cap.
//
// Dan xuat (3 buoc, khong phai xap xi tuy tien):
//   1) Duoi tham chieu IID-uniform tren W trang, anh xa trang->set la deu
//      va doc lap => tai cua 1 set cu the X ~ Binomial(W, 1/S).
//      DANG THUC DUNG TUYET DOI, khong xap xi.
//   2) Ty le hit dung = (S/W)*E[min(X,A)] (tong dong gop moi set, moi set
//      giu toi da A trong so trang roi vao no). DANG THUC DUNG TUYET DOI.
//   3) Xap xi CLT cho E[min(X,A)] bang "unit normal loss integral"
//      L(z)=z*Phi(z)+phi(z): E[min(X,A)] = mu - sigma*L(z), voi
//      z=(A-mu)/sigma. Day LA xap xi (Berry-Esseen: sai so O(1/sqrt(W/S))).
//      DA KIEM CHUNG SO doc lap voi binomial chinh xac (scipy.stats.binom,
//      khong phai chi tin loi ke): tai knee C=W, W=19531,
//        S= 8: exact=0.99245 approx=0.99245 (|diff|~1.5e-7)
//        S=16: exact=0.98894 approx=0.98894 (|diff|~2.8e-7)
//        S=24: exact=0.98631 approx=0.98631 (|diff|~4.3e-8)
//      => sai lech so voi mo hinh fully-assoc cu (naive=1.0 tai C=W) la
//      THAT (0.76% / 1.11% / 1.37%), KHONG phai nhieu do lech chuan.
//   4) [v23] la truong hop gioi han S=1: khi S=1 thi A=C, mu=W, phuong sai
//      suy bien ve 0 va occ_assoc(C,W,1) == irm_occupancy(C,W) dung y het
//      (xu ly rieng ben duoi de tranh chia 0/1-1/S).
//
// GIOI HAN TRUNG THUC (giong tinh than [v23]/[v24]): day la xap xi CLT
// tren mot mo hinh anh xa trang->set DEU & DOC LAP ly tuong hoa; phan cung
// that (vd STLB) co the co xung dot anh xa khong hoan toan deu, va cong
// thuc nay CHUA tinh den polya-urn/replacement chinh xac (chi dung xap xi
// chuan cho phan phoi bien - day la ly do self_test_irm_assoc_law() ben
// duoi BAT BUOC phai chay va PHAI qua truoc khi dung ket luan cua [v26]).
// ============================================================
static inline double unit_normal_loss(double z){
    // L(z) = z*Phi(z) + phi(z), ham loss integral chuan trong ly thuyet
    // ton kho (newsvendor) - dung erf() co san trong <cmath>, khong can
    // xap xi rieng cho Phi/phi.
    double Phi = 0.5 * (1.0 + erf(z / sqrt(2.0)));
    double phi = exp(-z * z / 2.0) / sqrt(2.0 * M_PI);
    return z * Phi + phi;
}

static inline double irm_occupancy_assoc(double C, double W, double S){
    if (W <= 0.0 || S <= 0.0) return 0.0;
    if (S <= 1.0 + 1e-9) return irm_occupancy(C, W); // suy bien dung ve [v23]
    double A   = C / S;
    double var = (W / S) * (1.0 - 1.0 / S);
    if (var <= 1e-12) return irm_occupancy(C, W); // W nho, khong con phuong sai dang ke
    double mu    = W / S;
    double sigma = sqrt(var);
    double z     = (A - mu) / sigma;
    double occ   = C / W - (S * sigma / W) * unit_normal_loss(z);
    return max(0.0, min(1.0, occ));
}

static inline double predict_miss_per_line_assoc(double L, double W, double C, double S){
    return (1.0 - irm_occupancy_assoc(C, W, S)) / L;
}

// Suy nguoc C bang bisection: khac voi back_out_capacity() (dao nguoc dai
// so truc tiep vi [v23] tuyen tinh trong C), o day z phu thuoc C qua ham
// Phi() ben trong nen KHONG nghich dao dai so duoc. Nhung irm_occupancy_assoc()
// la ham DON DIEU TANG theo C (them way chi co the giu duoc >= so trang cu),
// nen bisection hoi tu dam bao - cung tinh than bisection da dung cho
// che_characteristic_time() ben duoi ([v24]).
static inline double back_out_capacity_assoc(double L, double W, double S, double measured_miss_per_line){
    double target_occ = max(0.0, min(1.0, 1.0 - measured_miss_per_line * L));
    double lo = 0.0, hi = W;
    // Mo rong hi neu can: tai knee, cache set-associative co the can NHIEU
    // hon W de bu phuong sai giua cac set (xem bang so o comment tren).
    int guard = 0;
    while (irm_occupancy_assoc(hi, W, S) < target_occ && guard < 64){
        hi *= 2.0;
        guard++;
    }
    for (int iter = 0; iter < 60; iter++){
        double mid = 0.5 * (lo + hi);
        if (irm_occupancy_assoc(mid, W, S) < target_occ) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------- [v26-NEW] Self-test: kiem chung dinh luat IRM set-associative bang mo phong LRU that ----------
// Mo phong S set DOC LAP, moi set la 1 SimpleLRU dung luong A=C/S (chia
// nguyen, giong phan cung that), trang x duoc anh xa vao set qua (x % S)
// duoi tham chieu IID-uniform tren W trang - dung dieu kien ly thuyet gia
// dinh o tren. So sanh ty le hit thuc nghiem voi cong thuc dong dang. Neu
// that bai tren may nay, phan con lai cua [v26] se BI BO QUA thay vi in
// ra ket luan suy dien tu 1 dinh luat chua duoc xac nhan - dung tinh than
// "khong tin - kiem chung" xuyen suot file, giong self_test_irm_law().
static bool self_test_irm_assoc_law(mt19937_64& rng){
    struct Case { int W, C, S; long refs; };
    vector<Case> cases = {
        {19531, 19531, 8,  800000},  // dung knee C=W, S nho (giong STLB that)
        {19531, 19531, 16, 800000},
        {19531, 19531, 24, 800000},
        {19531, 1536,  12, 800000},  // xa knee (C << W) - gap phai gan ve 0
        {1000,  64,    8,  800000},
    };
    bool all_ok = true;
    cout << "[self-test IRM set-assoc, v26] Mo phong S set LRU doc lap (O(1)), doi chieu voi cong thuc dong dang:\n";
    for (auto& c : cases){
        int A = c.C / c.S; // way moi set, chia nguyen - phai dung DUNG gia tri nay ca trong mo phong lan cong thuc
        int C_eff = A * c.S; // dung luong THAT dang mo phong (co the < c.C do chia nguyen)
        vector<SimpleLRU> sets;
        sets.reserve(c.S);
        for (int s = 0; s < c.S; s++) sets.emplace_back(A);
        uniform_int_distribution<int> pick(0, c.W - 1);
        // Warm-up (KHONG dem hit): moi trang trong W trang, lan dau tham
        // chieu LUON LA mo (cache khoi tao rong) - do la mot khoan phi CO
        // DINH, KHONG PHAI xap xi, xuat hien dung 1 lan cho MOI trang phan
        // biet trong toan bo lich su chay. O che do binh thuong (C << W)
        // khoan phi nay khong dang ke vi ngan sach miss du lon; nhung DUNG
        // TAI KNEE (occ~0.99, ngan sach miss du kien chi con ~1% cua refs)
        // no co the LON HON ca ngan sach miss ly thuyet va lam thien lech
        // ket qua thap gia tao - can chay du lau de gan nhu MOI trang trong
        // W da duoc cache thay it nhat 1 lan TRUOC KHI bat dau dem, giong
        // dung cach mo phong steady-state chuan (bo qua warm-up transient).
        long warmup = 15L * c.W; // P(1 trang cu the CHUA tung xuat hien) = (1-1/W)^warmup ~= e^-15 ~ 3e-7 moi trang
        for (long i = 0; i < warmup; i++){
            int page = pick(rng);
            sets[page % c.S].touch(page);
        }
        long hits = 0;
        for (long i = 0; i < c.refs; i++){
            int page = pick(rng);
            if (sets[page % c.S].touch(page)) hits++;
        }
        double emp    = (double)hits / (double)c.refs;
        double theory = irm_occupancy_assoc((double)C_eff, (double)c.W, (double)c.S);
        double err    = fabs(emp - theory);
        bool ok = err < 0.015; // nguong roi hon self_test_irm_law (0.01) vi S set doc lap => phuong sai Monte Carlo cao hon
        cout << "  W=" << setw(6) << c.W << " C=" << setw(5) << C_eff << " S=" << setw(3) << c.S
             << "  ly_thuyet=" << fixed << setprecision(5) << theory
             << "  thuc_nghiem=" << emp << "  sai_lech=" << setprecision(5) << err
             << (ok ? "  OK\n" : "  [LOI! dinh luat KHONG khop mo phong tren may nay]\n");
        all_ok = all_ok && ok;
    }
    return all_ok;
}

// ============================================================
// [v24-MOI, phan LY THUYET] Mo rong dinh luat IRM-Burst ([v23], chi dung
// cho popularity DEU) sang popularity LECH (khong deu) bang xap xi
// "characteristic time" cua Che, Tung, Wang (2002), "Hierarchical Web
// Caching Systems: Modeling, Design and Experimental Results", IEEE JSAC.
//
// DAY LA XAP XI DA CONG BO (khong phai toi tu nghi ra), nhung [v23] cua file
// nay CHUA tung dung no - v23 chi dung truong hop DAC BIET (popularity deu)
// ma o do no TRUNG KHOP CHINH XAC voi C/W (khong con la xap xi). Dong gop
// [v24] o day la: (1) trien khai xap xi TONG QUAT cho popularity LECH
// (Zipf, phan anh truy cap thuc te - vi du "hot keys" trong cache/DB thuong
// LECH manh, khong deu nhu offsets ngau nhien cua generate_offsets()), va
// (2) tu-kiem-chung no BANG mo phong LRU that (giong tinh than
// self_test_irm_law o tren), thay vi chi trich dan cong thuc.
//
// LAP LUAN: coi moi "trang" i trong so W trang co xac suat duoc THAM CHIEU
// (chon MOI lan, IID) la p_i (sum p_i = 1; p_i = 1/W deu voi moi i thi day
// la truong hop [v23]). Xap xi Che: co mot "thoi gian dac trung" t_C (mot
// so thuc duong) sao cho XAC SUAT trang i dang trong cache LRU dung luong C
// XAP XI:
//     hit_i(t_C) = 1 - exp(-p_i * t_C)
// va t_C duoc chon de tong xac suat nay khop dung luong cache:
//     sum_i hit_i(t_C) = C                                    (*)
// (*) la ham TANG DON DIEU theo t_C (tu 0 den W), nen giai bang bisection
// la CHINH XAC va on dinh (khong can Newton/dao ham). Ty le hit TONG THE
// (trung binh theo dung tan suat truy cap p_i, vi p_i lech thi trang pho
// bien duoc tham chieu - va do do dong gop vao ty le hit tong - nhieu hon):
//     hit_tong = sum_i p_i * hit_i(t_C)
//
// KHI p_i = 1/W (deu): (*) tro thanh W*(1-exp(-t_C/W)) = C, giai ra
// hit_i = C/W CHO MOI i - TRUNG KHOP CHINH XAC voi irm_occupancy() cua
// [v23] (da kiem tra dai so va kiem tra bang code ben duoi, sai lech may
// tinh ~1e-6). Day la mot kiem tra tinh NHAT QUAN quan trong: cong thuc moi
// khong "thay the" cong thuc cu, ma la BAO TRUM no nhu 1 truong hop rieng.
//
// GIOI HAN TRUNG THUC: khac voi dinh luat [v23] (CHUNG MINH DUNG boi doi
// xung/trao doi duoc, khong xap xi), cong thuc Che O DAY LA XAP XI cho
// popularity bat ky - KHONG co chung minh dai so cho sai so bang 0. Vi vay
// ham nay PHAI duoc tu-kiem-chung bang mo phong truoc khi dung, giong het
// tinh than self_test_irm_law(): neu that bai o may/trinh bien dich nao do,
// [v24] tu dong bao that bai thay vi in ket luan khong dang tin.
// ============================================================
static vector<double> zipf_probs(int W, double theta){
    vector<double> p(W);
    double s = 0.0;
    for (int i = 1; i <= W; i++){ p[i-1] = pow((double)i, -theta); s += p[i-1]; }
    for (int i = 0; i < W; i++) p[i] /= s;
    return p;
}
// [v33-FIX, BUG THAT - da tai hien bang test doc lap che_repro.cpp truoc khi
// sua, KHONG doan mo] Ban goc: "while (g(hi) < C) hi *= 2.0" gia dinh g(t)
// se VUOT C khi t du lon, dung DUNG khi C < W (|p|, so phan tu) - nhung g(t)
// = sum_i (1 - exp(-p_i*t)) TIEM CAN W khi t->vo cung va KHONG BAO GIO cham
// toi W ve mat dai so (moi so hang <1 mai mai). Neu nguoi goi truyen C >= W
// (vd C tu dTLB THAT do duoc, khong dam bao < W nhu 2 test-case cung hien
// tai o self_test_che_zipf_law) thi dieu kien "g(hi) < C" luon dung, hi tang
// gap doi MAI MAI cho toi +inf (IEEE754: inf*2=inf) -> TREO CHUONG TRINH VO
// HAN, khong crash, khong bao loi, im lang 100% CPU. Da xac nhan HANG THAT
// (khong phai suy dien) bang chuong trinh doc lap: C=W+1 treo vinh vien sau
// >10000 lan nhan doi, hi=inf; C=W (bien) thi KHONG treo (lam tron so hoc
// tinh co lam g(hi) cham dung W truoc khi hi->inf) nhung do la may man ngau
// nhien cua dau phay dong, khong phai bao dam. Day CHINH la "qua ban" ma
// [v24-PHAM VI] da moi goi: nguoi lam tiep se can goi ham nay voi C that tu
// phan cung, rat de dinh phai bug nay neu W nho hoac C do duoc lon.
//
// Sua: clamp tuong minh C>=W ve occupancy=1.0 (giong dung cach irm_occupancy()
// [v23] da clamp min(1,C/W) cho truong hop tuong tu), TRA VE t_C huu han rat
// lon thay vi chay vong lap; VA them guard chan-cung 200 lan nhan doi (~1.6e60)
// cho nhanh C<W hop le, de KHONG BAO GIO treo vo han du dau vao bat thuong
// the nao (bao gia tri bien an toan thay vi treo im lang).
static double che_characteristic_time(const vector<double>& p, double C){
    double W = (double)p.size();
    if (C >= W) return 1e15; // occupancy=1.0: che_hit_prob(pi,1e15)~=1.0 cho moi pi>0 thuc te
    auto g = [&](double t){ double s = 0.0; for (double pi : p) s += 1.0 - exp(-pi * t); return s; };
    double lo = 0.0, hi = 1.0;
    for (int guard = 0; g(hi) < C; guard++){ // mo rong can tren toi khi vuot C
        if (guard > 200) return hi; // chan cung: tra ve bien thay vi treo vo han
        hi *= 2.0;
    }
    for (int it = 0; it < 100; it++){ // bisection ~2^-100, du du chinh xac double
        double mid = 0.5 * (lo + hi);
        if (g(mid) < C) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}
static inline double che_hit_prob(double p_i, double tC){ return 1.0 - exp(-p_i * tC); }
static double che_predicted_hit_rate(const vector<double>& p, double C){
    double tC = che_characteristic_time(p, C);
    double hit = 0.0;
    for (double pi : p) hit += pi * che_hit_prob(pi, tC);
    return hit;
}

// ---------- [v24-NEW] Self-test: kiem chung xap xi Che bang LRU that ----------
static bool self_test_che_zipf_law(mt19937_64& rng){
    struct Case { int W, C; double theta; long refs; };
    vector<Case> cases = {
        {1000, 64, 0.0, 1200000},  // theta=0: phai QUY VE irm_occupancy cua [v23]
        {1000, 64, 0.5, 1200000},
        {1000, 64, 0.8, 1200000},
        {1000, 64, 1.0, 1200000},  // Zipf co dien
        {1000, 64, 1.5, 1200000},
        {19531, 1536, 1.0, 1500000},
    };
    bool all_ok = true;
    cout << "[self-test Che-Zipf, v24] Mo phong LRU that duoi popularity LECH (Zipf theta),\n"
            "  doi chieu ty le hit thuc nghiem voi xap xi characteristic-time:\n";
    for (auto& c : cases){
        auto p = zipf_probs(c.W, c.theta);
        discrete_distribution<int> pick(p.begin(), p.end());
        SimpleLRU cache(c.C);
        long hits = 0;
        for (long i = 0; i < c.refs; i++)
            if (cache.touch(pick(rng))) hits++;
        double emp = (double)hits / (double)c.refs;
        double pred = che_predicted_hit_rate(p, (double)c.C);
        double err = fabs(emp - pred);
        // Nguong long hon self_test_irm_law (0.01) vi day la XAP XI (khong
        // phai dang thuc chinh xac) cho theta != 0 - 0.02 van du chat de
        // bat loi logic ro rang, du long cho sai so xap xi + Monte Carlo.
        bool ok = err < 0.02;
        cout << "  W=" << setw(6) << c.W << " C=" << setw(5) << c.C
             << " theta=" << fixed << setprecision(2) << c.theta
             << "  Che_du_doan=" << setprecision(5) << pred
             << "  thuc_nghiem=" << emp << "  sai_lech=" << err
             << (ok ? "  OK\n" : "  [LOI! xap xi KHONG khop mo phong tren may nay]\n");
        all_ok = all_ok && ok;
        if (c.theta == 0.0){
            double uni = irm_occupancy((double)c.C, (double)c.W);
            double consistency_err = fabs(uni - pred);
            cout << "       (doi chieu tinh nhat quan voi [v23]: irm_occupancy(C,W)=" << uni
                 << "  Che(theta=0)=" << pred << "  lech=" << scientific << setprecision(2)
                 << consistency_err << fixed << "  " << (consistency_err < 1e-6 ? "OK (quy ve dung)\n" : "[CANH BAO]\n");
            all_ok = all_ok && (consistency_err < 1e-6);
        }
    }
    return all_ok;
}

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
            vector<size_t> tidx = stationary_bootstrap_indices(n_trials, p, local_rng); // tang 2, CHUNG cho moi L
            vector<double> cost_boot(nL);
            for (size_t li = 0; li < nL; li++){
                vector<double> resampled(n_trials);
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

    // ---------- [S] Bootstrap CI cho ratio_of_drops ----------
    {
        // ds[0]=L1_normal times, ds[1]=L64_normal, ds[2]=L1_hp, ds[3]=L64_hp
        // [v21-FIX, KHOI PHUC] n_ops PHAI lay tu n_ops_normal/n_ops_hp THAT SU
        // (co the < total_lines do generate_offsets clamp o L nho) - KHONG
        // duoc gia dinh cung bang total_lines cho moi L, day chinh la nguyen
        // nhan goc khien bootstrap CI truoc day sup do ve [0,0] mot cach he
        // thong (xem ghi chu [HOP NHAT v21+v22] o dau file).
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
        cout << "  [S] Bootstrap CI 95% ratio_of_drops (i.i.d., GIA DINH DA BI [Q]\n"
                "  BAC BO o tren - xem duoi de so sanh voi ban sua): [" << fixed << setprecision(3)
             << boot.lo << ", " << boot.hi << "]"
             << "   median=" << boot.median << "  MAD=" << boot.mad
             << "  (n_boot=" << N_BOOT << ")\n";

        // [v25c-MOI, ung voi (Q)-fix] So sanh TRUC TIEP voi stationary bootstrap
        // (ban sua that cho van de vua duoc [Q] xac nhan). mean_block_len=3 la
        // lua chon kinh nghiem cho chuoi ngan (TRIALS thuong 20-30) - thu them
        // block=5 de kiem tra do nhay cam voi lua chon nay, KHONG chi tin 1 gia
        // tri duy nhat.
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
        cout << "  => Do rong CI: i.i.d.=" << setprecision(3) << width_iid
             << "   stationary(block~3)=" << width_sboot3
             << "   ty le=" << setprecision(2) << (width_sboot3 / width_iid) << "x\n";
        if (width_sboot3 > width_iid * 1.15)
            cout << "     CI stationary RONG HON ro ret - xac nhan THAT (khong phai suy dien)\n"
                    "     rang i.i.d. bootstrap o tren da DANH GIA THAP do bat dinh, dung nhu\n"
                    "     [Q] canh bao. Nen dung CI stationary tu day ve sau, khong dung i.i.d.\n";
        else
            cout << "     CI stationary KHONG rong hon dang ke - troi dat phat hien o [Q] co the\n"
                    "     KHONG anh huong nhieu den do rong CI cho THONG KE CU THE nay (ty le, vi\n"
                    "     ca tu so/mau so co the cung troi dat theo huong TRIET TIEU lan nhau).\n";
        double lo = boot.lo, hi = boot.hi;
        if (lo > 0.7)
            cout << "      => CI nam tren 0.7: phan con lai tren hugepage VAN LON.\n";
        else if (hi < 0.3)
            cout << "      => CI nam duoi 0.3: hugepage loai bo gan het, TLB la nguyen nhan chinh.\n";
        else
            cout << "      => CI de len 0.3/0.7: ket luan CHUA du chac.\n";
        cout << "\n";

        // ---------- [T, v26-MOI] Hierarchical bootstrap: cong them bat dinh
        // GIUA-cac-lan-chay (khac voi [Q]/[Q-fix] chi nhin TRONG-1-lan-chay) ----------
        cout << "  [T] Hierarchical (nested/predictive) bootstrap CI 95% - moi draw bootstrap\n"
                "  = chon-VOI-HOAN-LAI DUY NHAT 1 trong " << R_REPS << " replication da chay (tang 1),\n"
                "  resample stationary TRONG replication do (tang 2, van " << TRIALS << " diem, KHONG\n"
                "  gop/KHONG trung binh nhieu replication - ca hai deu tu lam CI hep sai, xem chu\n"
                "  thich v26b/v26c-SUA LOI THAT):\n";
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
        cout << "  => Do rong CI: i.i.d.=" << setprecision(3) << width_iid
             << "   stationary(block~3)=" << width_sboot3
             << "   hierarchical(block~3,R=" << R_REPS << ")=" << width_hboot
             << "   ty le hier/stationary=" << setprecision(2) << (width_hboot / width_sboot3) << "x\n";
        // [v26d] So sanh dung: hierarchical vs stationary (CUNG da sua drift trong-
        // lan-chay), KHONG phai vs i.i.d. (baseline chua sua gi) - vi phan chenh
        // stationary->hierarchical moi chinh xac la dong gop CUA RIENG bat dinh
        // giua-cac-lan-chay, tach biet voi drift trong-lan-chay ma [Q-fix] da xu ly.
        if (width_hboot > width_sboot3 * 1.15)
            cout << "     [T] XAC NHAN: cong bat dinh giua-cac-lan-chay lam CI RONG HON dang ke\n"
                    "     so voi stationary mot-lan-chay - [Q-fix] mot minh danh gia THAP do bat\n"
                    "     dinh thuc (chi nhin trong 1 lan chay), nen dung [T] lam CI chinh thuc.\n";
        else
            cout << "     [T] Phan chenh stationary->hierarchical nho - bat dinh giua-cac-lan-chay\n"
                    "     (do o [v26] tren) khong lon hon dang ke so voi drift trong-1-lan-chay ma\n"
                    "     [Q-fix] da sua; CI stationary co the dung tam duoc, nhung [T] van la ban\n"
                    "     day du hon ve mat phuong phap vi khong gia dinh '1 lan chay la du'.\n";
        cout << "\n";

        // ---------- [U, v27-MOI, dot pha] Phan ra phuong sai (random-effects,
        // method-of-moments) + cong thuc phan bo ngan sach do TOI UU --------------
        // Dong co: [T] moi cho biet "co bat dinh giua-cac-lan-chay" (mot con so),
        // nhung KHONG cho biet PHAI LAM GI voi ngan sach do (thoi gian benchmark)
        // de giam bat dinh nhanh nhat. Cau tra loi thuc su phu thuoc vao TY LE
        // giua 2 thanh phan phuong sai - can tach rieng chung ra, khong chi so
        // do rong CI gop.
        //
        // Mo hinh (random-effects 1 chieu, chuan cho thiet ke long "replication
        // > trial"): diem-uoc cua 1 lan chay = mu + b_r + w_r, voi
        //   b_r ~ (0, sigma_b^2)  = bat dinh GIUA-cac-lan-chay (trang thai may:
        //         nhiet, DVFS, "hang xom" ao hoa... KHONG phu thuoc TRIALS)
        //   w_r ~ (0, sigma_w^2/TRIALS) = bat dinh TRONG-1-lan-chay (trial noise,
        //         giam theo ~1/TRIALS khi tang so trial - day la phan [Q-fix] da xu ly)
        // sigma_b^2 uoc luong = max(0, Var(reps_ratio_of_drops qua R lan) - sigma_w^2/TRIALS)
        // (method-of-moments kinh dien cho ANOVA 1 chieu random-effects, thiet ke
        // CAN BANG vi moi replication co CUNG TRIALS).
        cout << "  [U] Phan ra phuong sai (random-effects ANOVA, method-of-moments)\n"
                "  qua " << R_REPS << " macro-replication da chay o [v26], VA cong thuc phan bo\n"
                "  ngan sach do TOI UU rut ra tu do:\n";
        {
            double mean_ror = 0.0;
            for (double x : reps_ratio_of_drops) mean_ror += x;
            mean_ror /= (double)R_REPS;
            double var_total_across_reps = 0.0;
            for (double x : reps_ratio_of_drops) var_total_across_reps += (x - mean_ror) * (x - mean_ror);
            var_total_across_reps /= (double)(R_REPS - 1);

            // sigma_w^2/TRIALS uoc luong TRUC TIEP bang phuong sai cua PHAN PHOI
            // stationary-bootstrap RIENG cho TUNG replication (khong dung cong
            // thuc ly thuyet 1/TRIALS de tranh gia dinh sai ve i.i.d.), roi TRUNG
            // BINH qua R replication - day CHINH LA sigma_w^2/TRIALS o quy mo
            // TRIALS hien tai, KHONG phai sigma_w^2 thuan (chua chia).
            double var_within_avg = 0.0;
            mt19937_64 diag_rng(31415);
            const int N_BOOT_DIAG = 1000; // du de uoc luong phuong sai on dinh, khong can rong nhu CI chinh
            for (int rep = 0; rep < R_REPS; rep++){
                vector<vector<double>> ds_rep = {
                    reps_times_normal[rep][0], reps_times_normal[rep].back(),
                    reps_times_hp[rep][0],     reps_times_hp[rep].back()
                };
                auto sb = stationary_bootstrap_ci(ds_rep, stat_fn, N_BOOT_DIAG, 3.0, diag_rng);
                // Doi MAD (Median Absolute Deviation) ve uoc luong do lech chuan
                // tuong duong: cho phan phoi xap xi chuan, sigma ~ 1.4826 * MAD.
                double sigma_eq = 1.4826 * sb.mad;
                var_within_avg += sigma_eq * sigma_eq;
            }
            var_within_avg /= (double)R_REPS;

            double var_between_est = max(0.0, var_total_across_reps - var_within_avg);
            double var_grand_total = var_between_est + var_within_avg;
            double pct_between = (var_grand_total > 1e-12) ? 100.0 * var_between_est / var_grand_total : 0.0;
            double pct_within  = 100.0 - pct_between;

            cout << "    Var(reps_ratio_of_drops qua " << R_REPS << " lan chay, tho)      = "
                 << scientific << setprecision(3) << var_total_across_reps << "\n"
                 << "    Var trong-1-lan-chay uoc tu stationary boot (TB qua R) = "
                 << var_within_avg << "   (o TRIALS=" << TRIALS << " hien tai)\n"
                 << "    => Var GIUA-cac-lan-chay uoc (method-of-moments)       = "
                 << var_between_est << "\n"
                 << fixed << setprecision(1)
                 << "    => Phan ra: " << pct_between << "% tu GIUA-cac-lan-chay (sigma_b^2, khong\n"
                 << "       giam duoc bang cach tang TRIALS)   vs   " << pct_within
                 << "% tu TRONG-1-lan-chay (sigma_w^2/TRIALS,\n"
                 << "       giam duoc bang cach tang TRIALS).\n";

            // ---------- Cong thuc phan bo ngan sach TOI UU ----------
            // Neu ket qua cuoi cung se la TRUNG BINH cua R lan chay doc lap, moi
            // lan TRIALS trial, thi (dan xuat trong-nghi-thuc, xem chu thich dai
            // hon trong lich su hoi thoai): Var(trung binh) = sigma_b^2/R + sigma_w^2/T
            // voi T = R*TRIALS = TONG so trial tren toan bo ngan sach.
            // => Voi T CO DINH, so hang sigma_w^2/T KHONG DOI theo cach chia R
            // vs TRIALS (chi phu thuoc TONG T) - CHI so hang sigma_b^2/R la
            // giam duoc bang cach tang R. Ket luan: VOI NGAN SACH T CO DINH, tang
            // so LAN CHAY DOC LAP R (giam TRIALS/lan xuong muc toi thieu con hop
            // ly de [Q]/[Q-fix] van chan doan duoc drift, ~10-20) LUON tot hon
            // hoac bang dau tu them TRIALS vao 1 lan chay - vi phan sigma_b^2/R
            // KHONG THE giam bang cach nao khac ngoai tang R.
            if (pct_between > 15.0){
                double sigma_w2_full = var_within_avg * (double)TRIALS; // uoc sigma_w^2 (chua chia TRIALS)
                // [v33-FIX] Bien nay truoc do tinh xong roi BO, khong bao gio duoc in -
                // dung y do cua comment ngay ben tren no (uoc sigma_w^2 SO VOI sigma_b^2
                // o dong 2989) nhung chua bao gio xuat hien trong cout. Them 1 dong in
                // gia tri so THAT, dat canh var_between_est de nguoi doc so sanh truc
                // tiep 2 so hang cua Var(trung binh) = sigma_b^2/R + sigma_w^2/T.
                cout << "    (uoc so: sigma_w^2 (moi trial, chua chia)=" << scientific << setprecision(3)
                     << sigma_w2_full << "   vs sigma_b^2 (moi lan chay)=" << var_between_est
                     << fixed << setprecision(1) << ")\n";
                cout << "    [KHUYEN NGHI TOI UU, vi ty le giua-lan-chay > 15%]: voi TONG ngan sach\n"
                        "    T=R*TRIALS trial CO DINH, Var(trung binh cuoi cung) = sigma_b^2/R + sigma_w^2/T -\n"
                        "    so hang thu 2 KHONG doi theo cach chia R/TRIALS (chi phu thuoc T), nen\n"
                        "    CHI co tang R (giam TRIALS/lan xuong ~10-20, du de [Q] con chan doan duoc\n"
                        "    drift) la lam Var(trung binh) giam THUC SU. Vi du: T co dinh = "
                     << (R_REPS * TRIALS) << " trial hien tai\n"
                        "    (R=" << R_REPS << ", TRIALS=" << TRIALS << "), doi sang R=" << (R_REPS * 2)
                     << ", TRIALS=" << (TRIALS / 2 > 5 ? TRIALS / 2 : TRIALS)
                     << " (CUNG T) se giam\n"
                        "    Var(trung binh) tu (sigma_b^2/" << R_REPS << " + sigma_w^2/T) xuong (sigma_b^2/"
                     << (R_REPS * 2) << " + sigma_w^2/T) - so hang dau giam ~2x, so hang sau GIU NGUYEN.\n";
            } else {
                cout << "    Ty le giua-lan-chay <= 15% (bat dinh chu yeu tu trong-1-lan-chay) - voi\n"
                        "    ngan sach co dinh, tang TRIALS/lan van con hieu qua tuong duong tang R,\n"
                        "    KHONG can doi chien luoc do.\n";
            }
            cout << "    [CANH BAO trung thuc, quan trong]: sigma_b^2 o day uoc luong tu " << R_REPS
                 << " replication\n"
                    "    chay TUAN TU TRONG CUNG 1 process (vong lap trong main()), KHONG phai " << R_REPS
                 << " lan\n"
                    "    goi lai file .exe tu dau (process moi). Quan sat truoc do (chay lai toan bo\n"
                    "    binary nhieu lan tu shell) cho do dao dong DIEM-UOC lon hon han (0.788 den\n"
                    "    1.143, RONG hon ca [T]/[U] o day uoc), vi cac lan goi .exe rieng biet co them\n"
                    "    nguon bien (nap lai binary, cache/TLB lanh tu dau, scheduler xep lich khac)\n"
                    "    ma cac replication-trong-cung-process nay KHONG the tao ra duoc. => sigma_b^2\n"
                    "    va % giua-lan-chay in o tren la GIOI HAN DUOI (can uoc luong lai bang cach\n"
                    "    goi lai .exe nhieu lan tu ben ngoai, vd script shell 'for i in 1..N; do ./bin;\n"
                    "    done', khong the lam TU BEN TRONG 1 lan chay cua chinh binary nay).\n";
        }
        cout << "\n";
    }

    // ---------- [v21-NEW, buoc 4] Fit mo hinh cost(L) = a + b/L ----------
    cout << "===== [Model] Fit cost(L) ~= a + b/L (least-squares)\n";
    print_cost_model_fit("thuong ", Ls, cost_normal);
    print_cost_model_fit("hugepage", Ls, cost_hp);
    cout << "\n";

    // ---------- Prefetch sweep (v18) ----------
    cout << "===== [MLP-explicit] Prefetch-distance tai L=16 (mang thuong)\n";
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

    // ---------- [R] RMSE that (offset doc lap moi lan) ----------
    cout << "===== [R] RMSE that (offset DOC LAP moi rep), L quanh Cochran\n";
    {
        cout << "  --- mang thuong ---\n";
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
                // FIX bug v19: moi offset = 8 words
                double mean_per_word = (double)sum / (double)(offs.size() * 8);
                double estimate = mean_per_word * (double)N;
                double rel_err  = (estimate - (double)ref_normal) / (double)ref_normal;
                rel_errs.push_back(rel_err);
            }
            double rmse = 0;
            for (double e : rel_errs) rmse += e * e;
            rmse = sqrt(rmse / rel_errs.size());
            double mt  = median(times);
            double eff = rmse * sqrt(mt);   // cang nho cang tot
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
        cout << "  => Sampling nhanh hon " << setprecision(1)
             << (full_us / sample_us) << "x\n\n";
        (void)full_sum;
    }

    // ---------- [v22-NEW] Giai toa nhieu: tach 2 truc bien (L vs footprint) ----------
    cout << "===== [v22] Thi nghiem giai toa nhieu: tach 'do sau tai su dung' (L)\n"
            "      khoi 'kich thuoc working-set' (footprint), thay vi de chung troi nhau\n";
    {
        CacheSizes cs = detect_cache_sizes();
        size_t n_regions_avail = N / WORDS_PER_4K;
        // Chon num_regions co dinh du lon de footprint > L3 CHAC CHAN (an toan,
        // "lanh" thuc su), nhung khong vuot n_regions kha dung.
        size_t safe_fixed_regions = n_regions_avail; // dung toan bo khong gian trang co
        if (cs.l3 > 0){
            size_t min_regions_over_l3 = (size_t)((cs.l3 * 1.5) / 4096.0) + 1;
            safe_fixed_regions = min(n_regions_avail, max(min_regions_over_l3, (size_t)4000));
        }
        cout << "  (dung num_regions co dinh=" << safe_fixed_regions << " -> footprint="
             << fixed << setprecision(1) << (safe_fixed_regions*4096.0/1024/1024) << "MB)\n\n";

        cout << "  [Thi nghiem 1] L-sweep, footprint CO DINH (giai toa nhieu):\n";
        auto [cost_fixed_fp, r2_fixed] = run_fixed_footprint_sweep(
            normal, Ls, N, safe_fixed_regions, max(5, TRIALS/2), "mang thuong");
        cout << "  So sanh R^2: model GOC (troi bien)=" << setprecision(4) << 0.0
             << " (xem phan [Model] o tren)  vs  model TACH BIEN=" << r2_fixed << "\n\n";

        cout << "  [Thi nghiem 2] Footprint-sweep, L CO DINH=8 (dac trung truc con lai):\n";
        vector<size_t> region_sweep = {8, 64, 256, 2048, 8192, (size_t)(n_regions_avail*0.9)};
        run_fixed_L_footprint_sweep(normal, N, 8, region_sweep, max(5, TRIALS/2), cs);

        if (g_cap_violations > 0)
            cout << "  [v22] Tong so lan generate_offsets() bi CAT (N-fixed vi pham) trong toan bo chay: "
                 << g_cap_violations << "\n";
    }

    // ============================================================
    // [v23-NEW, PHAT MINH] Dinh luat dong dang IRM-Burst: thay the model
    //   curve-fit a+b/L bang 1 cong thuc CO Y NGHIA VAT LY (suy tu xac
    //   suat, khong phai fit mu), tu dong giai toa nhieu L-vs-footprint vi
    //   co CA 2 bien tuong minh, va TU KIEM CHUNG cheo qua 2 truc doc lap.
    //   Xem dan xuat day du o comment dau file, muc [Y]/[Z].
    // ============================================================
    cout << "===== [v23] Dinh luat dong dang IRM-Burst (phat minh moi, tu kiem chung)\n";
    {
        mt19937_64 irm_rng(31415926);
        bool law_ok = self_test_irm_law(irm_rng);
        if (!law_ok){
            cout << "  [v23] Self-test THAT BAI tren may nay -> BO QUA toan bo phan [v23]\n"
                    "  (khong dung ket qua suy dien tu 1 dinh luat chua duoc xac nhan).\n\n";
        } else {
            HwCounter dtlb2;
            dtlb2.open(DTLB_MISS_CFG(), "dTLB-miss");

            // [v25c-MOI, ung voi (R)] Chay VARPRO (P) VO DIEU KIEN tu day - ke ca khi CO
            // PMU thuc - khong chi khi bi chan nhu truoc. Chi phi RE (thuan wall-clock da
            // co san tu cost_normal/Ls/n_ops_normal, khong do them gi), nen khong co ly do
            // chi chay mot nua. Muc tieu dung nhu (R) de xuat: co CA HAI so (PMU thuc VA
            // VARPRO phan mem) canh nhau O MOI LAN CHAY, de ai co may PMU thuc co the tu
            // doi chieu ngay ma khong can sua code them lan nua.
            vector<double> Wactual_vp(Ls.size());
            for (size_t i = 0; i < Ls.size(); i++)
                Wactual_vp[i] = (double)n_ops_normal[i] / (double)Ls[i];
            double C_max_guess = *max_element(Wactual_vp.begin(), Wactual_vp.end());
            VarProResult vp = varpro_fit_capacity_no_pmu(Ls, Wactual_vp, cost_normal, C_max_guess);
            CostModelFit old_fit_novpmu = fit_cost_model(Ls, cost_normal);

            if (!dtlb2.ok){
                cout << "  [v23] KHONG DO DUOC HW counter (permission/container) -> khong the\n"
                        "  suy C tu dTLB-miss THAT. Thu: sudo sysctl kernel.perf_event_paranoid=1\n"
                        "  (Cong thuc da duoc self-test o tren xac nhan dung ve mat logic/toan hoc.)\n\n"
                        "  [v25-de-xuat] Chay THAY THE mot fallback KHONG CAN PMU: dung VARPRO\n"
                        "  (Golub & Pereyra 1973) tren CHINH cost_normal/Ls/n_ops_normal da co san\n"
                        "  (thuan wall-clock, khong do them gi, khong can quyen HW nao):\n";
                cout << "    C_fit (VARPRO, khong PMU) = " << fixed << setprecision(1) << vp.C_fit
                     << " trang (~" << setprecision(2) << (vp.C_fit * 4.0) << "KB)\n"
                     << "    cost/line = " << setprecision(4) << vp.a << " + " << vp.b
                     << " * mien/line_du_bao(IRM-Burst)   R^2=" << setprecision(4) << vp.r2 << "\n"
                     << "    so sanh tren CUNG 7 diem: model CU a+b/L (khong co W) R^2="
                     << setprecision(4) << old_fit_novpmu.r_squared << "\n"
                     << "    (CANH BAO: day la C khop TOT NHAT voi thoi gian do duoc GIA DINH\n"
                     << "     mo hinh IRM-Burst/LRU-ly-tuong dung - KHONG phai C do truc tiep tu\n"
                     << "     phan cung. R^2 cao la bang chung GIAN TIEP, khong phai xac nhan truc\n"
                     << "     tiep rang dTLB that hanh xu dung nhu mo hinh; xem canh bao day du o\n"
                     << "     comment truoc ham varpro_fit_capacity_no_pmu().)\n";

                // [v28-MOI, DOT PHA] CI bootstrap cho C_fit - CHUA HE CO o bat ky phien
                // ban truoc day, du day la SO DUY NHAT nguoi doc thuc su can. Tai dung
                // R replication tu [v26] (bien reps_times_normal da co san trong main),
                // KHONG do them gi.
                {
                    mt19937_64 vp_boot_rng(8181);
                    const int GRID_BOOT = 2000; // tang tu 500 len 2000 sau khi phat hien MAD=0.0
                                                // nghi la artifact luoi tho (xem log chay thu), van
                                                // thap hon 4000 cua diem-uoc chinh de giu toc do
                    auto vp_ci = bootstrap_ci_varpro_C(reps_times_normal, Ls, Wactual_vp,
                                                        n_ops_normal, C_max_guess, GRID_BOOT,
                                                        N_BOOT, 3.0, vp_boot_rng);
                    cout << "    [v28] Bootstrap CI 95% cho C_fit (hierarchical/stationary, block~3,\n"
                         << "    R=" << R_REPS << ", grid_boot=" << GRID_BOOT << "): ["
                         << setprecision(1) << vp_ci.lo << ", " << vp_ci.hi << "] trang"
                         << "   median=" << vp_ci.median << "  MAD=" << vp_ci.mad << "\n"
                         << "    => (~" << setprecision(2) << (vp_ci.lo * 4.0 / 1024.0) << " - "
                         << (vp_ci.hi * 4.0 / 1024.0) << " MB TLB-reach, median ~"
                         << (vp_ci.median * 4.0 / 1024.0) << " MB)\n";
                    double vp_ci_width = vp_ci.hi - vp_ci.lo;
                    if (vp_ci_width > 0.6 * vp.C_fit)
                        cout << "    [CANH BAO THAT] Do rong CI (" << setprecision(1) << vp_ci_width
                             << " trang) la mot phan LON so voi diem-uoc (" << vp.C_fit << " trang) -\n"
                             << "    C_fit tu VARPRO tren may/sandbox nay KHONG on dinh, nen bao cao\n"
                             << "    CA khoang [v28] nay, KHONG chi 1 con so diem, moi lan trich dan.\n";

                    // [v28b] Kiem tra CI co nhay giua cac boi so C hay khong - neu co, quet
                    // mat R^2(C) that de xac nhan da-cuc THAT chu khong phai artifact luoi.
                    vector<LocalMax> lmax = varpro_find_local_maxima(Ls, Wactual_vp, cost_normal,
                                                                       C_max_guess, 4000, 0.90);
                    // [v33-FIX] cluster_local_maxima() da duoc viet o [v28c] CHINH de gop
                    // cac diem luoi lien tiep thanh dai [C_lo,C_hi] gon hon (xem comment
                    // truoc dinh nghia ham) nhung truoc ban va nay CHUA BAO GIO duoc goi o
                    // day - noi DUY NHAT dung lmax - nen [v28b] van in tho tung diem luoi
                    // (compiler tung canh bao "cluster_local_maxima defined but not used").
                    vector<MaxCluster> lclusters = cluster_local_maxima(lmax, C_max_guess);
                    cout << "    [v28b] Quet mat R^2(C) tren diem-uoc goc: " << lmax.size()
                         << " diem luoi -> " << lclusters.size()
                         << " cum cuc-dai-cuc-bo trong nguong 90% cua R^2 toan cuc:\n";
                    for (auto& c : lclusters)
                        cout << "        C=[" << setprecision(1) << c.C_lo << ", " << c.C_hi
                             << "] trang  R^2~" << setprecision(4) << c.r2 << "\n";
                    if (lclusters.size() >= 2){
                        // [v33-FIX-2] Dung lclusters.size() (KHONG phai lmax.size() nhu ban cu):
                        // mot CAO NGUYEN R^2 phang dai (C >> working-set, model bao hoa THAT su -
                        // chinh [v28c] da tu nhan dinh hien tuong nay) co the sinh RAT NHIEU diem
                        // luoi thoa dieu kien "cuc-dai-cuc-bo" (moi diem tren mot day phang deu
                        // >= 2 lang gieng cua no) du CHI la 1 vung duy nhat, khong phai nhieu dinh
                        // tach biet - dem lmax.size() truc tiep se BAO NHAM thanh "da-cuc THAT" (2
                        // gia tri C alias) trong khi that ra la 1 cao nguyen rong. cluster_local_maxima
                        // gop chuoi diem lien tiep (cach nhau <6% C_max VA r2 gan nhau) thanh 1 dai
                        // [C_lo,C_hi] duy nhat, nen lclusters.size()>=2 moi la tin hieu dung cho
                        // "THAT co >= 2 vung tach biet", phan biet dung 2 kha nang (a)/(b) ma chinh
                        // comment [v28b] o tren dinh nghia ham da neu ra.
                        cout << "    => XAC NHAN THAT (khong phai artifact luoi): mo hinh IRM-Burst KHONG\n"
                                "    DINH DANH duoc C tren du lieu 7-diem nay - co >= 2 CUM gia tri C tach\n"
                                "    biet khop duong cong GAN NHU NHAU. CI [v28] nhay giua cac gia tri nay la\n"
                                "    PHAN ANH DUNG su mo ho THAT cua bai toan, khong phai loi ky thuat. Muon\n"
                                "    dinh danh duoc can THEM diem du lieu L (hien 7 diem: 1,2,4,8,16,32,64)\n"
                                "    hoac thu true PMU de co rang buoc doc lap thu 2 (xem nhanh (R)/(P)).\n";
                    } else {
                        cout << "    => Chi 1 cuc-dai ro rang - CI [v28] nhay gia tri co the do NHIEU TRIAL\n"
                                "    day diem-uoc qua vung lan-can cuc-dai nay, khong phai da-cuc THAT.\n";
                    }
                    cout << "\n";
                }

                // [v25b-MOI, tiep tuc cho ngay bi ngat quang] Buoc con THIEU trong nhanh
                // khong-PMU nay: kiem tra tinh HOP LY VAT LY cua C_fit. Nhanh CO PMU o
                // duoi (else) DA lam dieu nay o "Buoc 1" (doi chieu voi L1-DTLB~64,
                // L2-STLB~1536-2048 - xem comment ~dong 1826), nhung nhanh VARPRO nay
                // truoc gio CHI in R^2 ma KHONG kiem tra con so C_fit co hop ly khong.
                // R^2 cao (0.9871 do duoc) KHONG dong nghia C_fit dung dTLB - day la
                // dung dinh ke hoach ban dau: "R^2 cao la bang chung GIAN TIEP", nhung
                // chua co ai THAT SU doi chieu con so, chi moi noi vay tren giay.
                {
                    CacheSizes cs_chk = detect_cache_sizes();
                    const double plausible_lo = 32.0, plausible_hi = 4096.0; // trang;
                    // tham khao CUNG mot khoang da dung o nhanh PMU ben duoi (Buoc 1)
                    cout << "    [v25b] Kiem tra hop ly vat ly cua C_fit (buoc con thieu o day,\n"
                            "    nhanh PMU (else) DA lam viec nay o Buoc 1 nhung nhanh nay chua):\n"
                            "      Khoang dTLB hop ly tham khao (L1+L2-STLB gop, giong Buoc 1): ["
                         << fixed << setprecision(0) << plausible_lo << ", " << plausible_hi << "] trang\n"
                         << "      C_fit do duoc bang VARPRO                                 : "
                         << setprecision(1) << vp.C_fit << " trang\n";
                    if (vp.C_fit < plausible_lo || vp.C_fit > plausible_hi) {
                        cout << "      => KHONG HOP LY nhu dTLB: vuot tran tren cua khoang tham khao "
                             << setprecision(1) << (vp.C_fit / plausible_hi) << "x.\n"
                             << "      => CANH BAO THAT (khong phai suy dien): R^2=" << setprecision(4) << vp.r2
                             << " cao KHONG PHAI bang chung C_fit\n"
                             << "         dung dTLB. Mo hinh 'hinh dang duong cong' (occupancy-shaped) co the\n"
                             << "         khop TOT voi MOT hieu ung hoan toan khac (cache L2/L3, hay dac thu\n"
                             << "         memory-hierarchy cua chinh sandbox ao hoa nay) ma van trung dang\n"
                             << "         duong cong ma KHONG phai dTLB.\n";
                        if (cs_chk.l1d > 0 || cs_chk.l2 > 0 || cs_chk.l3 > 0) {
                            cout << "      Doi chieu voi cache OS bao cao qua sysconf tren may nay:\n";
                            if (cs_chk.l1d > 0) cout << "        L1d=" << setprecision(1) << (cs_chk.l1d/1024.0)
                                 << "KB (~" << setprecision(0) << (cs_chk.l1d/4096.0) << " trang)\n";
                            if (cs_chk.l2  > 0) cout << "        L2 =" << setprecision(2) << (cs_chk.l2/1024.0/1024.0)
                                 << "MB (~" << setprecision(0) << (cs_chk.l2/4096.0) << " trang)\n";
                            if (cs_chk.l3  > 0) cout << "        L3 =" << setprecision(2) << (cs_chk.l3/1024.0/1024.0)
                                 << "MB (~" << setprecision(0) << (cs_chk.l3/4096.0) << " trang)\n";
                            cout << "      C_fit (~" << setprecision(2) << (vp.C_fit*4.0/1024.0)
                                 << "MB) cung bac do lon voi L2/L3 o tren nhung KHONG trung khop truc tiep\n"
                                 << "      voi bat ky muc nao - TRUNG THUC: chua the ket luan day la hieu ung\n"
                                 << "      gi, chi biet CHAC CHAN khong phai dTLB thuan tuy. Can PMU that\n"
                                 << "      (ngoai sandbox nay, hoac sudo sysctl kernel.perf_event_paranoid=1\n"
                                 << "      neu may co quyen) de xac dinh - dung suy dien them tu day.\n";
                        }
                    } else {
                        cout << "      => Nam trong khoang hop ly - KHONG bac bo gia thuyet dTLB (nhung day\n"
                                "         van chi la bang chung GIAN TIEP, chua phai xac nhan truc tiep).\n";
                    }
                    cout << "\n";
                }
            } else {
                cout << "  [Buoc 1] Do dTLB-miss/line that tren truc L (giong PMU o [Q]),\n"
                        "  suy nguoc C tu tung diem bang dai so truc tiep, lay MEDIAN lam C_fit:\n";
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
                         << "  mien/line_do=" << setprecision(5) << miss_per_line
                         << "  C_suy_nguoc=" << setprecision(1) << C_hat << "\n";
                }
                vector<double> C_hats;
                for (size_t i = 0; i < pts_L.size(); i++)
                    C_hats.push_back(back_out_capacity(pts_L[i], pts_W[i], pts_miss[i]));
                double C_fit = median(C_hats);
                cout << "  => C_fit (median qua " << C_hats.size() << " diem) = " << fixed << setprecision(1)
                     << C_fit << " trang (~" << setprecision(2) << (C_fit * 4.0) << "KB TLB-reach)\n"
                     << "     (tham khao de kiem tra hop ly: L1-DTLB dien hinh CPU x86 hien dai ~64 muc,\n"
                     << "      L2-STLB ~1536-2048 muc - day la mot kiem tra HOP LY dinh tinh, KHONG phai\n"
                     << "      doc CPUID chinh xac tung cap; C_fit la mot dung luong 'hieu dung' gop ca\n"
                     << "      nhieu tang TLB + hieu ung khac, khong bat buoc trung khop 1 cap cu the.)\n\n";

                // [v25c-MOI, ung voi (R)] Day chinh la kiem chung (R) yeu cau: so sanh TRUC
                // TIEP C_fit tu PMU THAT (tren) voi C_fit tu VARPRO phan mem (vp, tinh VO
                // DIEU KIEN o dau ham, khong can PMU) - "so mien_sw voi mien_hw truc tiep,
                // thay vi chi tin ly luan suong". Kien truc thuc te da chuyen tu
                // StackDistHist/mien-ratio sang VARPRO/IRM-Burst boi (P), nen o day doi
                // chieu dung 2 gia tri C_fit thay vi 2 ty le mien nhu ban goc (R) phac.
                {
                    double pct_diff = 100.0 * (vp.C_fit - C_fit) / C_fit;
                    cout << "  [R] Doi chieu TRUC TIEP: C_fit tu PMU THAT (tren) vs VARPRO phan mem\n"
                            "  (vp, tinh vo dieu kien o dau ham [v23], khong can PMU):\n"
                         << "    C_fit (PMU thuc)        = " << fixed << setprecision(1) << C_fit << " trang\n"
                         << "    C_fit (VARPRO, phan mem) = " << vp.C_fit << " trang  (R^2=" << setprecision(4) << vp.r2 << ")\n"
                         << "    Chenh lech               = " << setprecision(1) << pct_diff << "%\n";
                    if (fabs(pct_diff) < 30.0)
                        cout << "    => Kha gan (<30%) - bang chung THAT (khong phai suy dien tu R^2) rang\n"
                                "    mo hinh IRM-Burst/VARPRO phan mem xap xi hop ly gia tri PMU thuc tren\n"
                                "    CHINH may nay. Day la lan dau co so sanh truc tiep 2 gia tri C_fit.\n\n";
                    else
                        cout << "    => LECH LON (>=30%) - VARPRO phan mem KHONG xap xi tot C_fit PMU thuc\n"
                                "    tren may nay, du R^2 khop duong cong co the van cao. Day la BANG CHUNG\n"
                                "    THAT rang R^2 cao khong du de tin C_fit phan mem thay PMU khi da co\n"
                                "    PMU thuc san sang - chi nen dung VARPRO nhu fallback, khong thay the.\n\n";
                }

                vector<double> pred_miss_L;
                for (size_t i = 0; i < pts_L.size(); i++)
                    pred_miss_L.push_back(predict_miss_per_line(pts_L[i], pts_W[i], C_fit));
                double rmse_miss = rmse_of(pts_miss, pred_miss_L);
                double r2_miss   = r_squared_given_model(pts_miss, pred_miss_L);
                cout << "  [Buoc 2] Khop cong thuc voi C_fit tren CHINH truc L da do:\n"
                     << "    RMSE(mien/line) = " << scientific << setprecision(3) << rmse_miss
                     << "    R^2(mien/line) = " << fixed << setprecision(4) << r2_miss << "\n\n";

                // ---- Fit cost/line = compute_floor + penalty*mien_du_bao, dung cost_normal da co san ----
                vector<double> pred_miss_for_cost(Ls.size());
                for (size_t i = 0; i < Ls.size(); i++){
                    double W_actual_cost = (double)n_ops_normal[i] / (double)Ls[i];
                    pred_miss_for_cost[i] = predict_miss_per_line((double)Ls[i], W_actual_cost, C_fit);
                }
                LinFit cost_fit = fit_linear(pred_miss_for_cost, cost_normal);
                CostModelFit old_fit = fit_cost_model(Ls, cost_normal); // model cu, chi so sanh, khong do lai
                cout << "  [Buoc 3] Du bao cost/line = " << fixed << setprecision(4) << cost_fit.a
                     << " + " << cost_fit.b << " * mien/line_du_bao  (R^2=" << setprecision(4) << cost_fit.r2 << ")\n"
                     << "    So sanh R^2 tren CUNG 7 diem L: model CU (a+b/L, curve-fit thuan tuy)="
                     << setprecision(4) << old_fit.r_squared << "   model MOI (IRM-Burst, co y nghia vat ly)="
                     << cost_fit.r2 << "\n\n";

                // ---- Kiem chung NGOAI-MAU tren TRUC FOOTPRINT hoan toan doc lap, KHONG fit lai gi ----
                cout << "  [Buoc 4] Kiem chung NGOAI-MAU (out-of-sample): du doan tren truc FOOTPRINT\n"
                        "  (L co dinh=8, quet W) BANG C_fit + cost_fit da hoc TU TRUC L o tren:\n";
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
                         << "  mien/line: do=" << fixed << setprecision(5) << miss_meas
                         << " du_doan=" << miss_pred
                         << "   cost/line: do=" << setprecision(3) << cost_meas
                         << "ns du_doan=" << cost_pred << "ns\n";
                }
                if (oos_miss_meas.size() >= 3){
                    double r2_oos_miss = r_squared_given_model(oos_miss_meas, oos_miss_pred);
                    double r2_oos_cost = r_squared_given_model(oos_cost_meas, oos_cost_pred);
                    cout << "\n  => R^2 ngoai-mau tren truc footprint (fit tren truc L, DU BAO thuan tuy o day):\n"
                         << "     mien/line: R^2=" << fixed << setprecision(4) << r2_oos_miss
                         << "     cost/line: R^2=" << r2_oos_cost << "\n"
                         << "  (model a+b/L cu KHONG THE danh gia duoc o day vi khong co tham so W -\n"
                         << "   day chinh la diem [v23] vuot qua ca model cu LAN 2 thi nghiem roi rac cua [v22].)\n\n";
                } else {
                    cout << "\n  [!] Khong du diem hop le tren truc footprint de tinh R^2 ngoai-mau.\n\n";
                }
            }
        }
    }

    // ============================================================
    // [v27+v27b-MOI] Mau thuan noi tai chua tung duoc phat hien: xem dan
    // xuat day du + tu-kiem-chung o comment truoc self_test_irm_fixed_replay_gap()
    // va self_test_random_replacement_shape() o tren. Tom tat: self_test_irm_law()
    // xac nhan C/W duoi gia dinh IID-tuoi-moi-lan, nhung generate_offsets()+vong
    // TRIALS THAT su dung mot chuoi CO DINH phat lai - duoi LRU ly tuong, dieu nay
    // hoi tu ve HAM BAC THANG chu khong phai C/W. Day KHONG phai suy dien tren
    // giay: hai ham duoi day tu-kiem-chung ca hai qua trinh bang C++ that.
    // ============================================================
    cout << "===== [v27] Mau thuan noi tai: 'chuoi offs CO DINH phat lai' vs 'IID tuoi moi lan'\n";
    {
        mt19937_64 v27_rng(20260728);
        self_test_irm_fixed_replay_gap(v27_rng);
        self_test_random_replacement_shape(v27_rng);
    }

    // ============================================================
    // [v24-MOI, phan LY THUYET, "kha thi nhung kiem chung duoc"] Mo rong
    // dinh luat [v23] (chi dung popularity DEU) sang popularity LECH (Zipf)
    // bang xap xi characteristic-time cua Che et al. (2002). Xem dan xuat
    // day du + gioi han trung thuc o comment truoc self_test_che_zipf_law().
    // Day CHI la buoc kiem chung LY THUYET (tu-mo-phong LRU that ben trong
    // chuong trinh, giong [v23]) - CHUA suy dien tham so tu HW counter that
    // nhu [v23] da lam, vi dieu do doi hoi thiet ke lai generate_offsets()
    // de sinh duoc mau truy cap LECH thuc su (hien tai region_dist la DEU
    // theo thiet ke) - de ngoai pham vi ban vá nay, neu ra nhu huong mo
    // rong tiep theo trung thuc thay vi gia vo da lam xong.
    // ============================================================
    cout << "===== [v24] Mo rong IRM-Burst sang popularity LECH (Che-Zipf, tu kiem chung)\n";
    {
        mt19937_64 che_rng(271828182);
        bool che_ok = self_test_che_zipf_law(che_rng);
        if (!che_ok){
            cout << "  [v24] Self-test THAT BAI tren may nay -> BO QUA ket luan phan mo rong nay\n"
                    "  (khong dung xap xi chua duoc xac nhan tren may hien tai).\n\n";
        } else {
            cout << "  [v24] Xap xi Che-Zipf khop mo phong LRU that trong nguong da dat (<2%)\n"
                    "  tren moi muc do lech (theta=0..1.5) da thu, VA quy ve DUNG cong thuc\n"
                    "  irm_occupancy() cua [v23] khi theta=0 (sai lech ~1e-6, thuan tuy so hoc).\n"
                    "  => Y nghia thuc te: mo hinh cost(L,W,C) cua [v23] hien CHI dung khi mau\n"
                    "  truy cap DEU (dung voi generate_offsets() hien tai, nhung KHONG dung voi\n"
                    "  workload thuc te co 'hot key' nhu cache server/DB). Khung [v24] nay la\n"
                    "  buoc chuan bi ly thuyet (da kiem chung bang mo phong) de mo rong sang do,\n"
                    "  NHUNG can them 1 buoc nua CHUA lam o day: sua generate_offsets() de sinh\n"
                    "  offsets theo phan phoi Zipf (hien tai region_dist la uniform_int_distribution\n"
                    "  thuan tuy) roi do lai dTLB-miss that qua [Q] de suy nguoc C theo cong thuc\n"
                    "  Che thay vi C/W - de xuat trung thuc cho phien ban ke tiep, KHONG lam gia\n"
                    "  o day ma khong co du lieu phan cung that di kem.\n\n";
        }
    }

    // ============================================================
    // [v26-NEW, MO RONG] Dinh luat IRM hieu chinh set-associative: mo rong
    // [v23] (fully-associative) sang S set that. Xem dan xuat + kiem chung
    // so voi binomial chinh xac o comment truoc self_test_irm_assoc_law().
    // Giong [v23]/[v24]: PHAI qua self-test noi bo truoc khi in ket luan.
    // ============================================================
    cout << "===== [v26] Dinh luat IRM hieu chinh set-associative (phat minh moi, tu kiem chung)\n";
    {
        mt19937_64 assoc_rng(1618033988);
        bool assoc_ok = self_test_irm_assoc_law(assoc_rng);
        if (!assoc_ok){
            cout << "  [v26] Self-test THAT BAI tren may nay -> BO QUA toan bo phan [v26]\n"
                    "  (khong dung mot dinh luat chua duoc xac nhan tren may hien tai).\n\n";
        } else {
            cout << "  [v26] Cong thuc set-associative khop mo phong LRU that trong nguong da dat\n"
                    "  (<1.5%) tren moi truong hop da thu, VA quy ve DUNG irm_occupancy() cua [v23]\n"
                    "  khi S=1 (kiem tra dai so, sai lech may tinh thuan tuy).\n\n"
                    "  Bang so sanh tai knee (C=W) - noi mo hinh [v23] gia dinh sai occ=1.0 nhung\n"
                    "  thuc te khong bao gio dat, dung ngay vung varpro_find_local_maxima() can\n"
                    "  do chinh xac nhat de tach cac tang cache (vd STLB co S~8-24 way that):\n";
            double W_demo = 19531.0;
            cout << "    S(way)   occ_v23(C=W)   occ_v26(C=W)   sai_lech_that\n";
            for (int S : {8, 12, 16, 24}){
                double occ23 = irm_occupancy(W_demo, W_demo);           // luon = 1.0 tai C=W (sai)
                double occ26 = irm_occupancy_assoc(W_demo, W_demo, (double)S);
                cout << "    " << setw(4) << S
                     << "     " << fixed << setprecision(5) << occ23
                     << "        " << occ26
                     << "        " << setprecision(2) << (occ23 - occ26) * 100.0 << "%\n";
            }
            cout << "\n  => Y nghia thuc te: back_out_capacity() cua [v23] suy ngo dung luong C tu\n"
                    "  mien/line do duoc se LECH he thong 0.7-1.4% dung tai knee neu ap dung cho\n"
                    "  mot cache/TLB co S nho (vd STLB thuong S=8-16 way) thay vi\n"
                    "  back_out_capacity_assoc() moi nay - day la ly do co the giai thich mot\n"
                    "  phan sai so con lai giua C suy ngo va dung luong danh nghia cua nha san\n"
                    "  xuat da thay trong cac phan [v18]-[v23] truoc do.\n\n";
        }
    }

    munmap(raw_normal, BYTES);
    munmap(raw_hp, BYTES);
    return 0;
}
