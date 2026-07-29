#pragma once

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

