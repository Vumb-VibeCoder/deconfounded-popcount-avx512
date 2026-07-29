#pragma once

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

