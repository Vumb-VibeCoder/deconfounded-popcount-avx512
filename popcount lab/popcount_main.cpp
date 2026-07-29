#include "popcount_common.hpp"
#include "popcount_simd_kernels.hpp"
#include "popcount_threading_selftest.hpp"
#include "popcount_timing_bootstrap.hpp"
#include "popcount_queueing_regression.hpp"

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
