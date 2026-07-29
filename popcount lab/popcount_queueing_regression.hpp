#pragma once

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

