// examples/demo.cpp — Vi du dung nhanh libpopcount (C++ wrapper).
#include <popcount/popcount.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

int main() {
    popcount::init();
    std::cout << "Kernel dang active: " << popcount::active_level_name() << "\n\n";

    // 1) Tu-kiem-chung truoc khi tin bat ky ket qua nao.
    bool ok = popcount::self_test(/*seed=*/42, /*n_blocks=*/100000);
    std::cout << "self_test: " << (ok ? "OK (khop scalar 100%)" : "THAT BAI") << "\n\n";

    // 2) popcount 1 mang lon.
    const size_t N = 20'000'000;
    std::mt19937_64 rng(1234);
    std::vector<uint64_t> data(N);
    for (auto& x : data) x = rng();

    auto t0 = std::chrono::steady_clock::now();
    uint64_t total = popcount::array(data);
    auto t1 = std::chrono::steady_clock::now();
    double ns_per_word = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)N;
    std::cout << "popcount::array:    tong = " << total
              << "   (" << ns_per_word << " ns/tu)\n";

    // 3) Ban da luong hoa cua cung phep tinh.
    t0 = std::chrono::steady_clock::now();
    uint64_t total_mt = popcount::array_mt(data);
    t1 = std::chrono::steady_clock::now();
    double ns_per_word_mt = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)N;
    std::cout << "popcount::array_mt: tong = " << total_mt
              << "   (" << ns_per_word_mt << " ns/tu, " << std::thread::hardware_concurrency()
              << " luong logic)\n";

    if (total != total_mt) {
        std::cerr << "LOI: array() va array_mt() cho ket qua khac nhau!\n";
        return 1;
    }

    // 4) popcount kieu "gather" tren cac khoi 512-bit ngau nhien.
    std::vector<size_t> offsets;
    std::uniform_int_distribution<size_t> pick(0, N - 8);
    for (int i = 0; i < 100000; i++) offsets.push_back(pick(rng) & ~size_t(7));
    uint64_t gather_sum = popcount::bulk_gather(data.data(), offsets, /*prefetch_distance=*/8);
    std::cout << "popcount::bulk_gather: tong = " << gather_sum << " tren "
              << offsets.size() << " khoi 512-bit\n";

    return 0;
}
