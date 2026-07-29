// tests/test_popcount.cpp — Kiem chung dung dan cua libpopcount, doi chieu
// voi __builtin_popcountll() (nguon-chan-ly doc lap) tren nhieu kich thuoc
// va pattern truy cap khac nhau. Khong dung Google Test de giu thu vien
// khong phu thuoc gi ngoai chuan; chi 1 file don gian, exit code != 0 neu
// co bat ky check nao that bai (phu hop CTest qua add_test trong CMake).
#include <popcount/popcount.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (tai " << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_failures++; \
        } \
    } while (0)

static uint64_t reference_popcount(const std::vector<uint64_t>& v) {
    uint64_t s = 0;
    for (uint64_t x : v) s += __builtin_popcountll(x);
    return s;
}

int main() {
    popcount::init();
    std::cout << "Kernel active: " << popcount::active_level_name() << "\n";

    // 1) Self-test noi bo cua thu vien (SIMD dispatch vs scalar).
    CHECK(popcount::self_test(1, 200000), "popcount::self_test");

    // 2) popcount_array() tren nhieu kich thuoc (bao gom cac truong hop
    //    "duoi" khong chia het cho 4/8/16/64 tu).
    std::mt19937_64 rng(2024);
    for (size_t n : {0u, 1u, 3u, 7u, 8u, 15u, 16u, 31u, 63u, 64u, 65u, 127u,
                      1000u, 100003u, 1u << 20}) {
        std::vector<uint64_t> v(n);
        for (auto& x : v) x = rng();
        uint64_t expect = reference_popcount(v);
        uint64_t got = popcount::array(v);
        CHECK(got == expect, "popcount::array n=" << n
              << " expect=" << expect << " got=" << got);
    }

    // 3) popcount_array_mt() phai KHOP BIT-IDENTICAL voi ban don-luong.
    {
        std::vector<uint64_t> v(5'000'000);
        for (auto& x : v) x = rng();
        uint64_t single = popcount::array(v);
        for (unsigned nt : {0u, 1u, 2u, 4u, 8u}) {
            uint64_t multi = popcount::array_mt(v, nt);
            CHECK(multi == single, "array_mt n_threads=" << nt
                  << " single=" << single << " multi=" << multi);
        }
    }

    // 4) popcount_bulk_gather() tren offset ngau nhien, co/khong prefetch.
    {
        std::vector<uint64_t> v(200000);
        for (auto& x : v) x = rng();
        std::uniform_int_distribution<size_t> pick(0, v.size() - 8);
        std::vector<size_t> offsets;
        for (int i = 0; i < 5000; i++) offsets.push_back(pick(rng) & ~size_t(7));

        uint64_t expect = 0;
        for (size_t off : offsets)
            for (int w = 0; w < 8; w++) expect += __builtin_popcountll(v[off + w]);

        for (int pd : {0, 1, 4, 16}) {
            uint64_t got = popcount::bulk_gather(v.data(), offsets, pd);
            CHECK(got == expect, "bulk_gather prefetch_distance=" << pd
                  << " expect=" << expect << " got=" << got);
        }
    }

    // 5) popcount_block512() tren 1 khoi don.
    {
        alignas(64) uint64_t block[8];
        for (auto& x : block) x = rng();
        uint64_t expect = 0;
        for (auto x : block) expect += __builtin_popcountll(x);
        CHECK(popcount::block512(block) == expect, "block512");
    }

    if (g_failures == 0) {
        std::cout << "TAT CA CHECK DEU PASS.\n";
        return 0;
    }
    std::cerr << g_failures << " check THAT BAI.\n";
    return 1;
}
