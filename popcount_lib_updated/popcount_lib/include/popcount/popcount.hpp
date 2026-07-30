// popcount/popcount.hpp — C++ convenience wrapper quanh popcount.h (C API).
//
// Chi la mot lop mong (header-only, inline) goi thang xuong ham C ben duoi
// - khong them logic moi - de nguoi dung C++ duoc: namespace, overload
// std::vector<size_t>, enum class thay vi enum C, va std::string cho ten
// kernel. Neu ban dang link tu ngon ngu khac hoac muon ABI on dinh tuyet
// doi, hay dung truc tiep <popcount/popcount.h>.
#ifndef POPCOUNT_API_HPP
#define POPCOUNT_API_HPP

#include "popcount/popcount.h"

#include <cstdint>
#include <string>
#include <vector>

namespace popcount {

enum class SimdLevel {
    Scalar = POPCOUNT_LEVEL_SCALAR,
    AVX2 = POPCOUNT_LEVEL_AVX2,
    AVX512VPopcntdq = POPCOUNT_LEVEL_AVX512_VPOPCNTDQ,
};

inline void init() { popcount_init(); }

inline SimdLevel active_level() {
    return static_cast<SimdLevel>(popcount_active_level());
}

inline std::string active_level_name() { return popcount_active_level_name(); }

inline uint64_t block512(const uint64_t block[8]) { return popcount_block512(block); }

inline uint64_t array(const uint64_t* data, size_t count) {
    return popcount_array(data, count);
}

// Overload tien loi cho std::vector<uint64_t>.
inline uint64_t array(const std::vector<uint64_t>& data) {
    return popcount_array(data.data(), data.size());
}

inline uint64_t array_mt(const uint64_t* data, size_t count, unsigned n_threads = 0) {
    return popcount_array_mt(data, count, n_threads);
}

inline uint64_t array_mt(const std::vector<uint64_t>& data, unsigned n_threads = 0) {
    return popcount_array_mt(data.data(), data.size(), n_threads);
}

inline uint64_t bulk_gather(const uint64_t* data, const std::vector<size_t>& offsets,
                             int prefetch_distance = 0) {
    return popcount_bulk_gather(data, offsets.data(), offsets.size(), prefetch_distance);
}

// Tra ve true neu kernel dang dispatch khop 100% voi scalar tren du lieu
// ngau nhien (nen luon true tren phan cung dung dac ta).
inline bool self_test(uint32_t seed = 0, size_t n_blocks = 200000) {
    return popcount_self_test(seed, n_blocks) != 0;
}

} // namespace popcount

#endif // POPCOUNT_API_HPP
