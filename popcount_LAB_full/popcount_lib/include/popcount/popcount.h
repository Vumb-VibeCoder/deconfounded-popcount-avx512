/* popcount/popcount.h — Public C API of libpopcount.
 *
 * Thu vien popcount toi uu cho x86-64: tu dong phat hien CPU luc chay
 * (runtime dispatch) va chon kernel nhanh nhat co san — AVX-512 VPOPCNTDQ
 * (8-accumulator) > AVX2 (Harley-Seal CSA cho mang lien tuc, nibble-LUT cho
 * truy cap gather) > scalar POPCNT — luon fallback an toan tren CPU cu
 * khong ho tro AVX. Ket qua BIT-IDENTICAL giua moi muc SIMD (da tu-kiem-
 * chung, xem popcount_self_test()).
 *
 * API nay la C-linkage (extern "C") nen dung duoc tu C, hoac lam FFI boundary
 * cho ngon ngu khac (Python ctypes/cffi, Rust, v.v.). Neu dung C++, xem them
 * <popcount/popcount.hpp> de co wrapper tien loi hon (std::vector, namespace).
 */
#ifndef POPCOUNT_API_H
#define POPCOUNT_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Muc SIMD dang duoc dung o hau truong (tham khao/debug, khong can thiet cho
 * dung API thong thuong vi dispatch la tu dong). */
typedef enum {
    POPCOUNT_LEVEL_SCALAR = 0,
    POPCOUNT_LEVEL_AVX2 = 1,
    POPCOUNT_LEVEL_AVX512_VPOPCNTDQ = 2
} popcount_simd_level_t;

/* Khoi tao dispatch (phat hien CPU 1 LAN, thread-safe, idempotent). KHONG
 * bat buoc phai goi truoc — moi ham public tu goi lazy-init noi bo — nhung
 * co the goi truoc (vd luc app khoi dong) de tranh chi phi kiem tra o lan
 * dung dau tien nam tren duong nong. */
void popcount_init(void);

/* Muc SIMD dang active sau khi dispatch (chi de log/debug). */
popcount_simd_level_t popcount_active_level(void);

/* Ten nguoi-doc-duoc cua kernel dang active (vd "AVX-512 VPOPCNTDQ
 * (contig: 8-acc | floor: 8-acc | gather: 2-acc)"). Con tro tra ve co doi
 * song bang thoi gian chay chuong trinh (khong can free). */
const char* popcount_active_level_name(void);

/* popcount cua 1 khoi 512-bit (8 x uint64_t) tai con tro `block` (KHONG can
 * align). Day la kernel "atom" nhanh nhat cho 1 don vi 64 byte don le. */
uint64_t popcount_block512(const uint64_t block[8]);

/* Tong popcount cua `count` phan tu uint64_t LIEN TUC trong bo nho, bat dau
 * tu `data`. Day la duong dan chinh cho "popcount 1 mang lon" — tu dong dung
 * Harley-Seal CSA (AVX2) hoac 8-accumulator VPOPCNTDQ (AVX-512) tuy CPU. */
uint64_t popcount_array(const uint64_t* data, size_t count);

/* Giong popcount_array() nhung chia cong viec cho nhieu thread (huu ich khi
 * count du lon, vd >= vai trieu phan tu, de bu chi phi tao/dieu phoi thread).
 * n_threads = 0 nghia la tu chon (min(std::hardware_concurrency, mac dinh
 * hop ly)). Voi count nho, ham tu dong roi lai don-luong. */
uint64_t popcount_array_mt(const uint64_t* data, size_t count, unsigned n_threads);

/* Tong popcount cua nhieu khoi 512-bit (8 x uint64_t) NAM RAI RAC trong bo
 * nho `data`, vi tri tung khoi cho boi `offsets[i]` (don vi: SO PHAN TU
 * uint64_t, khong phai byte) — dung cho pattern "gather" (vd cac trang
 * 4KB/cache-line duoc truy cap theo thu tu ngau nhien). `prefetch_distance`
 * > 0 se prefetch truoc offsets[i+prefetch_distance] khi xu ly offsets[i]
 * (0 = tat prefetch). */
uint64_t popcount_bulk_gather(const uint64_t* data, const size_t* offsets,
                               size_t n_offsets, int prefetch_distance);

/* Tu-kiem-chung: so sanh kernel dang dispatch (co the la SIMD) voi scalar
 * tren `n_blocks` khoi 512-bit ngau nhien (seed cho de tai lap). Tra ve 1
 * neu KHOP 100%, 0 neu phat hien sai lech (khong bao gio nen xay ra tren
 * phan cung dung dac ta — neu co, thuong la loi microcode/compiler). */
int popcount_self_test(uint32_t seed, size_t n_blocks);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POPCOUNT_API_H */
