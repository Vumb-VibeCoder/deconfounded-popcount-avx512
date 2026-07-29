# Thay đổi so với bản gốc `popcount_v39_fixed.cpp`

## 1) Dọn dead code (đã xác nhận bằng `-Wall -Wextra`)
- Xoá `floor_bulk_avx512_legacy` (2-acc), `noise_floor_v36`/`K_NOISE_V37`, `gen_this_workload_v36`.
- Sửa warning `unused parameter 'rng'` trong `self_test_irm_fixed_replay_gap`.

## 2 & 3) Giảm cấp phát heap lặp lại trên hot path bootstrap
Thêm `stationary_bootstrap_indices_into(idx, n, p, rng)` — ghi vào buffer có sẵn
thay vì trả `vector` mới. Trong `bootstrap_ci`, `stationary_bootstrap_ci`,
`hierarchical_bootstrap_ci`, `bootstrap_ci_varpro_C`: dùng buffer `thread_local`
(mỗi thread cấp phát 1 lần đầu, các draw bootstrap sau chỉ ghi đè giá trị,
không `push_back`/tạo `vector` mới mỗi lần).

## 4) Tách nhánh `prefetch_distance > 0` ra khỏi thân vòng lặp nóng
Áp dụng cho `popcount_bulk_scalar`, `popcount_bulk_avx2`, `popcount_bulk_avx512`,
`popcount_bulk_avx512_Kacc<K>` — chia thành 2 vòng lặp riêng (có/không prefetch)
thay vì kiểm tra điều kiện bất biến ở mỗi vòng lặp.

## 5) Tách file
File gốc (~4500 dòng) được chia thành 5 header + 1 file main, **include đúng
thứ tự gốc** nên preprocess ra cùng một TU y hệt bản cũ — không có rủi ro
lỗi liên kết (link) hay đổi hành vi:

- `popcount_common.hpp` — include, hằng số, popcount scalar cơ bản
- `popcount_simd_kernels.hpp` — các kernel SIMD bulk (popcount/floor/contig, AVX2/AVX-512), dispatch
- `popcount_threading_selftest.hpp` — topology core vật lý, thread pool, self-test bit-identical, benchmark Harley-Seal
- `popcount_timing_bootstrap.hpp` — RDTSCP/PMU/hugepages, hạ tầng song song hoá bootstrap, cost model, cache sizes
- `popcount_queueing_regression.hpp` — mô hình IRM/occupancy, self-test mô phỏng, Fenwick, sinh workload, hồi quy
- `popcount_main.cpp` — include 5 header trên + `bootstrap_ci_varpro_C`, `main()`, và phần còn lại

## Kiểm chứng
- Build với `-Wall -Wextra`: **0 warning** (trước đó có 4).
- Build bản tách file: **0 lỗi/warning**.
- Chạy cả 2 bản (gốc-đã-sửa monolithic vs bản tách file) với cùng N=2,000,000:
  toàn bộ self-test bit-identical đều **OK** ở cả 2 bản, log giống nhau 100%
  ngoại trừ các số đo thời gian (ns/tu) — chênh lệch đó là nhiễu đo đạc phần
  cứng bình thường giữa 2 lần chạy, không phải lỗi logic.

## Build
```bash
g++ -std=c++17 -O2 -mavx2 -mavx512f -mavx512vpopcntdq -pthread popcount_main.cpp -o popcount
```
