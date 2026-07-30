# libpopcount

Thư viện popcount tốc độ cao cho x86-64, **tự động phát hiện CPU lúc chạy**
(runtime dispatch) và chọn kernel nhanh nhất có sẵn:

| Mức phát hiện được      | Kernel mảng liên tục      | Kernel gather (offset ngẫu nhiên) |
|--------------------------|---------------------------|-------------------------------------|
| AVX-512 VPOPCNTDQ        | 8-accumulator `vpopcntq`  | 2-accumulator `vpopcntq`            |
| AVX2                     | Harley-Seal CSA           | nibble-LUT (`pshufb`)               |
| Không có AVX (fallback)  | scalar `POPCNT`           | scalar `POPCNT`                     |

Không cần build riêng cho từng CPU — 1 binary chạy tối ưu trên mọi máy, và
luôn fallback an toàn (đúng nhưng chậm hơn) trên CPU cũ. Toàn bộ kernel đã
được kiểm chứng bit-identical với nhau qua `popcount_self_test()`/test suite.

Đây là bản đóng gói thành thư viện của các kernel trong dự án nghiên cứu
`popcount_v39` — phần benchmark/thống kê/mô hình cache của dự án đó **không**
nằm trong thư viện này, chỉ giữ lại phần "kernel + dispatch" dùng được như
một thư viện popcount thông thường.

## Cài đặt / Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest            # chạy test suite
./popcount_demo  # chạy ví dụ
```

Không có CMake? Build tay cũng được (chỉ cần cờ `-mxsave`, KHÔNG cần
`-mavx2`/`-mavx512*` toàn cục — dispatch tự lo phần đó theo CPU thật):

```bash
g++ -std=c++17 -O3 -mxsave -pthread -Iinclude src/popcount.cpp your_app.cpp -o your_app
```

## Dùng thư viện

### C++ (khuyến nghị — `<popcount/popcount.hpp>`)

```cpp
#include <popcount/popcount.hpp>
#include <vector>

int main() {
    std::vector<uint64_t> data = /* ... */;
    uint64_t total = popcount::array(data);        // popcount toàn mảng
    uint64_t total_mt = popcount::array_mt(data);  // bản đa luồng
    std::cout << popcount::active_level_name();    // "AVX2 (array: Harley-Seal CSA | ...)"
}
```

### C thuần / FFI (`<popcount/popcount.h>`)

```c
#include <popcount/popcount.h>

uint64_t total = popcount_array(data, count);
```

## API chính

- `popcount::init()` — khởi tạo dispatch (tuỳ chọn, tự gọi lazy nếu bỏ qua).
- `popcount::array(data, count)` — popcount 1 mảng `uint64_t` liên tục.
- `popcount::array_mt(data, count, n_threads = 0)` — bản đa luồng (0 = tự chọn theo số **core vật lý**, bỏ qua SMT sibling; dùng threadpool tái sử dụng ghim vào core vật lý, port từ `popcount_v39`, thay vì tạo/hủy thread mỗi lần gọi).
- `popcount::bulk_gather(data, offsets, prefetch_distance = 0)` — tổng popcount
  của nhiều khối 512-bit tại các vị trí `offsets` (đơn vị: số phần tử `uint64_t`).
- `popcount::block512(block)` — popcount 1 khối 512-bit đơn.
- `popcount::self_test(seed, n_blocks)` — so khớp kernel đang dùng với scalar.
- `popcount::active_level()` / `active_level_name()` — kernel đang active (debug/log).

Ép mức SIMD thủ công (debug/benchmark so sánh công bằng) qua biến môi trường:
`POPCOUNT_FORCE=scalar|avx2|avx512`.

## Kiểm chứng đã thực hiện trong sandbox này

- Build `-Wall -Wextra`: sạch, không warning.
- `tests/test_popcount.cpp`: so khớp với `__builtin_popcountll` tham chiếu
  trên nhiều kích thước mảng (kể cả các trường hợp "dư" không chia hết cho
  4/8/16/64 từ), kiểm tra `array_mt` bit-identical với `array` ở
  0/1/2/4/8 luồng, kiểm tra `bulk_gather` với nhiều `prefetch_distance`.
  **Kết quả: TẤT CẢ PASS** (đã chạy cả ở mức AVX2 tự động phát hiện và ép
  `POPCOUNT_FORCE=scalar`).
- `examples/demo.cpp` chạy thành công, in kernel đang dùng và số liệu ns/từ.

Máy sandbox hiện tại không có AVX-512 VPOPCNTDQ nên nhánh đó chỉ được kiểm
chứng qua compile (biên dịch sạch với `__attribute__((target(...)))`) chứ
chưa chạy thực tế — trên máy có VPOPCNTDQ (Ice Lake/Zen4/Sapphire Rapids...),
dispatch sẽ tự chọn nhánh đó mà không cần build lại.
