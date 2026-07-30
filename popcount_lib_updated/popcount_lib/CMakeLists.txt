cmake_minimum_required(VERSION 3.14)
project(popcount LANGUAGES CXX)

option(POPCOUNT_BUILD_SHARED "Build libpopcount as a shared library" OFF)
option(POPCOUNT_BUILD_EXAMPLES "Build examples/demo" ON)
option(POPCOUNT_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Thu vien tu phat hien CPU luc CHAY (khong luc build) - nen KHONG can bat
# -mavx2/-mavx512* toan cuc; cac kernel SIMD duoc bao boc rieng bang
# __attribute__((target(...))) trong src/popcount.cpp. Chi can bat -mxsave
# toan cuc de read_xcr0() bien dich duoc tren moi may (bao gom may khong co
# AVX) — day la yeu cau da duoc xac nhan tu du an goc (xem comment [v21-FIX-CRITICAL]
# trong src/popcount.cpp).
add_compile_options(-mxsave)

add_library(popcount ${CMAKE_CURRENT_SOURCE_DIR}/src/popcount.cpp)
if(POPCOUNT_BUILD_SHARED)
  set_target_properties(popcount PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(popcount PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
find_package(Threads REQUIRED)
target_link_libraries(popcount PUBLIC Threads::Threads)

if(POPCOUNT_BUILD_EXAMPLES)
  add_executable(popcount_demo examples/demo.cpp)
  target_link_libraries(popcount_demo PRIVATE popcount)
endif()

if(POPCOUNT_BUILD_TESTS)
  enable_testing()
  add_executable(popcount_tests tests/test_popcount.cpp)
  target_link_libraries(popcount_tests PRIVATE popcount)
  add_test(NAME popcount_tests COMMAND popcount_tests)
endif()

include(GNUInstallDirs)
install(TARGETS popcount
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
