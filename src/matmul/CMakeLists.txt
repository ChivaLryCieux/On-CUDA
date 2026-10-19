add_executable(bench_matmul
    bench_matmul.cu
    matmul.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
)

target_include_directories(bench_matmul PRIVATE ${PROJECT_SOURCE_DIR}/src/common)
target_link_libraries(bench_matmul PRIVATE CUDA::cublas)

# Debug harness: runs one impl on a tiny problem and dumps the error pattern.
add_executable(debug_mm
    debug_mm.cu
    matmul.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
)
target_include_directories(debug_mm PRIVATE ${PROJECT_SOURCE_DIR}/src/common)
