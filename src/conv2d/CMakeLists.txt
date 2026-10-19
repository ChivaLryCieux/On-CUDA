add_executable(bench_conv
    bench_conv.cu
    conv.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
    ${PROJECT_SOURCE_DIR}/src/matmul/matmul.cu
)

target_include_directories(bench_conv PRIVATE
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/matmul
)

# Debug harness: isolates im2col / GEMM / bias against with- and without-bias
# CPU references.
add_executable(debug_conv
    debug_conv.cu
    conv.cu
    ${PROJECT_SOURCE_DIR}/src/matmul/matmul.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
)
target_include_directories(debug_conv PRIVATE
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/matmul
)
