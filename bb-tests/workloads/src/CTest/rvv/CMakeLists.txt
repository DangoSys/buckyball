set(ELF_CC "riscv64-unknown-elf-gcc")
set(ELF_CXX "riscv64-unknown-elf-g++")

#-------------------------------------------------------------------------------
# Set baremetal compilation flags for RVV
#-------------------------------------------------------------------------------
set(RVV_C_FLAGS -g -DPREALLOCATE=1 -DNR_LANES=64 -mcmodel=medany -static -O2 -ffast-math
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99
  -I${CTEST_RVV_WORKLOAD_DIR}/env
  -I${CTEST_RVV_WORKLOAD_DIR}/common)

set(RVV_CXX_FLAGS -g -DPREALLOCATE=1 -mcmodel=medany -static -O2 -ffast-math
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=c++17
  -specs=htif_nano.specs
  -I${CTEST_RVV_WORKLOAD_DIR}/env
  -I${CTEST_RVV_WORKLOAD_DIR}/common
  -I${CTEST_RVV_WORKLOAD_DIR}/utasks)

# Link options for baremetal
set(RVV_LINK_OPTS -static -nostdlib -nostartfiles -lm -lgcc
  -T ${CTEST_RVV_WORKLOAD_DIR}/common/test.ld)

# Common sources
file(GLOB RVV_COMMON_C_SRCS
  ${CTEST_RVV_WORKLOAD_DIR}/common/*.c
  ${CTEST_RVV_WORKLOAD_DIR}/common/ara/*.c)
file(GLOB RVV_COMMON_ASM_SRCS
  ${CTEST_RVV_WORKLOAD_DIR}/common/*.S)
set(RVV_COMMON_SRCS ${RVV_COMMON_C_SRCS} ${RVV_COMMON_ASM_SRCS})

#-------------------------------------------------------------------------------
# Define compilation functions for RVV benchmarks
#-------------------------------------------------------------------------------

# Generate baremetal executable for C benchmarks
function(add_rvv_baremetal_test_target TEST_NAME)
  set(EXECUTABLE "${TEST_NAME}-baremetal")
  set(TEST_DIR "${CTEST_RVV_WORKLOAD_DIR}/${TEST_NAME}")

  # Find all C sources in the test directory
  file(GLOB TEST_C_SRCS ${TEST_DIR}/*.c)

  # Check if there's a data generation script (only for tests that generate .S files)
  # Exclude tests that generate C header files instead
  set(GEN_SCRIPT "")
  if(NOT TEST_NAME MATCHES "vec-conv-3|vec-sep-conv-3|vec-slide-conv|vec-sgemm|vec-sgemv|vec-transpose-load|vec-transpose-store")
    file(GLOB GEN_SCRIPT ${TEST_DIR}/gen_data.py ${TEST_DIR}/*_gendata.py ${TEST_DIR}/gendata.py)
  endif()

  # Find existing assembly sources (non-data files like vec-*.S)
  file(GLOB TEST_ASM_SRCS ${TEST_DIR}/*.S)
  list(FILTER TEST_ASM_SRCS EXCLUDE REGEX "data\\.S$")

  set(TEST_SRCS ${TEST_C_SRCS} ${TEST_ASM_SRCS})
  set(DATA_ASM_FILE "")

  # If there's a generation script, add command to generate data.S
  if(GEN_SCRIPT)
    set(DATA_ASM_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_data.S)

    # Determine default arguments for each test
    set(GEN_ARGS "")
    if(TEST_NAME MATCHES "vec-cos|vec-exp|vec-log")
      set(GEN_ARGS "64")  # N_f64
    elseif(TEST_NAME MATCHES "vec-jacobi2d")
      set(GEN_ARGS "16 16")  # R C
    elseif(TEST_NAME MATCHES "vec-pathfinder")
      set(GEN_ARGS "10 20 30")  # num_runs cols rows
    elseif(TEST_NAME MATCHES "vec-conjugate-gradient|vec-spmv")
      set(GEN_ARGS "16 16 0.5")  # S N D
    elseif(TEST_NAME MATCHES "vec-igemm")
      set(GEN_ARGS "8 8 8")  # M N P
    elseif(TEST_NAME MATCHES "vec-softmax")
      set(GEN_ARGS "8 8")  # channels innerSize
    elseif(TEST_NAME MATCHES "vec-dropout")
      set(GEN_ARGS "1024")  # N
    elseif(TEST_NAME MATCHES "vec-fconv2d|vec-iconv2d")
      set(GEN_ARGS "64 3")  # matrix_width filter_size
    elseif(TEST_NAME MATCHES "vec-fconv3d")
      set(GEN_ARGS "16 3")  # matrix_width filter_size
    elseif(TEST_NAME MATCHES "vec-fdotprod|vec-dotprod")
      set(GEN_ARGS "1024")  # N
    elseif(TEST_NAME MATCHES "vec-roi-align")
      set(GEN_ARGS "2 3 16 16 4 8 8")  # batch_size depth height width n_boxes crop_h crop_w
    endif()

    # Convert space-separated args to list
    string(REPLACE " " ";" GEN_ARGS_LIST "${GEN_ARGS}")

    add_custom_command(
      OUTPUT ${DATA_ASM_FILE}
      COMMAND ${CMAKE_COMMAND} -E env python3 ${GEN_SCRIPT} ${GEN_ARGS_LIST} > ${DATA_ASM_FILE}
      DEPENDS ${GEN_SCRIPT}
      COMMENT "Generating data for ${TEST_NAME} with args: ${GEN_ARGS}"
      WORKING_DIRECTORY ${TEST_DIR}
    )
    list(APPEND TEST_SRCS ${DATA_ASM_FILE})
  else()
    # If no generation script, use existing data.S if present
    file(GLOB DATA_S ${TEST_DIR}/data.S)
    if(DATA_S)
      list(APPEND TEST_SRCS ${DATA_S})
    endif()
  endif()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${RVV_C_FLAGS}
      -I${TEST_DIR}
      -o ${EXECUTABLE}
      ${TEST_SRCS}
      ${RVV_COMMON_SRCS}
      ${RVV_LINK_OPTS}
    DEPENDS ${TEST_SRCS} ${RVV_COMMON_SRCS}
    COMMENT "Building RVV baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

# Generate baremetal executable for C++ benchmarks
# Note: C++ programs don't link common sources, they rely on htif_nano.specs
function(add_rvv_cpp_baremetal_test_target TEST_NAME)
  set(EXECUTABLE "${TEST_NAME}-baremetal")
  set(TEST_DIR "${CTEST_RVV_WORKLOAD_DIR}/${TEST_NAME}")

  # Find all C++ and assembly sources in the test directory
  file(GLOB TEST_CPP_SRCS ${TEST_DIR}/*.cc)
  file(GLOB TEST_ASM_SRCS ${TEST_DIR}/*.S)
  set(TEST_SRCS ${TEST_CPP_SRCS} ${TEST_ASM_SRCS})

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CXX} ${RVV_CXX_FLAGS}
      -I${TEST_DIR}
      -o ${EXECUTABLE}
      ${TEST_SRCS}
      ${RVV_LINK_OPTS}
    DEPENDS ${TEST_SRCS} ${CTEST_RVV_WORKLOAD_DIR}/utasks/utasks.h
    COMMENT "Building RVV C++ baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

#-------------------------------------------------------------------------------
# Build list - C benchmarks (baremetal only)
#-------------------------------------------------------------------------------
add_rvv_baremetal_test_target(vec-conditional)
add_rvv_baremetal_test_target(vec-conjugate-gradient)
add_rvv_baremetal_test_target(vec-conv-3)
add_rvv_baremetal_test_target(vec-cos)
add_rvv_baremetal_test_target(vec-div-approx)
add_rvv_baremetal_test_target(vec-dotprod)
add_rvv_baremetal_test_target(vec-dropout)
add_rvv_baremetal_test_target(vec-exp)
add_rvv_baremetal_test_target(vec-fconv2d)
add_rvv_baremetal_test_target(vec-fconv3d)
add_rvv_baremetal_test_target(vec-fdotprod)
add_rvv_baremetal_test_target(vec-iconv2d)
add_rvv_baremetal_test_target(vec-igemm)
add_rvv_baremetal_test_target(vec-jacobi2d)
add_rvv_baremetal_test_target(vec-log)
add_rvv_baremetal_test_target(vec-mixed_width_mask)
add_rvv_baremetal_test_target(vec-pathfinder)
add_rvv_baremetal_test_target(vec-roi-align)
add_rvv_baremetal_test_target(vec-sep-conv-3)
add_rvv_baremetal_test_target(vec-sgemm)
add_rvv_baremetal_test_target(vec-sgemm-v2)
add_rvv_baremetal_test_target(vec-sgemm-v3)
add_rvv_baremetal_test_target(vec-sgemv)
add_rvv_baremetal_test_target(vec-slide-conv)
add_rvv_baremetal_test_target(vec-softmax)
add_rvv_baremetal_test_target(vec-strlen)
add_rvv_baremetal_test_target(vec-spmv)
add_rvv_baremetal_test_target(vec-square-root-approx)
add_rvv_baremetal_test_target(vec-transpose-load)
add_rvv_baremetal_test_target(vec-transpose-store)

#-------------------------------------------------------------------------------
# Build list - C++ benchmarks
#-------------------------------------------------------------------------------
# add_rvv_cpp_baremetal_test_target(vec-tasks)
# add_rvv_cpp_baremetal_test_target(vec-daxpy)

#-------------------------------------------------------------------------------
# Create master build target
#-------------------------------------------------------------------------------
add_custom_target(buckyball-rvv-build ALL DEPENDS
  vec-conditional_baremetal
  vec-conjugate-gradient_baremetal
  vec-conv-3_baremetal
  vec-cos_baremetal
  vec-div-approx_baremetal
  vec-dotprod_baremetal
  vec-dropout_baremetal
  vec-exp_baremetal
  vec-fconv2d_baremetal
  vec-fconv3d_baremetal
  vec-fdotprod_baremetal
  vec-iconv2d_baremetal
  vec-igemm_baremetal
  vec-jacobi2d_baremetal
  vec-log_baremetal
  vec-mixed_width_mask_baremetal
  vec-pathfinder_baremetal
  vec-roi-align_baremetal
  vec-sep-conv-3_baremetal
  vec-sgemm_baremetal
  vec-sgemm-v2_baremetal
  vec-sgemm-v3_baremetal
  vec-sgemv_baremetal
  vec-slide-conv_baremetal
  vec-softmax_baremetal
  vec-strlen_baremetal
  vec-spmv_baremetal
  vec-square-root-approx_baremetal
  vec-transpose-load_baremetal
  vec-transpose-store_baremetal
  COMMENT "Building all RVV baremetal workloads for Buckyball"
  VERBATIM)
