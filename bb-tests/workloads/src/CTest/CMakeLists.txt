set(RISCV_GNU_TOOLCHAIN ${RISCV})
set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-g++") 

#-------------------------------------------------------------------------------
# build linux version workload (current not supported)
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")


add_library(buckyball-lib STATIC ${CTEST_WORKLOAD_DIR}/buckyball.c)

add_executable(ctest_mvin_mvout_acc_test-linux ${CTEST_WORKLOAD_DIR}/mvin_mvout_acc_test.c)
target_link_libraries(ctest_mvin_mvout_acc_test-linux buckyball-lib)
add_executable(ctest_mvin_mvout_alternate_test-linux ${CTEST_WORKLOAD_DIR}/mvin_mvout_alternate_test.c)
target_link_libraries(ctest_mvin_mvout_alternate_test-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_ones-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_ones.c)
target_link_libraries(ctest_vecunit_matmul_ones-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_identity_random-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_identity_random.c)
target_link_libraries(ctest_vecunit_matmul_identity_random-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_row_col_vector-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_row_col_vector.c)
target_link_libraries(ctest_vecunit_matmul_row_col_vector-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_col_row_vector-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_col_row_vector.c)
target_link_libraries(ctest_vecunit_matmul_col_row_vector-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_random1-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random1.c)
target_link_libraries(ctest_vecunit_matmul_random1-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_random2-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random2.c)
target_link_libraries(ctest_vecunit_matmul_random2-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_random3-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random3.c)
target_link_libraries(ctest_vecunit_matmul_random3-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_zero_random-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_zero_random.c)
target_link_libraries(ctest_vecunit_matmul_zero_random-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_16xn_ones-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_ones.c)
target_link_libraries(ctest_vecunit_matmul_16xn_ones-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_16xn_random1-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random1.c)
target_link_libraries(ctest_vecunit_matmul_16xn_random1-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_16xn_random2-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random2.c)
target_link_libraries(ctest_vecunit_matmul_16xn_random2-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_16xn_random3-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random3.c)
target_link_libraries(ctest_vecunit_matmul_16xn_random3-linux buckyball-lib)
add_executable(ctest_vecunit_matmul_16xn_zero_random-linux ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_zero_random.c)
target_link_libraries(ctest_vecunit_matmul_16xn_zero_random-linux buckyball-lib)
add_executable(ctest_vecunit_simple_nn_forward_pass_test-linux ${CTEST_WORKLOAD_DIR}/vecunit_simple_nn_forward_pass_test.c)
target_link_libraries(ctest_vecunit_simple_nn_forward_pass_test-linux buckyball-lib)
add_executable(ctest_im2col_test-linux ${CTEST_WORKLOAD_DIR}/im2col_test.c)
target_link_libraries(ctest_im2col_test-linux buckyball-lib)

add_custom_target(buckyball-CTest-linux ALL DEPENDS
  ctest_mvin_mvout_acc_test-linux
  ctest_mvin_mvout_alternate_test-linux
  ctest_vecunit_matmul_ones-linux
  ctest_vecunit_matmul_identity_random-linux
  ctest_vecunit_matmul_row_col_vector-linux
  ctest_vecunit_matmul_col_row_vector-linux
  ctest_vecunit_matmul_random1-linux
  ctest_vecunit_matmul_random2-linux
  ctest_vecunit_matmul_random3-linux
  ctest_vecunit_matmul_zero_random-linux
  ctest_vecunit_matmul_16xn_ones-linux
  ctest_vecunit_matmul_16xn_random1-linux
  ctest_vecunit_matmul_16xn_random2-linux
  ctest_vecunit_matmul_16xn_random3-linux
  ctest_vecunit_matmul_16xn_zero_random-linux
  ctest_vecunit_simple_nn_forward_pass_test-linux
  ctest_im2col_test-linux
  COMMENT "Building linux workloads for Buckyball"
  VERBATIM)

#-------------------------------------------------------------------------------
# build baremetal version workload
#-------------------------------------------------------------------------------

set(C_FLAGS -std=gnu11 -g -DCHECK=1 -fno-common -O2 -static 
  -fno-builtin-printf -specs=htif_nano.specs -I${CTEST_WORKLOAD_DIR})

set(ELF_CC "riscv64-unknown-elf-gcc")

add_custom_target(buckyball-CTest-multicore-baremetal ALL
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_mvin_mvout_acc_test_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_acc_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_mvin_mvout_alternate_test_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_alternate_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_ones_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_ones.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_identity_random_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_identity_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_row_col_vector_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_row_col_vector.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_col_row_vector_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_col_row_vector.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random1_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random2_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random3_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_zero_random_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_zero_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_ones_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_ones.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random1_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random1.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random2_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random2.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random3_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random3.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_zero_random_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_zero_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_simple_nn_forward_pass_test_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/vecunit_simple_nn_forward_pass_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_im2col_test_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/im2col_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_bbfp_matmul_test_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/bbfpmatmul.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_bbfp_matmul_random1_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_bbfp_matmul_random2_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_bbfp_matmul_random3_multicore-baremetal ${CTEST_WORKLOAD_DIR}/start.S
    -DMULTICORE=3
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c   
  DEPENDS ${CTEST_WORKLOAD_DIR}/mvin_mvout_acc_test.c
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_alternate_test.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_ones.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_identity_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_row_col_vector.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_col_row_vector.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_zero_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_ones.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random1.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random2.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random3.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_zero_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_simple_nn_forward_pass_test.c
    ${CTEST_WORKLOAD_DIR}/im2col_test.c
    ${CTEST_WORKLOAD_DIR}/bbfpmatmul.c
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/bbfp_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/start.S
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMENT "Building baremetal workloads for Buckyball"
  VERBATIM)

add_custom_target(buckyball-CTest-singlecore-baremetal ALL
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_mvin_mvout_acc_test_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_acc_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_mvin_mvout_alternate_test_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_alternate_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_ones_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_ones.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_identity_random_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_identity_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_row_col_vector_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_row_col_vector.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_col_row_vector_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_col_row_vector.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random1_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random2_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_random3_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_zero_random_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_zero_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_ones_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_ones.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random1_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random1.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random2_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random2.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_random3_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random3.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_zero_random.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_vecunit_simple_nn_forward_pass_test_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/vecunit_simple_nn_forward_pass_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMAND ${ELF_CC} ${C_FLAGS} -o ctest_im2col_test_singlecore-baremetal 
    ${CTEST_WORKLOAD_DIR}/im2col_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  DEPENDS ${CTEST_WORKLOAD_DIR}/mvin_mvout_acc_test.c
    ${CTEST_WORKLOAD_DIR}/mvin_mvout_alternate_test.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_ones.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_identity_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_row_col_vector.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_col_row_vector.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random1.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random2.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_random3.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_zero_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_ones.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random1.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random2.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_random3.c
    ${CTEST_WORKLOAD_DIR}/vecunit_matmul_16xn_zero_random.c
    ${CTEST_WORKLOAD_DIR}/vecunit_simple_nn_forward_pass_test.c
    ${CTEST_WORKLOAD_DIR}/im2col_test.c
    ${CTEST_WORKLOAD_DIR}/buckyball.c
  COMMENT "Building baremetal workloads for Buckyball"
  VERBATIM)


add_custom_target(buckyball-CTest-build ALL DEPENDS
  buckyball-CTest-multicore-baremetal
  buckyball-CTest-singlecore-baremetal
  buckyball-CTest-linux
  COMMENT "Building baremetal workloads for Buckyball"
  VERBATIM
  )