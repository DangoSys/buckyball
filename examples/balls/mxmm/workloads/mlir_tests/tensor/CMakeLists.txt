if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX tensor)

set(PYTHON ${Python3_EXECUTABLE})
set(RUNTIME_PARAMS ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET})
if(NOT TARGET runtime)
  add_subdirectory(${BUCKYBALL_REPO_ROOT}/stack/runtime ${CMAKE_CURRENT_BINARY_DIR}/runtime)
endif()
if(NOT TARGET rvv_kernels)
  set(RUNTIME_PARAMS ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET})
  add_subdirectory(${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv
                   ${CMAKE_CURRENT_BINARY_DIR}/rvv)
endif()
if(NOT TARGET mxfp8_pack)
  add_subdirectory(${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/mxfp8
                   ${CMAKE_CURRENT_BINARY_DIR}/mxfp8)
endif()
file(STRINGS ${RUNTIME_PARAMS}/params.h ROWS_DEFINE REGEX "^#define BANK_LINES ")
file(STRINGS ${RUNTIME_PARAMS}/params.h WIDTH_DEFINE REGEX "^#define BANK_WIDTH ")
string(REGEX REPLACE ".* " "" BANK_ROWS "${ROWS_DEFINE}")
string(REGEX REPLACE ".* " "" ROW_BITS "${WIDTH_DEFINE}")
math(EXPR PACKED_BYTES "${BANK_ROWS} * ${ROW_BITS} / 8")
math(EXPR DOUBLE_PACKED_BYTES "${PACKED_BYTES} * 2")
math(EXPR HALF_PACKED_BYTES "${PACKED_BYTES} / 2")
set(BARE_RVV_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libquant_rvv_baremetal.a)
set(BARE_RVV_TARGET mxmm_quant_rvv_baremetal_${BUCKYBALL_MLIR_ACTIVE_TARGET})
set(RVV_SOURCE ${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv)
get_target_property(RVV_IMAGE_DIR rvv_images BINARY_DIR)
set(BARE_RVV_FLAGS -O2 -march=rv64gc -mcmodel=medany -std=c++20
  -I${RUNTIME_PARAMS} -I${RVV_IMAGE_DIR}
  -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
  -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib
  -I${BUCKYBALL_REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine)
add_custom_command(OUTPUT ${BARE_RVV_ARCHIVE}
  COMMAND ${ELF_CC} ${BARE_RVV_FLAGS} -c ${RVV_SOURCE}/quant_dispatch.cpp -o quant_dispatch_bare.o
  COMMAND ${ELF_CC} ${BARE_RVV_FLAGS} -c ${RVV_SOURCE}/../mxfp8/pack.cpp -o quant_pack_bare.o
  COMMAND ${ELF_CC} ${BARE_RVV_FLAGS} -c ${RVV_IMAGE_DIR}/images.cpp -o quant_images_bare.o
  COMMAND ${ELF_CC} ${BARE_RVV_FLAGS} -c ${RVV_SOURCE}/pack_banks.cpp -o pack_banks_bare.o
  COMMAND ${CMAKE_AR} rcs ${BARE_RVV_ARCHIVE} quant_dispatch_bare.o quant_pack_bare.o quant_images_bare.o pack_banks_bare.o
  DEPENDS rvv-images ${RVV_SOURCE}/pack_banks.cpp ${RVV_SOURCE}/quant_dispatch.cpp ${RVV_SOURCE}/../mxfp8/pack.cpp
          ${RVV_SOURCE}/../mxfp8/quant.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} VERBATIM)
add_custom_target(${BARE_RVV_TARGET}-build DEPENDS ${BARE_RVV_ARCHIVE})
add_library(${BARE_RVV_TARGET} STATIC IMPORTED)
set_target_properties(${BARE_RVV_TARGET} PROPERTIES IMPORTED_LOCATION ${BARE_RVV_ARCHIVE})
add_dependencies(${BARE_RVV_TARGET} ${BARE_RVV_TARGET}-build)
add_buckyball_mlir_test(mxmm_fp32 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET} PASSES
  -one-shot-bufferize=bufferize-function-boundaries
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball
  -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
  -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts
  LIBRARIES rvv_kernels rvv_images BAREMETAL_LIBRARIES ${BARE_RVV_TARGET})

set(QUANT_PASSES
  -canonicalize -cse -one-shot-bufferize=bufferize-function-boundaries
  -buffer-deallocation-pipeline=private-function-dynamic-ownership=false
  -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics
  -expand-strided-metadata -convert-linalg-to-loops -lower-buckyball
  -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
  -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts)
foreach(BEHAVIOR shared layout input branch)
  add_buckyball_mlir_test(mxmm_quant_${BEHAVIOR}
    TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET} PASSES ${QUANT_PASSES}
    LIBRARIES mxfp8_pack rvv_quant_backend rvv_images BAREMETAL_LIBRARIES ${BARE_RVV_TARGET})
  set(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET}-mxmm_quant_${BEHAVIOR}.mlir)
  set(CHECKED ${CMAKE_CURRENT_BINARY_DIR}/mxmm_quant_${BEHAVIOR}-checked.mlir)
  add_custom_command(OUTPUT ${CHECKED}
    COMMAND ${BUDDY_OPT} ${SOURCE} --target=${BUCKYBALL_MLIR_ACTIVE_TARGET}
      -canonicalize -cse -one-shot-bufferize=bufferize-function-boundaries
      -buffer-deallocation-pipeline=private-function-dynamic-ownership=false -o ${CHECKED}
    COMMAND ${BUCKYBALL_REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/build/bin/FileCheck
      ${SOURCE} --input-file=${CHECKED}
    DEPENDS ${SOURCE} ${BUDDY_OPT} VERBATIM)
  buckyball_mlirtest_name(TEST ${BUCKYBALL_MLIR_ACTIVE_TARGET} tensor_mxmm_quant_${BEHAVIOR})
  add_custom_target(${TEST}-ir-check DEPENDS ${CHECKED})
  add_dependencies(${TEST}-linux-build ${TEST}-ir-check)
  add_dependencies(${TEST}-baremetal-build ${TEST}-ir-check)
  add_dependencies(${BUCKYBALL_MLIR_GROUP_TARGET} ${TEST}-ir-check)
endforeach()
