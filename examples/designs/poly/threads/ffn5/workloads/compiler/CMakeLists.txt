set(HART ${DESIGN_HART_HART_poly})
set(OUT "${CMAKE_CURRENT_BINARY_DIR}")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/matrix5_main.cpp")
foreach(NAME IN ITEMS matrix5_m1 matrix5_m3 matrix8_m1 matrix8_m3)
  string(REGEX MATCH "matrix([58])_m([13])" MATCH "${NAME}")
  set(BITS ${CMAKE_MATCH_1})
  set(INPUT_ROWS ${CMAKE_MATCH_2})
  set(SOURCE "matrix${BITS}.mlir")
  set(MAIN "matrix${BITS}_main.cpp")
  set(INPUT_K 1024)
  set(CHANNELS 64)
  set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-ffn5_${NAME}")
  ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${MAIN}")
  ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
  add_custom_command(
    OUTPUT ${OUT}/${NAME}.o
    COMMAND ${DESIGN_TARGET_OPT_ffn} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
      -convert-linalg-to-ffn5 -canonicalize -convert-linalg-to-acc -canonicalize -o ${OUT}/${NAME}-acc.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
      --input-file=${OUT}/${NAME}-acc.mlir
    COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/${NAME}-acc.mlir
      -eliminate-empty-tensors -convert-elementwise-to-linalg
      "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
      -drop-equivalent-buffer-results
      "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
      -ownership-based-buffer-deallocation -buffer-deallocation-simplification
      -bufferization-lower-deallocations -canonicalize
      -assign-physical-banks -lower-acc
      -expand-strided-metadata -lower-affine -canonicalize
      "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/${NAME}-kernel.mlir
    COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/${NAME}-kernel.mlir
      -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
      -o ${OUT}/${NAME}-copies.mlir
    COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/${NAME}-copies.mlir
      -convert-linalg-to-loops -expand-strided-metadata -lower-affine
      -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
      -reconcile-unrealized-casts -o ${OUT}/${NAME}-llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/${NAME}-llvm.mlir
      --buddy-to-llvmir -o ${OUT}/${NAME}.ll
    COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/${NAME}.ll -mtriple=riscv64
      -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUT}/${NAME}-opt.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/${NAME}-opt.ll
      -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
      -code-model=medium -O2 -o ${OUT}/${NAME}.o
    DEPENDS ${SOURCE} ${DESIGN_TARGET_PLUGIN_ffn} ${DESIGN_PB}
    VERBATIM)
  add_custom_command(
    OUTPUT ${OUT}/${TEST}-baremetal
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20 -DOUTPUT_CHANNELS=${CHANNELS} -DINPUT_K=${INPUT_K} -DINPUT_ROWS=${INPUT_ROWS}
      -I${DESIGN_REPO}/stack/kernel/include
      -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN} ${OUT}/${NAME}.o
      ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
      -lm -lgcc -o ${OUT}/${TEST}-baremetal
    DEPENDS ${MAIN} ${OUT}/${NAME}.o runtime-baremetal kernel
      ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${TEST}-build DEPENDS ${OUT}/${TEST}-baremetal)
  add_dependencies(MLIRTest-all ${TEST}-build)
endforeach()

add_custom_target(poly-ffn5-selection-check
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${CMAKE_CURRENT_SOURCE_DIR}/unsupported.mlir
    -convert-linalg-to-ffn5 -verify-diagnostics -o /dev/null
  DEPENDS unsupported.mlir ${DESIGN_TARGET_PLUGIN_ffn} ${DESIGN_PB}
  VERBATIM)
add_dependencies(MLIRTest-all poly-ffn5-selection-check)
