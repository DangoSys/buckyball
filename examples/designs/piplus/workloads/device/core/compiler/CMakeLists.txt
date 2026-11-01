set(CORE_PLUGIN "${DESIGN_DEVICE_BUILD}/core/compiler/libcore-compiler.so")
set(CORE_OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${CORE_PLUGIN} --load-pass-plugin=${CORE_PLUGIN})
foreach(PROFILE dense action)
  set(HART ${DESIGN_CORE_HART_${PROFILE}})
  set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-${PROFILE}_mxint8")
  set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/mxint8")
  add_custom_command(
    OUTPUT ${OUT}/mxint8.o
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
    COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8.mlir
      "-convert-linalg-to-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/core.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/core.mlir
      -convert-linalg-to-acc -o ${OUT}/acc.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
      -lower-frontend-fake-op -o ${OUT}/frontend.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
      -eliminate-empty-tensors -convert-elementwise-to-linalg
      "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
      -drop-equivalent-buffer-results
      "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
      -ownership-based-buffer-deallocation -buffer-deallocation-simplification
      -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
      -assign-physical-banks -o ${OUT}/banks.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
      -lower-acc -o ${OUT}/instructions.mlir
    COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
      -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
    COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
      "-lower-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/engine.mlir
    COMMAND ${CORE_OPT} ${OUT}/engine.mlir
      -convert-linalg-to-loops -expand-strided-metadata -lower-affine
      -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
      -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
      --buddy-to-llvmir -o ${OUT}/mxint8.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/mxint8.ll
      -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
      -code-model=medium -O2 -o ${OUT}/mxint8.o
    DEPENDS mxint8.mlir ${CORE_PLUGIN}
      ${DESIGN_THREAD_BUILD_${PROFILE}}/lib${PROFILE}-compiler.so
    VERBATIM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/mxint8.c ${OUT}/mxint8.o
      ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
      -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    DEPENDS mxint8.c ${OUT}/mxint8.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      runtime-baremetal engine
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
  add_dependencies(design-ctests-build ${TEST}-build)
endforeach()

foreach(PROFILE dense action)
  set(HART ${DESIGN_CORE_HART_${PROFILE}})
  set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-${PROFILE}_attention_tail")
  set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/attention_tail")
  add_custom_command(
    OUTPUT ${OUT}/attention_tail.o
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
    COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/attention_tail.mlir
      "-convert-linalg-to-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/core.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/core.mlir
      -convert-linalg-to-acc -o ${OUT}/acc.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
      -lower-frontend-fake-op -o ${OUT}/frontend.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
      -eliminate-empty-tensors -convert-elementwise-to-linalg
      "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
      -drop-equivalent-buffer-results
      "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
      -ownership-based-buffer-deallocation -buffer-deallocation-simplification
      -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
      -assign-physical-banks -o ${OUT}/banks.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
      -lower-acc -o ${OUT}/instructions.mlir
    COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
      -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
    COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
      "-lower-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/engine.mlir
    COMMAND ${CORE_OPT} ${OUT}/engine.mlir
      -convert-linalg-to-loops -expand-strided-metadata -lower-affine
      -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
      -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
      --buddy-to-llvmir -o ${OUT}/attention_tail.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/attention_tail.ll
      -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
      -code-model=medium -O2 -o ${OUT}/attention_tail.o
    DEPENDS attention_tail.mlir ${CORE_PLUGIN}
      ${DESIGN_THREAD_BUILD_${PROFILE}}/lib${PROFILE}-compiler.so
    VERBATIM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/attention_tail.c ${OUT}/attention_tail.o
      ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
      -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    DEPENDS attention_tail.c ${OUT}/attention_tail.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      runtime-baremetal engine
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
  add_dependencies(design-ctests-build ${TEST}-build)
endforeach()

foreach(PROFILE dense action)
  set(HART ${DESIGN_CORE_HART_${PROFILE}})
  set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-${PROFILE}_packed_weights")
  set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/packed_weights")
  add_custom_command(
    OUTPUT ${OUT}/packed_weights.o
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
    COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.mlir
      "-convert-linalg-to-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/core.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/core.mlir
      -convert-linalg-to-acc -o ${OUT}/acc.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
      -lower-frontend-fake-op -o ${OUT}/frontend.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
      -eliminate-empty-tensors -convert-elementwise-to-linalg
      "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
      -drop-equivalent-buffer-results
      "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
      -ownership-based-buffer-deallocation -buffer-deallocation-simplification
      -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
      -assign-physical-banks -o ${OUT}/banks.mlir
    COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
      -lower-acc -o ${OUT}/instructions.mlir
    COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
      -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
    COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
      "-lower-core=design=${DESIGN_PB} core=${PROFILE}" -o ${OUT}/engine.mlir
    COMMAND ${CORE_OPT} ${OUT}/engine.mlir
      -convert-linalg-to-loops -expand-strided-metadata -lower-affine
      -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
      -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
      --buddy-to-llvmir -o ${OUT}/packed_weights.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_weights.ll
      -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
      -code-model=medium -O2 -o ${OUT}/packed_weights.o
    DEPENDS packed_weights.mlir ${CORE_PLUGIN}
      ${DESIGN_THREAD_BUILD_${PROFILE}}/lib${PROFILE}-compiler.so
    VERBATIM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.c ${OUT}/packed_weights.o
      ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
      -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    DEPENDS packed_weights.c ${OUT}/packed_weights.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
      runtime-baremetal engine
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
  add_dependencies(design-ctests-build ${TEST}-build)
endforeach()
