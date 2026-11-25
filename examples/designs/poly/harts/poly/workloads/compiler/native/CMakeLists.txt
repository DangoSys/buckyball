set(HART ${DESIGN_HART_HART_poly})
# Native Ant preparation, completion, dependency and control contracts.
foreach(NATIVE_TEST IN ITEMS native_dependencies native_control native_edges)
  ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.cpp")
  ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.mlir")
  set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_${NATIVE_TEST}")
  set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${NATIVE_TEST}")
  add_custom_command(
    OUTPUT ${OUT}/host.o ${OUT}/native-bindings.o
    BYPRODUCTS ${OUT}/native.json ${OUT}/thread-0.bin ${OUT}/thread-1.bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
    COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.mlir
      -expand-strided-metadata -lower-affine -canonicalize
      "-lower-hart-native=design=${DESIGN_PB} hart=poly output-dir=${OUT}" -o ${OUT}/host.mlir
    COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir
      -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
    COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
      -convert-linalg-to-loops -expand-strided-metadata -lower-affine
      -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm
      -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/host-llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/host-llvm.mlir --buddy-to-llvmir -o ${OUT}/host.ll
    COMMAND ${DESIGN_COMPILER}/../llvm/build/bin/opt ${OUT}/host.ll -mtriple=riscv64 -mattr=+m
      "-passes=default<O2>" -S -o ${OUT}/host-opt.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/host-opt.ll
      -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
      -code-model=medium -O2 -o ${OUT}/host.o
    COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/thread-0.mlir
      -expand-strided-metadata -lower-affine -convert-scf-to-cf
      -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/thread-0-llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/thread-0-llvm.mlir --buddy-to-llvmir -o ${OUT}/thread-0.ll
    COMMAND ${DESIGN_COMPILER}/../llvm/build/bin/opt ${OUT}/thread-0.ll -mtriple=riscv64 -mattr=+m
      "-passes=default<O2>" -S -o ${OUT}/thread-0-opt.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/thread-0-opt.ll
      -filetype=obj -mtriple=riscv64 -mattr=+xbuckyball,+m -target-abi=lp64
      -code-model=medium -O2 -riscv-disable-using-constant-pool-for-large-ints -o ${OUT}/thread-0.o
    COMMAND ${ELF_CC} -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
      -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
      -Wl,-T,${BUCKYBALL_REPO_ROOT}/stack/kernel/program/code.ld
      ${OUT}/thread-0.S ${OUT}/thread-0.o -o ${OUT}/thread-0.elf
    COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text ${OUT}/thread-0.elf ${OUT}/thread-0.bin
    COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/thread-1.mlir
      -expand-strided-metadata -lower-affine -convert-scf-to-cf
      -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
      -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/thread-1-llvm.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/thread-1-llvm.mlir --buddy-to-llvmir -o ${OUT}/thread-1.ll
    COMMAND ${DESIGN_COMPILER}/../llvm/build/bin/opt ${OUT}/thread-1.ll -mtriple=riscv64 -mattr=+m
      "-passes=default<O2>" -S -o ${OUT}/thread-1-opt.ll
    COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/thread-1-opt.ll
      -filetype=obj -mtriple=riscv64 -mattr=+xbuckyball,+m -target-abi=lp64
      -code-model=medium -O2 -riscv-disable-using-constant-pool-for-large-ints -o ${OUT}/thread-1.o
    COMMAND ${ELF_CC} -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
      -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
      -Wl,-T,${BUCKYBALL_REPO_ROOT}/stack/kernel/program/code.ld
      ${OUT}/thread-1.S ${OUT}/thread-1.o -o ${OUT}/thread-1.elf
    COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text ${OUT}/thread-1.elf ${OUT}/thread-1.bin
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -ffreestanding
      -fno-builtin -msmall-data-limit=0 "--embed-dir=${OUT}"
      -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
      -c ${OUT}/native-bindings.c -o ${OUT}/native-bindings.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
      -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.cpp ${OUT}/host.o ${OUT}/native-bindings.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
      -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../${NATIVE_TEST}.cpp ${OUT}/host.o ${OUT}/native-bindings.o
      runtime-baremetal kernel
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
  add_dependencies(MLIRTest-all ${TEST}-build)
endforeach()
