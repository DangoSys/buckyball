# Shared rushB object + host runner wiring for toy chip MLIR optests.
# Requires BuckyballWorkloads.cmake (add_buckyball_rushb_native) and
# BUCKYBALL_LOWER_*_RUSHB from bb-tests/workloads/CMakeLists.txt.

function(add_toy_mlir_rushb)
  cmake_parse_arguments(ARG "CXX" "NAME;TARGET_STEM"
    "FRONT_PASSES;SOURCES;INCLUDE_DIRS;DEPENDS" ${ARGN})

  if(NOT ARG_NAME OR NOT ARG_TARGET_STEM)
    message(FATAL_ERROR "add_toy_mlir_rushb requires NAME and TARGET_STEM")
  endif()
  if(NOT DEFINED BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS OR
     NOT DEFINED BUCKYBALL_LOWER_BUCKYBALL_RUSHB)
    message(FATAL_ERROR
      "add_toy_mlir_rushb requires BUCKYBALL_LOWER_*_RUSHB "
      "(define them in bb-tests/workloads/CMakeLists.txt)")
  endif()
  if(NOT DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST OR
     NOT DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
    return()
  endif()

  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_NAME}.mlir)
  set(RUSHB_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}-rushB.o)

  add_custom_command(
    OUTPUT ${RUSHB_OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      ${ARG_FRONT_PASSES}
      "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}"
      ${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS}
      -convert-linalg-to-loops
      -expand-strided-metadata
      -lower-affine
      -convert-scf-to-cf
      -convert-cf-to-llvm
      ${BUCKYBALL_LOWER_BUCKYBALL_RUSHB}
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -lower-buckyball-intrinsics-to-rushb
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=x86_64 -O2 -o ${RUSHB_OBJ}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT} ${BUDDY_TRANSLATE} ${BUDDY_LLC}
            ${ARG_DEPENDS}
    COMMENT "Building rushB ${ARG_NAME}.o from Toy MLIR"
    VERBATIM)

  set(RUSHB_CXX_FLAG)
  if(ARG_CXX)
    set(RUSHB_CXX_FLAG CXX)
  endif()

  add_buckyball_rushb_native(${ARG_TARGET_STEM}
    ${RUSHB_CXX_FLAG}
    OUTPUT_SUBDIR src/CTest/rushB
    SOURCES ${ARG_SOURCES} ${RUSHB_OBJ}
    INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
    DEPENDS ${RUSHB_OBJ} ${ARG_DEPENDS})
endfunction()
