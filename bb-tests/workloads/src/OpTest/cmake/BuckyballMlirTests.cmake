if(NOT DEFINED LLVM_BUILD_DIR)
  message(FATAL_ERROR "LLVM_BUILD_DIR is not set")
endif()

if(NOT DEFINED BUDDY_OPT)
  message(FATAL_ERROR "BUDDY_OPT is not set")
endif()

set(FILECHECK ${LLVM_BUILD_DIR}/bin/FileCheck)

function(add_buckyball_mlir_contract TEST_NAME)
  cmake_parse_arguments(ARG "" "" "PASSES" ${ARGN})

  if(NOT DEFINED BUCKYBALL_MLIR_TEST_PREFIX)
    message(FATAL_ERROR "BUCKYBALL_MLIR_TEST_PREFIX is not set")
  endif()

  if(NOT ARG_PASSES)
    set(ARG_PASSES ${BUCKYBALL_LOWER_BUCKYBALL})
  endif()

  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.out.mlir)
  set(TARGET ${BUCKYBALL_MLIR_TEST_PREFIX}-${TEST_NAME}-mlir-test)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} ${ARG_PASSES} > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Buckyball MLIR contract IR: ${TARGET}"
    VERBATIM)

  add_custom_target(${TARGET}
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Buckyball MLIR contract: ${TARGET}"
    VERBATIM)

  if(DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
    add_dependencies(${BUCKYBALL_MLIR_GROUP_TARGET} ${TARGET})
  endif()
endfunction()
