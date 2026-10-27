# Bank-level / ball-op MLIR -> baremetal + linux ELF (same pipeline as toy MLIRTest).
function(buckyball_enforce_mlir_test_line_limit SOURCE_FILE)
    if(NOT EXISTS "${SOURCE_FILE}")
        message(FATAL_ERROR "MLIR test source does not exist: ${SOURCE_FILE}")
    endif()
    file(READ "${SOURCE_FILE}" _content)
    string(REGEX MATCHALL "\n" _newlines "${_content}")
    list(LENGTH _newlines _lines)
    string(REGEX MATCH "[^\n]$" _trailing_content "${_content}")
    if(_trailing_content)
        math(EXPR _lines "${_lines} + 1")
    endif()
    if(_lines GREATER 100)
        message(
            FATAL_ERROR
            "MLIR test exceeds 100 lines: ${SOURCE_FILE} (${_lines} lines). "
            "Split it into one focused test per file."
        )
    endif()
endfunction()

function(add_buckyball_mlir_test NAME)
    cmake_parse_arguments(
        ARG
        ""
        "TARGET"
        "PASSES;LIBRARIES;BAREMETAL_LIBRARIES"
        ${ARGN}
    )

    if(NOT DEFINED BUCKYBALL_MLIR_TEST_PREFIX)
        message(FATAL_ERROR "BUCKYBALL_MLIR_TEST_PREFIX is not set")
    endif()
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "${NAME}: TARGET is required")
    endif()
    list(FIND DESIGN_PROFILES "${ARG_TARGET}" TARGET_INDEX)
    if(TARGET_INDEX EQUAL -1)
        message(FATAL_ERROR "${NAME}: unknown Buckyball target ${ARG_TARGET}")
    endif()
    if(NOT DESIGN_PROFILES_${ARG_TARGET}_THREAD_MEM_BANK_NUM GREATER 0)
        message(FATAL_ERROR "${ARG_TARGET}: no physical banks")
    endif()
    set(ASSIGN_PHYSICAL_BANKS -assign-physical-banks)
    set(BIN "${DESIGN_COMPILER}/bin")
    # buddy-opt with this target's compiler plugins.
    string(JOIN " " OPT ${DESIGN_TARGET_OPT_${ARG_TARGET}})
    # The banks and Ball ISA of the target, as compiler flags.
    set(TARGET_DEFINES ${DESIGN_TARGET_DEFS_${ARG_TARGET}})
    list(TRANSFORM TARGET_DEFINES PREPEND "-D")
    set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.mlir)
    set(MLIR_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.mlir.in)
    if(EXISTS ${MLIR_TEMPLATE})
        set(BB_BANK_LINES ${DESIGN_PROFILES_${ARG_TARGET}_THREAD_MEM_BANK_ENTRIES})
        set(BB_BANK_WIDTH_BITS ${DESIGN_PROFILES_${ARG_TARGET}_THREAD_MEM_BANK_WIDTH})
        math(EXPR BB_BANK_I8_LANES "${BB_BANK_WIDTH_BITS} / 8")
        math(EXPR BB_BANK_I32_LANES "${BB_BANK_WIDTH_BITS} / 32")
        if(
            BB_BANK_LINES LESS 1
            OR BB_BANK_I8_LANES LESS 1
            OR BB_BANK_I32_LANES LESS 1
        )
            message(FATAL_ERROR "${NAME}: invalid generated bank parameters")
        endif()
        set(MLIR_SRC ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${NAME}.mlir)
        configure_file(${MLIR_TEMPLATE} ${MLIR_SRC} @ONLY)
    endif()
    set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_main.cpp)
    buckyball_enforce_mlir_test_line_limit(${MLIR_SRC})
    buckyball_enforce_mlir_test_line_limit(${MAIN_SRC})
    set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${NAME}.o)
    set(PREFIX ${BUCKYBALL_MLIR_TEST_PREFIX})
    set(TEST_CASE ${NAME})
    if(NAME MATCHES "^${PREFIX}_(.+)$")
        set(TEST_CASE ${CMAKE_MATCH_1})
    endif()
    set(TEST_NAME
        "${DESIGN_NAME}-${ARG_TARGET}-mlirtest-${PREFIX}_${TEST_CASE}"
    )
    set(BAREMETAL_BIN ${TEST_NAME}-baremetal)
    set(LINUX_BIN ${TEST_NAME}-linux)
    set(BAREMETAL_TARGET ${BAREMETAL_BIN}-build)
    set(LINUX_TARGET ${LINUX_BIN}-build)
    set(GROUP_TARGET ${TEST_NAME})

    set(C_FLAGS
        -g
        -fno-common
        -O2
        -static
        -march=rv64gc
        -mcmodel=medany
        -fno-builtin-printf
        -specs=nano.specs
        -specs=nosys.specs
        -nostartfiles
        -DBAREMETAL
        -Wl,-T,${BBSIM_LD}
    )
    set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
    set(CRUNNER_UTILS_SRC
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/mlir.cpp
    )
    set(BBHW_MEM_C ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/dma.c)
    set(LLVM_MLIR_EXECUTION_ENGINE_DIR
        ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine
    )

    if(ARG_PASSES)
        set(MLIR_PASSES ${ARG_PASSES})
    else()
        set(MLIR_PASSES
            ${ASSIGN_PHYSICAL_BANKS}
            -lower-buckyball
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
        )
    endif()
    string(JOIN " " MLIR_PASS_COMMAND ${MLIR_PASSES})

    add_custom_command(
        OUTPUT ${OBJ}
        COMMAND
            bash -o pipefail -c
            "${OPT} ${MLIR_SRC} ${MLIR_PASS_COMMAND} | ${BIN}/buddy-translate --buddy-to-llvmir | ${BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o ${OBJ}"
        DEPENDS ${MLIR_SRC} ${DESIGN_TARGET_PLUGIN_${ARG_TARGET}}
        COMMENT "Building ${NAME}.o from ${PREFIX} MLIR"
        VERBATIM
    )

    set(MLIR_LINK_LIBRARIES)
    foreach(LIBRARY IN LISTS ARG_LIBRARIES)
        if(NOT TARGET ${LIBRARY})
            message(FATAL_ERROR "${NAME}: unknown runtime library ${LIBRARY}")
        endif()
        list(APPEND MLIR_LINK_LIBRARIES "$<TARGET_FILE:${LIBRARY}>")
    endforeach()

    set(MLIR_BAREMETAL_LIBRARIES ${MLIR_LINK_LIBRARIES})
    if(ARG_BAREMETAL_LIBRARIES)
        set(MLIR_BAREMETAL_LIBRARIES)
        foreach(LIBRARY IN LISTS ARG_BAREMETAL_LIBRARIES)
            if(NOT TARGET ${LIBRARY})
                message(
                    FATAL_ERROR
                    "${NAME}: unknown bare-metal library ${LIBRARY}"
                )
            endif()
            list(APPEND MLIR_BAREMETAL_LIBRARIES "$<TARGET_FILE:${LIBRARY}>")
        endforeach()
    endif()

    add_custom_command(
        OUTPUT ${BAREMETAL_BIN}
        COMMAND
            ${ELF_CC} ${C_FLAGS} -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${WORKLOAD_LIB_DIR} ${TARGET_DEFINES} -o
            ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN} ${BAREMETAL_DIR}/crt0.S
            ${BAREMETAL_DIR}/syscalls.c ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C}
            ${MAIN_SRC} ${OBJ} ${MLIR_BAREMETAL_LIBRARIES}
        DEPENDS
            ${OBJ}
            ${ARG_BAREMETAL_LIBRARIES}
            ${ARG_LIBRARIES}
            ${MAIN_SRC}
            ${CRUNNER_UTILS_SRC}
            ${BBHW_MEM_C}
            ${BAREMETAL_DIR}/crt0.S
            ${BAREMETAL_DIR}/syscalls.c
            ${BBSIM_LD}
        COMMENT "Linking ${BAREMETAL_BIN}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${LINUX_BIN}
        COMMAND
            ${LINUX_CXX} ${LINUX_FLAGS} -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${WORKLOAD_LIB_DIR} ${TARGET_DEFINES} -o
            ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN} ${CRUNNER_UTILS_SRC} -x c
            ${BBHW_MEM_C} -x none ${MAIN_SRC} ${OBJ} ${MLIR_LINK_LIBRARIES}
        DEPENDS
            ${OBJ}
            ${ARG_LIBRARIES}
            ${MAIN_SRC}
            ${CRUNNER_UTILS_SRC}
            ${BBHW_MEM_C}
        COMMENT "Linking ${LINUX_BIN}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )

    add_custom_target(
        ${BAREMETAL_TARGET}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
    )
    add_custom_target(
        ${LINUX_TARGET}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
    )
    add_custom_target(
        ${GROUP_TARGET}
        DEPENDS ${BAREMETAL_TARGET} ${LINUX_TARGET}
    )

    if(DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
        add_dependencies(${BUCKYBALL_MLIR_GROUP_TARGET} ${GROUP_TARGET})
    endif()
endfunction()

add_custom_target(balls-mlir-tests-build)
add_custom_target(MLIRTest-all ALL)

add_custom_target(design-mlirtests-build)

add_custom_target(
    mlirtest-workloads-build
    DEPENDS MLIRTest-all design-mlirtests-build
    COMMENT "Building MLIRTest workloads"
)

add_custom_target(
    sync-mlirtest-bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src
    COMMAND
        rsync -a --delete --exclude=CMakeFiles "--include=*/"
        "--include=*-mlirtest-*-baremetal" "--include=*-mlirtest-*-linux"
        "--exclude=*" ${CMAKE_BINARY_DIR}/src/ ${OUTPUT_BIN_DIR}/src/
    DEPENDS mlirtest-workloads-build
    COMMENT "Syncing registered MLIRTest binaries"
)
