# ===============================================================
# Generic functions to help add CTest workloads
# ===============================================================
function(buckyball_enforce_ctest_line_limit SOURCE_FILE)
    if(IS_ABSOLUTE "${SOURCE_FILE}")
        set(_src "${SOURCE_FILE}")
    else()
        set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    endif()
    if(NOT EXISTS "${_src}")
        message(FATAL_ERROR "ctest source does not exist: ${_src}")
    endif()

    file(READ "${_src}" _content)
    if(_content STREQUAL "")
        set(_n 0)
    else()
        string(REGEX MATCHALL "\n" _newlines "${_content}")
        list(LENGTH _newlines _n)
        string(REGEX MATCH "[^\n]$" _no_trailing_nl "${_content}")
        if(_no_trailing_nl)
            math(EXPR _n "${_n} + 1")
        endif()
    endif()

    if(_n GREATER CTEST_MAX_LINES)
        message(
            FATAL_ERROR
            "ctest exceeds ${CTEST_MAX_LINES} lines: ${_src} (${_n} lines). "
            "Split it into smaller focused ctests. "
            "Do not evade this check by moving functional code into headers."
        )
    endif()
endfunction()

function(buckyball_ctest_deps OUT_DEPS SOURCE_DIR SOURCE_FILE)
    if(NOT DEFINED BUCKYBALL_CTEST_TARGET OR BUCKYBALL_CTEST_TARGET STREQUAL "")
        message(FATAL_ERROR "BUCKYBALL_CTEST_TARGET is required for C tests")
    endif()
    list(
        FIND DESIGN_PROFILES
        "${BUCKYBALL_CTEST_TARGET}"
        _buckyball_target_index
    )
    if(_buckyball_target_index EQUAL -1)
        message(FATAL_ERROR "unknown C test target: ${BUCKYBALL_CTEST_TARGET}")
    endif()
    set(DEPS
        ${SOURCE_DIR}/${SOURCE_FILE}
        $<TARGET_FILE:bbsw-thread-${BUCKYBALL_CTEST_TARGET}-baremetal>
        $<TARGET_FILE:bbsw-core-baremetal>
        $<TARGET_FILE:bbsw-device-baremetal>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_CORE_HART_ID}-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
        ${BUCKYBALL_DESIGN_COMMON_SOURCES}
        ${BUCKYBALL_DESIGN_COMMON_HEADERS}
        ${BUCKYBALL_BALL_ISA_DEPS}
    )
    file(GLOB _LOCAL_HDRS CONFIGURE_DEPENDS ${SOURCE_DIR}/*.h)
    list(APPEND DEPS ${_LOCAL_HDRS})

    set(${OUT_DEPS} ${DEPS} PARENT_SCOPE)
endfunction()

# A C test for Linux: the C API of the target thread, the core and the device.
function(add_buckyball_linux_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
    set(EXECUTABLE "${TEST_NAME}-linux")

    add_executable(
        ${EXECUTABLE}
        ${SOURCE_DIR}/${SOURCE_FILE}
        ${BUCKYBALL_DESIGN_COMMON_SOURCES}
    )
    target_include_directories(
        ${EXECUTABLE}
        PRIVATE ${BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS} ${SOURCE_DIR}
    )
    target_link_libraries(
        ${EXECUTABLE}
        PRIVATE
            bbsw-thread-${BUCKYBALL_CTEST_TARGET}-linux
            bbsw-core-linux
            bbsw-device-linux
    )
    set_target_properties(${EXECUTABLE} PROPERTIES LINKER_LANGUAGE C)

    add_custom_target(${TEST_NAME}-linux-build DEPENDS ${EXECUTABLE})
endfunction()

# A C test on the hart of the target: the C API of the target thread, the
# core and the device, booted by crt0 of that hart.
function(add_buckyball_baremetal_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
    set(EXECUTABLE "${TEST_NAME}-baremetal")
    buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

    set(BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS)
    foreach(DESIGN_COMMON_INCLUDE_DIR ${BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS})
        list(
            APPEND BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS
            -I${DESIGN_COMMON_INCLUDE_DIR}
        )
    endforeach()
    set(THREAD bbsw-thread-${BUCKYBALL_CTEST_TARGET}-baremetal)
    set(CRT0 bbsw-crt0-hart${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_CORE_HART_ID}-baremetal)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS> -o ${EXECUTABLE}
            $<TARGET_OBJECTS:${CRT0}> ${BUCKYBALL_DESIGN_COMMON_SOURCES}
            ${SOURCE_DIR}/${SOURCE_FILE}
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            ${BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS} -I${SOURCE_DIR}
            $<TARGET_FILE:${THREAD}> $<TARGET_FILE:bbsw-core-baremetal>
            $<TARGET_FILE:bbsw-device-baremetal>
        DEPENDS ${TEST_DEPS}
        COMMENT "Building baremetal executable: ${EXECUTABLE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    add_custom_target(
        ${TEST_NAME}-baremetal-build
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    )
endfunction()

function(add_buckyball_ctest SOURCE_FILE)
    get_filename_component(_stem ${SOURCE_FILE} NAME_WE)
    if(NOT DEFINED BUCKYBALL_CTEST_TARGET OR BUCKYBALL_CTEST_TARGET STREQUAL "")
        message(
            FATAL_ERROR
            "BUCKYBALL_CTEST_TARGET is required for ${SOURCE_FILE}"
        )
    endif()
    set(TEST_NAME
        "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-${_stem}"
    )
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    buckyball_enforce_ctest_line_limit(${SOURCE_FILE})

    add_buckyball_linux_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
    add_buckyball_baremetal_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})

    add_custom_target(
        ${TEST_NAME}-ctest-build
        DEPENDS ${TEST_NAME}-linux-build ${TEST_NAME}-baremetal-build
        COMMENT "Building CTest workload ${TEST_NAME}"
    )

    if(DEFINED BUCKYBALL_CTEST_GROUP_TARGET)
        add_dependencies(
            ${BUCKYBALL_CTEST_GROUP_TARGET}
            ${TEST_NAME}-ctest-build
        )
    endif()
endfunction()

add_custom_target(component-ctests-build)

function(add_buckyball_component_workload NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    if(
        NOT DEFINED BUCKYBALL_CTEST_TARGET
        OR BUCKYBALL_CTEST_TARGET STREQUAL ""
        OR NOT ARG_SOURCES
    )
        message(
            FATAL_ERROR
            "Workload target and sources are required for ${NAME}"
        )
    endif()
    list(FIND DESIGN_PROFILES "${BUCKYBALL_CTEST_TARGET}" _profile)
    set(_test
        "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-pipeline-${NAME}"
    )
    set(_elf "${_test}-multicore-baremetal")
    set(_sources)
    foreach(_source IN LISTS ARG_SOURCES)
        buckyball_enforce_ctest_line_limit("${_source}")
        list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    endforeach()
    set(_thread bbsw-thread-${BUCKYBALL_CTEST_TARGET}-baremetal)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
        COMMAND
            ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
            -DBB_COMPONENT_WORKLOAD_PROFILE=${_profile}
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${_thread},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${_thread},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            -o ${_elf} $<TARGET_OBJECTS:bbsw-start-baremetal> ${_sources}
            $<TARGET_FILE:${_thread}> $<TARGET_FILE:bbsw-core-baremetal>
            $<TARGET_FILE:bbsw-device-baremetal>
        DEPENDS
            ${_sources}
            ${BUCKYBALL_BBSW_HEADERS}
            ${BUCKYBALL_BALL_ISA_DEPS}
            $<TARGET_FILE:${_thread}>
            $<TARGET_OBJECTS:bbsw-start-baremetal>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(
        ${_test}-thread-build
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
    )
    add_dependencies(${BUCKYBALL_CTEST_GROUP_TARGET} ${_test}-thread-build)
endfunction()

# ===============================================================
# Common workloads for all designs
# ===============================================================
add_custom_target(
    ctest-workloads-build
    DEPENDS design-ctests-build
    COMMENT "Building registered CTest workloads"
)

add_custom_target(
    sync-ctest-bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/CTest
    COMMAND
        rsync -a --delete --exclude=CMakeFiles "--exclude=*/mlir_tests/**"
        "--include=*/" "--include=*-baremetal" "--include=*-linux"
        "--include=*-baremetal.expect" "--exclude=*"
        ${CMAKE_CURRENT_BINARY_DIR}/ ${OUTPUT_BIN_DIR}/src/CTest/
    DEPENDS ctest-workloads-build
    COMMENT "Syncing CTest binaries to output directory"
)

add_custom_target(
    sync-native-bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/CTest
    COMMAND
        rsync -a --delete --exclude=CMakeFiles "--exclude=mlir_tests"
        "--include=*/" "--include=*-baremetal" "--include=*-baremetal.expect"
        "--include=*-linux" "--include=*-run" "--exclude=*"
        ${CMAKE_CURRENT_BINARY_DIR}/ ${OUTPUT_BIN_DIR}/src/CTest/
    DEPENDS ctest-workloads-build
    COMMENT "Syncing CTest and benchmark binaries"
)
