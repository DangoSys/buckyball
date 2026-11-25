# ===============================================================
# C tests. A test is written for a profile (CTEST_PROFILE) and built for each
# instance of it, each its own top: every hart whose CPU has the profile
# (<design>-hart<c>-ctest-<test>), and every thread of the profile, through
# its hart (<design>-hart<c>-thread<t>-ctest-<test>).
# ===============================================================
function(ctest_line_limit SOURCE_FILE)
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

# The instances of CTEST_PROFILE: "<hart>" for a hart whose CPU has it,
# "<hart>/<thread>" for a thread of it.
function(ctest_instances OUT)
    if(NOT CTEST_PROFILE IN_LIST DESIGN_PROFILES)
        message(FATAL_ERROR "unknown C test profile: ${CTEST_PROFILE}")
    endif()
    set(INSTANCES)
    foreach(HART IN LISTS DESIGN_DEVICE_HARTS)
        if(DESIGN_DEVICE_HARTS_${HART}_PROFILE STREQUAL CTEST_PROFILE)
            list(APPEND INSTANCES ${HART})
        endif()
        foreach(THREAD IN LISTS DESIGN_DEVICE_HARTS_${HART}_THREAD_INDICES)
            if(DESIGN_THREADS_${THREAD}_PROFILE STREQUAL CTEST_PROFILE)
                list(APPEND INSTANCES ${HART}/${THREAD})
            endif()
        endforeach()
    endforeach()
    set(${OUT} ${INSTANCES} PARENT_SCOPE)
endfunction()

# The name of test STEM of INSTANCE (of ctest_instances).
function(ctest_name OUT INSTANCE STEM)
    # A thread is named by its id within its hart.
    string(REPLACE "/" ";" PARTS ${INSTANCE})
    list(GET PARTS 0 HART)
    set(PATH ${HART})
    list(LENGTH PARTS LENGTH)
    if(LENGTH EQUAL 2)
        list(GET PARTS 1 THREAD)
        set(PATH "${HART}-thread${DESIGN_THREADS_${THREAD}_LOCAL_ID}")
    endif()
    set(${OUT} "${DESIGN_NAME}-hart${PATH}-ctest-${STEM}" PARENT_SCOPE)
endfunction()

function(ctest_deps OUT_DEPS SOURCE_DIR SOURCE_FILE HART)
    set(DEPS
        ${SOURCE_DIR}/${SOURCE_FILE}
        $<TARGET_FILE:bbsw-thread-${CTEST_PROFILE}-baremetal>
        $<TARGET_FILE:bbsw-hart-baremetal>
        $<TARGET_FILE:bbsw-device-baremetal>
        $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
        ${BUCKYBALL_DESIGN_COMMON_SOURCES}
        ${BUCKYBALL_DESIGN_COMMON_HEADERS}
        ${BUCKYBALL_BALL_ISA_DEPS}
    )
    file(GLOB _LOCAL_HDRS CONFIGURE_DEPENDS ${SOURCE_DIR}/*.h)
    list(APPEND DEPS ${_LOCAL_HDRS})

    set(${OUT_DEPS} ${DEPS} PARENT_SCOPE)
endfunction()

# A C test of a hart for Linux: the C API of the profile, the hart and the
# device.
function(add_linux_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
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
            bbsw-thread-${CTEST_PROFILE}-linux
            bbsw-hart-linux
            bbsw-device-linux
    )
    set_target_properties(${EXECUTABLE} PROPERTIES LINKER_LANGUAGE C)

    add_custom_target(${TEST_NAME}-linux-build DEPENDS ${EXECUTABLE})
endfunction()

# A C test booted by crt0 of hart HART: the C API of the profile, the hart
# and the device. A test of THREAD (else empty) issues its Ball instructions
# through the kernel's stream to that thread (bb_thread_run), whose loader
# runs the stream interpreter on it.
function(add_baremetal_ctest TEST_NAME SOURCE_DIR SOURCE_FILE HART THREAD)
    set(EXECUTABLE "${TEST_NAME}-baremetal")
    ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE} ${HART})

    set(BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS)
    foreach(DESIGN_COMMON_INCLUDE_DIR ${BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS})
        list(
            APPEND BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS
            -I${DESIGN_COMMON_INCLUDE_DIR}
        )
    endforeach()
    set(LIBRARY bbsw-thread-${CTEST_PROFILE}-baremetal)
    set(CRT0 bbsw-crt0-hart${HART}-baremetal)
    set(STREAM_FLAGS)
    set(STREAM_LINK)
    if(NOT THREAD STREQUAL "")
        set(STREAM_FLAGS
            -DBUCKYBALL_STREAM
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREAD}_SIGNATURE}ULL
            -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include)
        # The kernel binds every thread of the hart: one loader per kind.
        set(LOADERS)
        foreach(INDEX IN LISTS DESIGN_DEVICE_HARTS_${HART}_THREAD_INDICES)
            list(APPEND LOADERS ${DESIGN_THREAD_BUILD_${DESIGN_THREADS_${INDEX}_PROFILE}}/loader.o)
        endforeach()
        list(REMOVE_DUPLICATES LOADERS)
        set(STREAM_LINK ${LOADERS} $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>)
        list(APPEND TEST_DEPS ${LOADERS} runtime-baremetal kernel)
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${CTEST_C_FLAGS}
            $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS> -o ${EXECUTABLE}
            $<TARGET_OBJECTS:${CRT0}> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${BUCKYBALL_DESIGN_COMMON_SOURCES} ${SOURCE_DIR}/${SOURCE_FILE}
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${LIBRARY},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${LIBRARY},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            ${BUCKYBALL_DESIGN_COMMON_INCLUDE_FLAGS} -I${SOURCE_DIR} ${STREAM_FLAGS}
            $<TARGET_FILE:${LIBRARY}> $<TARGET_FILE:bbsw-hart-baremetal>
            $<TARGET_FILE:bbsw-device-baremetal> ${STREAM_LINK}
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

# The test SOURCE_FILE of CTEST_PROFILE, for every instance of it. A thread
# is the kernel's, not the OS's: its tests are baremetal.
function(add_ctest SOURCE_FILE)
    get_filename_component(STEM ${SOURCE_FILE} NAME_WE)
    ctest_line_limit(${SOURCE_FILE})
    ctest_instances(INSTANCES)
    foreach(INSTANCE IN LISTS INSTANCES)
        ctest_name(TEST_NAME ${INSTANCE} ${STEM})
        string(REPLACE "/" ";" PARTS ${INSTANCE})
        list(GET PARTS 0 HART)
        list(LENGTH PARTS LENGTH)
        if(LENGTH EQUAL 2)
            list(GET PARTS 1 THREAD)
            add_baremetal_ctest(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILE} ${HART} ${THREAD})
            add_custom_target(${TEST_NAME}-ctest-build DEPENDS ${TEST_NAME}-baremetal-build)
        else()
            add_baremetal_ctest(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILE} ${HART} "")
            add_linux_ctest(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILE})
            add_custom_target(${TEST_NAME}-ctest-build
                DEPENDS ${TEST_NAME}-linux-build ${TEST_NAME}-baremetal-build)
        endif()
        if(DEFINED CTEST_GROUP)
            add_dependencies(${CTEST_GROUP} ${TEST_NAME}-ctest-build)
        endif()
    endforeach()
endfunction()

# The C tests of the Balls of every profile.
function(add_ball_ctests)
    foreach(CTEST_PROFILE IN LISTS DESIGN_PROFILES)
        foreach(DIR IN LISTS DESIGN_PROFILES_${CTEST_PROFILE}_BALL_CTEST_DIRS)
            cmake_path(GET DIR PARENT_PATH WORKLOADS)
            cmake_path(GET WORKLOADS PARENT_PATH ROOT)
            cmake_path(GET ROOT FILENAME BALL)
            add_subdirectory("${BUCKYBALL_REPO_ROOT}/${DIR}"
                "${CMAKE_CURRENT_BINARY_DIR}/balls/${CTEST_PROFILE}/${BALL}")
        endforeach()
    endforeach()
endfunction()

add_custom_target(component-ctests-build)

function(add_component_workload NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    if(
        NOT DEFINED CTEST_PROFILE
        OR CTEST_PROFILE STREQUAL ""
        OR NOT ARG_SOURCES
    )
        message(
            FATAL_ERROR
            "Workload target and sources are required for ${NAME}"
        )
    endif()
    list(FIND DESIGN_PROFILES "${CTEST_PROFILE}" _profile)
    set(_test
        "${DESIGN_NAME}-${CTEST_PROFILE}-ctest-pipeline-${NAME}"
    )
    set(_elf "${_test}-multicore-baremetal")
    set(_sources)
    foreach(_source IN LISTS ARG_SOURCES)
        ctest_line_limit("${_source}")
        list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    endforeach()
    set(_thread bbsw-thread-${CTEST_PROFILE}-baremetal)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
        COMMAND
            ${ELF_CXX} ${CTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
            -DBB_COMPONENT_WORKLOAD_PROFILE=${_profile}
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${_thread},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${_thread},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
            -o ${_elf} $<TARGET_OBJECTS:bbsw-start-baremetal> ${_sources}
            $<TARGET_FILE:${_thread}> $<TARGET_FILE:bbsw-hart-baremetal>
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
    add_dependencies(${CTEST_GROUP} ${_test}-thread-build)
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
