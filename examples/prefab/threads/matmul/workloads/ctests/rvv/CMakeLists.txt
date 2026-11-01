# RVV CTests of the GP domain: each test is a kernel (<test>_kernel.cpp, RVV
# intrinsics, linked into an image with the layout of bbsw-thread-rvv) and a
# program that loads and runs it with the Ball ISA (<test>.cpp, embeds the
# image).
set(_BANK_NUM ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_NUM})
set(_BANK_WIDTH ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH})
set(_BANK_ENTRIES ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
if(_BANK_NUM LESS 4 OR NOT _BANK_WIDTH EQUAL 128 OR _BANK_ENTRIES LESS 32)
    message(
        FATAL_ERROR
        "RVV kernel CTests require 4 banks, 128-bit rows and 512 bytes per bank; "
        "${BUCKYBALL_CTEST_TARGET} has ${_BANK_NUM} banks, ${_BANK_WIDTH}-bit rows, ${_BANK_ENTRIES} lines"
    )
endif()
find_program(RVV_CXX clang++ PATHS "${BUCKYBALL_REPO_ROOT}/result/bin" NO_DEFAULT_PATH REQUIRED)
find_program(RVV_LD riscv64-unknown-elf-ld PATHS "${BUCKYBALL_REPO_ROOT}/result/bin" NO_DEFAULT_PATH REQUIRED)
find_program(RVV_OBJCOPY riscv64-unknown-elf-objcopy PATHS "${BUCKYBALL_REPO_ROOT}/result/bin" NO_DEFAULT_PATH REQUIRED)
set(RVV_KERNEL_FLAGS
    --target=riscv64 -march=rv64imf_zve32f_zvl128b -mabi=lp64f -mcmodel=medany
    -ffreestanding -O2 -ffp-contract=off -ffunction-sections -fdata-sections
)
# The C API of the target and the boot of its hart.
set(THREAD bbsw-thread-${BUCKYBALL_CTEST_TARGET}-baremetal)
set(CRT0 bbsw-crt0-hart${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_CORE_HART_ID}-baremetal)

# ===----------------------------------------------------------------------===
# silu
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT silu.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/silu_kernel.cpp" -o silu.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e silu
        --defsym=kernel_entry=silu $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        silu.o -o silu.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata silu.elf silu.image
    DEPENDS silu_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-silu-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/silu.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_silu")
buckyball_enforce_ctest_line_limit(silu.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} silu.cpp)
add_dependencies(${NAME}-linux rvv-silu-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} silu.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/silu.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/silu.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)

# ===----------------------------------------------------------------------===
# swiglu
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT swiglu.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/swiglu_kernel.cpp" -o swiglu.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e swiglu
        --defsym=kernel_entry=swiglu $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        swiglu.o -o swiglu.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata swiglu.elf swiglu.image
    DEPENDS swiglu_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-swiglu-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/swiglu.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_swiglu")
buckyball_enforce_ctest_line_limit(swiglu.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} swiglu.cpp)
add_dependencies(${NAME}-linux rvv-swiglu-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} swiglu.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/swiglu.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/swiglu.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)

# ===----------------------------------------------------------------------===
# snake
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT snake.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/snake_kernel.cpp" -o snake.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e snake
        --defsym=kernel_entry=snake $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        snake.o -o snake.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata snake.elf snake.image
    DEPENDS snake_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-snake-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/snake.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_snake")
buckyball_enforce_ctest_line_limit(snake.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} snake.cpp)
add_dependencies(${NAME}-linux rvv-snake-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} snake.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/snake.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/snake.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)

# ===----------------------------------------------------------------------===
# mxfp8_quant
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT mxfp8_quant.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/mxfp8_quant_kernel.cpp" -o mxfp8_quant.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e mxfp8_quant
        --defsym=kernel_entry=mxfp8_quant $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        mxfp8_quant.o -o mxfp8_quant.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata mxfp8_quant.elf mxfp8_quant.image
    DEPENDS mxfp8_quant_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-mxfp8_quant-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/mxfp8_quant.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_mxfp8_quant")
buckyball_enforce_ctest_line_limit(mxfp8_quant.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} mxfp8_quant.cpp)
add_dependencies(${NAME}-linux rvv-mxfp8_quant-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} mxfp8_quant.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/mxfp8_quant.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/mxfp8_quant.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)

# ===----------------------------------------------------------------------===
# matmul
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT matmul.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernel.cpp" -o matmul.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e matmul
        --defsym=kernel_entry=matmul $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        matmul.o -o matmul.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata matmul.elf matmul.image
    DEPENDS matmul_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-matmul-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/matmul.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_matmul")
buckyball_enforce_ctest_line_limit(matmul.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} matmul.cpp)
add_dependencies(${NAME}-linux rvv-matmul-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} matmul.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/matmul.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)

# ===----------------------------------------------------------------------===
# norm
# ===----------------------------------------------------------------------===
add_custom_command(
    OUTPUT norm.image
    COMMAND "${RVV_CXX}" ${RVV_KERNEL_FLAGS}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/norm_kernel.cpp" -o norm.o
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e norm
        --defsym=kernel_entry=norm $<TARGET_PROPERTY:bbsw-thread-rvv,INTERFACE_LINK_OPTIONS>
        norm.o -o norm.elf
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata norm.elf norm.image
    DEPENDS norm_kernel.cpp ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(rvv-norm-image DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/norm.image")
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_norm")
buckyball_enforce_ctest_line_limit(norm.cpp)
add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} norm.cpp)
add_dependencies(${NAME}-linux rvv-norm-image)
target_compile_options(${NAME}-linux PRIVATE -O2 "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${NAME}-linux PRIVATE m)
buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} norm.cpp)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND
        ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS} $<TARGET_PROPERTY:${CRT0},INTERFACE_LINK_OPTIONS>
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:${THREAD},INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        -o "${NAME}-baremetal" $<TARGET_OBJECTS:${CRT0}> -x c
        ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
        "${CMAKE_CURRENT_SOURCE_DIR}/norm.cpp" -x none $<TARGET_FILE:${THREAD}>
        $<TARGET_FILE:bbsw-device-baremetal> -lm
    DEPENDS ${DEPS} "${CMAKE_CURRENT_BINARY_DIR}/norm.image"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
)
add_dependencies(component-ctests-build ${NAME}-ctest-build)
