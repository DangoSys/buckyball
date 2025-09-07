cmake_minimum_required(VERSION 3.12)
project(BuckyballTest C CXX)

# get the root directory of the project
# execute_process(
#   COMMAND git rev-parse --show-toplevel
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#   OUTPUT_VARIABLE BBDIR
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )

set(WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)

set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})
set(BUDDY_MLIR_BIN_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/bin)
set(BUDDY_OPT ${BUDDY_MLIR_BIN_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_MLIR_BIN_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_MLIR_BIN_DIR}/buddy-llc)


# 为 Linux 平台的代码
set(CMAKE_C_COMPILER_LINUX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER_LINUX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

# 为 baremetal 平台的代码
set(CMAKE_C_COMPILER_BAREMETAL ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER_BAREMETAL ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-g++)

set(BUILD_BIN_DIR "${WORKLOAD_DIR}/build")
set(OUTPUT_BIN_DIR "${WORKLOAD_DIR}/output/workloads")
#-------------------------------------------------------------------------------
# add_subdirectory
#-------------------------------------------------------------------------------
# add_subdirectory(lib)
add_subdirectory(src)

#-------------------------------------------------------------------------------
# setup directory
#-------------------------------------------------------------------------------
# cmake 时候就会创建
file(MAKE_DIRECTORY ${OUTPUT_BIN_DIR})

#-------------------------------------------------------------------------------
# set custom target
#-------------------------------------------------------------------------------
add_custom_target(create-dirs
  COMMAND mkdir -p ${OUTPUT_BIN_DIR}
  COMMENT "Creating necessary directories"
)


# add_custom_target(build-compiler
#   COMMAND ${BBDIR}/bb-tests/scripts/build-compiler.sh
#   COMMENT "Building Buddy Compiler"
# )

add_custom_target(sync-bin
  COMMAND rsync -av --exclude="CMakeFiles" 
          --include="*/" --include="*-baremetal" --include="*-linux" 
          --exclude="*" 
          ${BUILD_BIN_DIR}/ ${OUTPUT_BIN_DIR}/
  COMMENT "Syncing binary files to output directory"
)

# 设置执行顺序：build-compiler 必须先于 buckyball-OpTest 完成，sync-bin 必须最后执行
# add_dependencies(gemmini-native create-dirs)
# add_dependencies(buckyball-OpTest build-compiler) # 属于 npu-build 的子目录，由于依赖需要直接指定

add_dependencies(sync-bin 
  tutorial-build
	buckyball-CTest-build
	OpTest-all
	)
  
add_custom_target(build-all ALL 
  DEPENDS sync-bin
)

add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND rm -rf ${OUTPUT_BIN_DIR}
  COMMENT "Cleaning all build artifacts"
)
