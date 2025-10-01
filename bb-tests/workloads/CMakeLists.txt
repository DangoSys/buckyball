cmake_minimum_required(VERSION 3.12)
project(BuckyballTest C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
set(BUDDY_MLIR_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/../)
set(BUDDY_BINARY_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)

set(ELF_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc)
set(LINUX_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)

set(BUILD_BIN_DIR ${WORKLOAD_DIR}/../build/workloads)
set(OUTPUT_BIN_DIR ${WORKLOAD_DIR}/../output/workloads)
#-------------------------------------------------------------------------------
# add_subdirectory
#-------------------------------------------------------------------------------
add_subdirectory(lib)
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


add_dependencies(sync-bin
  tutorial-build
	buckyball-CTest-build
	OpTest-all
  # sparse-build
  # ModelTest
	)

add_custom_target(build-all ALL
  DEPENDS sync-bin
)

add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND rm -rf ${OUTPUT_BIN_DIR}
  COMMENT "Cleaning all build artifacts"
)
