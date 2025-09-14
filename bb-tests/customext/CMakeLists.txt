project(customext)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(TOY_EXT_DIR ${CUSTOMEXT_DIR}/toy)
set(GEMMINI_EXT_DIR ${CUSTOMEXT_DIR}/gemmini)

# Check if RISCV environment variable is set
if(NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "RISCV is unset")
else()
  message(STATUS "Running with RISCV=$ENV{RISCV}")
  set(RISCV_PATH $ENV{RISCV})
endif()

# Add subdirectories
add_subdirectory(toy)
add_subdirectory(gemmini)
