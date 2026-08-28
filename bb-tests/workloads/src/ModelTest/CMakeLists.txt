set(E2E_MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/e2e)

if(NOT DEFINED MODEL OR MODEL STREQUAL "")
  return()
endif()

if(NOT DEFINED ARCH)
  set(ARCH_GEMMINI ON)
  set(ARCH_X86 ON)
  if(DEFINED BUCKYBALL_WORKLOAD_CHIP AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/e2e/models/archs/buckyball/${BUCKYBALL_WORKLOAD_CHIP}")
    set(ARCH_BUCKYBALL ON)
  endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/e2e/models/CMakeLists.txt")
  message(FATAL_ERROR "e2e models CMakeLists.txt is missing")
endif()
add_subdirectory(e2e/models)

unset(ARCH CACHE)
unset(MODEL CACHE)
