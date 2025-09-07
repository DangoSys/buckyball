set(OPTEST_TOY_DIR ${OPTEST_WORKLOAD_DIR}/toy)
set(OPTEST_GEMMINI_DIR ${OPTEST_WORKLOAD_DIR}/gemmini)

add_subdirectory(toy)
add_subdirectory(gemmini)

add_custom_target(OpTest-all ALL DEPENDS
  OpTest-gemmini
#   OpTest-toy
)
