set(BUCKYBALL_MLIR_TEST_PREFIX bank)
add_buckyball_mlir_test(mxquant TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
set(TEST_NAME
    "${BUCKYBALL_WORKLOAD_DESIGN}-${BUCKYBALL_MLIR_ACTIVE_TARGET}-mlirtest-bank_mxquant"
)
foreach(KIND baremetal linux)
    add_custom_command(
        OUTPUT ${TEST_NAME}-${KIND}
        APPEND
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../reference.h
    )
endforeach()
