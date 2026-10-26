add_subdirectory(CTest)
add_subdirectory(MLIRTest)
add_subdirectory(SocTest)

# ===============================================================
# Design-specific workloads
# ===============================================================
if(NOT DEFINED BUCKYBALL_DESIGN_WORKLOAD_DIR)
    message(
        FATAL_ERROR
        "BUCKYBALL_DESIGN_WORKLOAD_DIR must select a registered simulation workload entry"
    )
endif()

if(NOT EXISTS ${BUCKYBALL_DESIGN_WORKLOAD_DIR}/CMakeLists.txt)
    message(
        FATAL_ERROR
        "Buckyball design workloads do not exist: ${BUCKYBALL_DESIGN_WORKLOAD_DIR}"
    )
endif()

add_subdirectory(
    ${BUCKYBALL_DESIGN_WORKLOAD_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/CTest/designs/${BUCKYBALL_WORKLOAD_DESIGN}
)

if(NOT TARGET design-workloads-build)
    message(
        FATAL_ERROR
        "Buckyball design workloads must define target design-workloads-build: ${BUCKYBALL_WORKLOAD_DESIGN}"
    )
endif()
