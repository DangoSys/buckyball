add_subdirectory(CTest)
add_subdirectory(MLIRTest)
add_subdirectory(SocTest)

# ===============================================================
# Design-specific workloads: the workload entry of the simulation.
# It defines target design-workloads-build.
# ===============================================================
add_subdirectory(
    ${DESIGN_WORKLOADS}
    ${CMAKE_CURRENT_BINARY_DIR}/CTest/designs/${DESIGN_NAME}
)
