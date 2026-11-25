add_subdirectory(CTest)
add_subdirectory(SocTest)
# The benchmarks, built only by `bbdev bench --build`.
add_subdirectory(benchmark)

# ===============================================================
# Design-specific workloads: the workload entry of the simulation.
# It defines target design-workloads-build.
# ===============================================================
add_subdirectory(
    ${DESIGN_WORKLOADS}
    ${CMAKE_CURRENT_BINARY_DIR}/CTest/designs/${DESIGN_NAME}
)
