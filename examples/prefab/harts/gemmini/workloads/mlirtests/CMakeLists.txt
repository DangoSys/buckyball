set(BUDDY "${DESIGN_REPO}/stack/thirdparty/buddy-mlir")
add_custom_target(gemmini-mlirtests-check
    COMMAND python3 "${BUDDY}/llvm/build/bin/llvm-lit" -v
        --param "buddy_bin=${BUDDY}/build/bin"
        --param "llvm_bin=${BUDDY}/llvm/build/bin"
        --param "output=${CMAKE_CURRENT_BINARY_DIR}/lit"
        --xunit-xml-output "${CMAKE_CURRENT_BINARY_DIR}/junit.xml"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
)
add_custom_target(sync-mlirtest-bin DEPENDS gemmini-mlirtests-check)
add_dependencies(design-workloads-build gemmini-mlirtests-check)
