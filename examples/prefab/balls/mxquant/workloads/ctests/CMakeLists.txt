math(
    EXPR MXQUANT_BANK_BYTES
    "${DESIGN_PROFILES_${CTEST_PROFILE}_THREAD_MEM_BANK_ENTRIES} * ${DESIGN_PROFILES_${CTEST_PROFILE}_THREAD_MEM_BANK_WIDTH} / 8"
)
math(EXPR MXQUANT_BANK_TAIL "${MXQUANT_BANK_BYTES} % 128")
if(MXQUANT_BANK_BYTES LESS 3840 OR MXQUANT_BANK_TAIL)
    message(
        FATAL_ERROR
        "Mxquant CTests require banks >= 3840 bytes and a 32-element FP32 capacity; actual ${MXQUANT_BANK_BYTES} bytes"
    )
endif()
add_ctest(mxquant_values_test.c)
add_ctest(mxquant_capacity_test.c)
ctest_instances(INSTANCES)
foreach(INSTANCE IN LISTS INSTANCES)
    ctest_name(TEST ${INSTANCE} mxquant_values_test)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        APPEND
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../reference.h
    )
endforeach()
