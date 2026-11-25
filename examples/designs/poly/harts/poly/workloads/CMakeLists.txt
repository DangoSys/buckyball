# Thread, kernel, compiler, and service tests of the poly hart.

add_subdirectory(thread)
add_subdirectory(kernel)
set(POLY_ANT_IMAGE_DIR "${POLY_ANT_IMAGE_DIR}" PARENT_SCOPE)
set(POLY_ANT_SOURCE_DIR "${POLY_ANT_SOURCE_DIR}" PARENT_SCOPE)

add_subdirectory(compiler)

add_ctest(service_payload.cpp)

set(CTEST_PROFILE "ffn")
add_ctest(service_payload.cpp)
