set(_params "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET}/params.h")
file(STRINGS "${_params}" _num_define REGEX "^#define BANK_NUM ")
file(STRINGS "${_params}" _lines_define REGEX "^#define BANK_LINES ")
file(STRINGS "${_params}" _width_define REGEX "^#define BANK_WIDTH ")
string(REGEX REPLACE ".* " "" _num "${_num_define}")
string(REGEX REPLACE ".* " "" _lines "${_lines_define}")
string(REGEX REPLACE ".* " "" _width "${_width_define}")
if(NOT _width EQUAL 128 OR _lines LESS 16 OR _num LESS 6)
  message(FATAL_ERROR
    "VecBall MLIR tests require BANK_WIDTH=128, BANK_LINES>=16, and BANK_NUM>=6 "
    "(got width=${_width}, lines=${_lines}, num=${_num})")
endif()

add_subdirectory(ball)
add_subdirectory(bank)
