# Shared by all chips (and ball ctests pulled in via chip workloads).
set(BUCKYBALL_CTEST_MAX_LINES 100)

function(buckyball_enforce_ctest_line_limit SOURCE_FILE)
  if(IS_ABSOLUTE "${SOURCE_FILE}")
    set(_src "${SOURCE_FILE}")
  else()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
  endif()
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "ctest source does not exist: ${_src}")
  endif()

  file(READ "${_src}" _content)
  if(_content STREQUAL "")
    set(_n 0)
  else()
    string(REGEX MATCHALL "\n" _newlines "${_content}")
    list(LENGTH _newlines _n)
    string(REGEX MATCH "[^\n]$" _no_trailing_nl "${_content}")
    if(_no_trailing_nl)
      math(EXPR _n "${_n} + 1")
    endif()
  endif()

  if(_n GREATER BUCKYBALL_CTEST_MAX_LINES)
    message(FATAL_ERROR
      "ctest exceeds ${BUCKYBALL_CTEST_MAX_LINES} lines: ${_src} (${_n} lines). "
      "Split it into smaller focused ctests. "
      "Do not evade this check by moving functional code into headers.")
  endif()
endfunction()
