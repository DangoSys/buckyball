# Poly reuses the common workload definitions, but owns its Core placement.
# Derive the type -> physical Core IDs directly from the tile TOML so adding a
# third Core type only requires editing the chip configuration.
get_filename_component(_BUCKYBALL_REPO_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(_POLY_TILE_TOML "${CMAKE_CURRENT_SOURCE_DIR}/../configs/tiles/default.toml")
file(STRINGS "${_POLY_TILE_TOML}" _POLY_CORE_LINES REGEX "^name[ \\t]*=")
set(_POLY_CORE_INDEX 0)
foreach(_POLY_CORE_LINE IN LISTS _POLY_CORE_LINES)
  string(REGEX REPLACE "^name[ \\t]*=[ \\t]*\"([^\"]+)\".*$" "\\1"
    _POLY_CORE_TYPE "${_POLY_CORE_LINE}")
  if(DEFINED _POLY_CORE_IDS_${_POLY_CORE_TYPE})
    set(_POLY_CORE_IDS_${_POLY_CORE_TYPE}
        "${_POLY_CORE_IDS_${_POLY_CORE_TYPE}};${_POLY_CORE_INDEX}")
  else()
    set(_POLY_CORE_IDS_${_POLY_CORE_TYPE} "${_POLY_CORE_INDEX}")
  endif()
  math(EXPR _POLY_CORE_INDEX "${_POLY_CORE_INDEX} + 1")
endforeach()
set(BUCKYBALL_RUSHB_CORE_TYPES)
foreach(_POLY_CORE_LINE IN LISTS _POLY_CORE_LINES)
  string(REGEX REPLACE "^name[ \\t]*=[ \\t]*\"([^\"]+)\".*$" "\\1"
    _POLY_CORE_TYPE "${_POLY_CORE_LINE}")
  list(FIND BUCKYBALL_RUSHB_CORE_TYPES "${_POLY_CORE_TYPE}" _POLY_TYPE_SEEN)
  if(_POLY_TYPE_SEEN EQUAL -1)
    list(APPEND BUCKYBALL_RUSHB_CORE_TYPES "${_POLY_CORE_TYPE}")
  endif()
endforeach()
set(BUCKYBALL_RUSHB_PLACEMENT_STRICT ON)

# Poly selects the prefill Core as the host BEMU ABI provider.  The runtime
# still reads the tile TOML and creates one worker per physical Core instance.
set(BUCKYBALL_RUSHB_BEMU_MANIFEST
    ${_BUCKYBALL_REPO_ROOT}/examples/cores/prefill/emu/Cargo.toml)
set(BUCKYBALL_RUSHB_BEMU_LIBRARY
    ${_BUCKYBALL_REPO_ROOT}/examples/cores/prefill/emu/target/release/libbemu_prefill.so)
set(BUCKYBALL_RUSHB_VERILATOR_LIBRARY
    ${_BUCKYBALL_REPO_ROOT}/bebop/target/release/deps/libbebop_verilator.so)
add_subdirectory(
  ${_BUCKYBALL_REPO_ROOT}/examples/chips/goban/workloads
  ${CMAKE_CURRENT_BINARY_DIR}/goban-workloads
)
