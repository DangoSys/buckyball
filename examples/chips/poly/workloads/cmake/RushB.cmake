# Poly owns a heterogeneous tile but uses the common RushB host ABI.
set(BUCKYBALL_RUSHB_BEMU_MANIFEST
    ${BUCKYBALL_REPO_ROOT}/examples/cores/prefill/emu/Cargo.toml)
set(BUCKYBALL_RUSHB_BEMU_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/examples/cores/prefill/emu/target/release/libbemu_prefill.so)
set(BUCKYBALL_RUSHB_VERILATOR_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/bebop/target/release/deps/libbebop_verilator.so)

# Build a type -> physical Core-ID map from the tile configuration.  The
# workload placement code consumes these variables for typed subgraphs.
set(_POLY_TILE_TOML
    ${BUCKYBALL_REPO_ROOT}/examples/chips/poly/configs/tiles/default.toml)
file(STRINGS ${_POLY_TILE_TOML} _POLY_CORE_LINES REGEX "^name[ \t]*=")
set(_POLY_CORE_INDEX 0)
foreach(_POLY_CORE_LINE IN LISTS _POLY_CORE_LINES)
  string(REGEX REPLACE "^name[ \t]*=[ \t]*\"([^\"]+)\".*$" "\\1"
    _POLY_CORE_TYPE "${_POLY_CORE_LINE}")
  if(DEFINED _POLY_CORE_IDS_${_POLY_CORE_TYPE})
    set(_POLY_CORE_IDS_${_POLY_CORE_TYPE}
        "${_POLY_CORE_IDS_${_POLY_CORE_TYPE}};${_POLY_CORE_INDEX}")
  else()
    set(_POLY_CORE_IDS_${_POLY_CORE_TYPE} "${_POLY_CORE_INDEX}")
  endif()
  math(EXPR _POLY_CORE_INDEX "${_POLY_CORE_INDEX} + 1")
endforeach()
set(BUCKYBALL_RUSHB_PLACEMENT_STRICT ON)
