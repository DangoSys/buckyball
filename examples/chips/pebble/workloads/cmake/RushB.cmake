# Pebble's host-native rushB backends. Model CMake files consume these names
# through the chip-neutral BUCKYBALL_RUSHB_* interface.
set(BUCKYBALL_RUSHB_BEMU_MANIFEST
    ${BUCKYBALL_REPO_ROOT}/examples/chips/pebble/emu/Cargo.toml)
set(BUCKYBALL_RUSHB_BEMU_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/examples/chips/pebble/emu/target/release/libbebop_bemu.so)
set(BUCKYBALL_RUSHB_VERILATOR_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/bebop/target/release/deps/libbebop_verilator.so)
