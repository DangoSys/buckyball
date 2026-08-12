# Goban's host-native rushB backend is owned by the Goban Core. The chip layer
# supplies the selected Core runtime to model workload CMake files.
set(BUCKYBALL_RUSHB_BEMU_MANIFEST
    ${BUCKYBALL_REPO_ROOT}/examples/cores/goban/emu/Cargo.toml)
set(BUCKYBALL_RUSHB_BEMU_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/examples/cores/goban/emu/target/release/libbebop_bemu.so)
set(BUCKYBALL_RUSHB_VERILATOR_LIBRARY
    ${BUCKYBALL_REPO_ROOT}/bebop/target/release/deps/libbebop_verilator.so)
