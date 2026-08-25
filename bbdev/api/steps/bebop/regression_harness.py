def nextest_harness_args(
    workload_toml: str,
    bb_tests_root: str,
    env: dict,
    *,
    p2e_bitstream: str | None = None,
) -> str:
    env["BEBOP_WORKLOAD_TOML"] = workload_toml
    env["BEBOP_BB_TESTS_ROOT"] = bb_tests_root
    if p2e_bitstream:
        env["BEBOP_P2E_BITSTREAM"] = p2e_bitstream
    return ""
