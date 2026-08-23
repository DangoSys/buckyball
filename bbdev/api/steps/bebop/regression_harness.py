import shlex


def nextest_harness_args(
    workload_toml: str,
    bb_tests_root: str,
    *,
    p2e_bitstream: str | None = None,
) -> str:
    args = [
        "--",
        "--workload-toml",
        shlex.quote(workload_toml),
        "--bb-tests-root",
        shlex.quote(bb_tests_root),
    ]
    if p2e_bitstream:
        args.extend(["--p2e-bitstream", shlex.quote(p2e_bitstream)])
    return " ".join(args)
