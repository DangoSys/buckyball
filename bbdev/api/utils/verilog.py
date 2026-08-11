"""Shared RTL elaboration commands for synthesis-oriented flows."""


def seq_mem_elaboration_command(elaborate_config: str, build_dir: str, mem_conf: str) -> str:
    """Emit split Verilog with behavioral memories replaced by named modules."""
    return (
        f"mill -i __.test.runMain sims.verilator.Elaborate {elaborate_config} "
        "--disable-annotation-unknown --strip-debug-info -O=debug "
        "-lowering-options=disallowLocalVariables "
        f"--repl-seq-mem --repl-seq-mem-file={mem_conf} "
        f"--split-verilog -o={build_dir}"
    )
