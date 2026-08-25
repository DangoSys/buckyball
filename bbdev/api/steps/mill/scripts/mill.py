from __future__ import annotations

import os
import shlex


def elaborate_cmd(main: str, config: str, out: str, *, seq_mem: bool = False) -> str:
    extra = ""
    if seq_mem:
        extra = (
            " -lowering-options=disallowLocalVariables"
            f" --repl-seq-mem --repl-seq-mem-file={shlex.quote(os.path.join(out, 'mems.conf'))}"
        )
    return (
        f"mill -i __.test.runMain {main} {config} "
        "--disable-annotation-unknown --strip-debug-info -O=debug "
        f"--split-verilog -o={shlex.quote(out)}{extra}"
    )
