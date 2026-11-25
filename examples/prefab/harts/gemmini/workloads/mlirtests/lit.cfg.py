import os
from pathlib import Path
import lit.formats

config.name = "GemminiCore MLIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = str(Path(__file__).parent)
config.test_exec_root = lit_config.params["output"]
config.environment["PATH"] = os.pathsep.join(
    [
        lit_config.params["buddy_bin"],
        lit_config.params["llvm_bin"],
        config.environment["PATH"],
    ]
)
