import os


def normalize_p2e_timescale(build_dir, logger):
    if not os.path.isdir(build_dir):
        raise FileNotFoundError(f"P2E rtl dir not found: {build_dir}")
    patched = 0
    for root, _, files in os.walk(build_dir):
        for name in files:
            if not name.endswith((".v", ".sv")):
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                content = f.read()
            if "`timescale" in content:
                continue
            with open(path, "w") as f:
                f.write("`timescale 1ns/1ps\n")
                f.write(content)
            patched += 1
    logger.info(f"Normalized P2E timescale in {patched} generated Verilog files")
    return patched
