use std::{env, path::PathBuf, process::Command};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(8)
        .unwrap()
        .to_path_buf();
    let source = root.join("bb-tests/workloads/lib/bbsw/kernels/rvv");
    for name in [
        "build_images.py",
        "kernel.ld",
        "kernels.h",
        "silu.cpp",
        "snake.cpp",
        "quant.cpp",
        "math/exp.cpp",
        "math/sin.cpp",
        "math/math.h",
        "math/constants.h",
    ] {
        println!("cargo:rerun-if-changed={}", source.join(name).display());
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernels");
    assert!(Command::new("python3")
        .arg(source.join("build_images.py"))
        .arg("--output")
        .arg(output)
        .status()
        .unwrap()
        .success());
}
