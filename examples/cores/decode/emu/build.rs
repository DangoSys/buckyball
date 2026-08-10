use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let build_root = manifest.join("target/release/build");
    let Ok(entries) = fs::read_dir(build_root) else { return };
    for entry in entries.flatten() {
        let lib_dir = entry.path().join("out/spike_install/lib");
        if lib_dir.join("libriscv.so").is_file() {
            println!("cargo:rustc-link-arg-cdylib=-Wl,-rpath,{}", lib_dir.display());
            break;
        }
    }
}
