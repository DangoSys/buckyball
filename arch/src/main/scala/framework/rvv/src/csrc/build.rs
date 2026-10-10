use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(8)
        .unwrap()
        .to_path_buf();
    let ip = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).parent().unwrap().parent().unwrap().to_path_buf();
    let parameters = ip.join("src/main/resources/rvv.json");
    println!("cargo:rerun-if-changed={}", parameters.display());
    let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(parameters).unwrap()).unwrap();
    let value = |key: &str| config[key].as_u64().unwrap() as usize;
    let constants = format!("const VLEN: usize = {};\nconst ELEN: usize = {};\nconst IBUF_BYTES: usize = {};\n",
        value("vLen"), value("eLen"), value("iBufWords") * 4);
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("params.rs"), constants).unwrap();
    let rtl = ip.join("build/KernelEngine");
    fs::create_dir_all(&rtl).unwrap();
    fs::write(rtl.join("rvv_config.svh"), format!("`ifndef RVV_MEMORY_PORTS\n`define RVV_MEMORY_PORTS {}\n`endif\n", value("memoryPorts"))).unwrap();
    let mut connections = String::new();
    for port in 0..value("memoryPorts") {
        for (direction, fields) in [
            ("Read", vec![("bank_id", "read_bank"), ("group_id", "read_group"), ("io_req_valid", "read_valid"), ("io_req_ready", "read_ready"), ("io_req_bits_addr", "read_row"), ("io_resp_valid", "read_response_valid"), ("io_resp_ready", "read_response_ready"), ("io_resp_bits_data", "read_data")]),
            ("Write", vec![("bank_id", "write_bank"), ("group_id", "write_group"), ("io_req_valid", "write_valid"), ("io_req_ready", "write_ready"), ("io_req_bits_addr", "write_row"), ("io_resp_valid", "write_response_valid"), ("io_resp_ready", "write_response_ready"), ("io_req_bits_data", "write_data"), ("io_resp_bits_ok", "write_ok")]),
        ] {
            for (field, signal) in fields {
                connections += &format!(".io_bank{direction}_{port}_{field}(vif.{signal}[{port}]),\n");
            }
        }
        for bit in 0..16 {
            connections += &format!(".io_bankWrite_{port}_io_req_bits_mask_{bit}(vif.write_mask[{port}][{bit}]),\n");
        }
    }
    connections.truncate(connections.trim_end().len() - 1);
    fs::write(rtl.join("rvv_ports.svh"), connections).unwrap();
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
        "math/cos.cpp",
        "math/erf.cpp",
        "rope.cpp",
        "trig.cpp",
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
