use bebop_bemu::{device::cbus, sim::run::Args, DeviceModel};
use clap::Parser;
use std::{path::{Path, PathBuf}, process::Command};

fn run(command: &mut Command) {
    assert!(command.status().expect("execute verification tool").success());
}

#[test]
fn independent_devices_keep_gemmini_configuration_and_memory_private() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.ancestors().nth(6).expect("simulation entry location");
    let output = manifest.join("target/verification/instance-isolation");
    std::fs::create_dir_all(&output).unwrap();
    let isa = output.join("isa");
    run(Command::new("python3").arg(repo.join("stack/scripts/design.py"))
        .arg("--design-pb").arg(repo.join("examples/designs/toy/output/generated/design.pb"))
        .arg("--target").arg("device")
        .arg("--out").arg(&output));
    let bare = repo.join("stack/os/baremetal");
    bebop_bemu::configs::load(simulation::SIMULATION.design);
    let mut cpus = Vec::new();
    for name in ["gemmini_loop_subrob", "gemmini_loop_mmio"] {
        let elf = output.join(format!("{name}.elf"));
        run(Command::new("riscv64-none-elf-gcc")
            .args(["-nostdlib", "-nostartfiles", "-O2", "-march=rv64gc", "-mabi=lp64d",
                   "-mcmodel=medany", "-msmall-data-limit=0", "-fno-builtin", "-Wl,--no-relax"])
            .arg("-T").arg(bare.join("bbsim.ld"))
            .arg("-I").arg(isa.join("toy"))
            .arg("-I").arg(repo.join("bb-tests/workloads/lib"))
            .arg("-I").arg(repo.join("examples/components/balls/gemmini/workloads"))
            .arg(bare.join("crt0.S"))
            .arg(repo.join(format!("examples/components/balls/gemmini/workloads/ctests/{name}.c")))
            .arg("-o").arg(&elf));
        let log = output.join(name);
        let args = Args::try_parse_from([Path::new("test"), Path::new("--elf"), &elf,
                                       Path::new("--log-dir"), &log]).unwrap();
        let device = DeviceModel::new(512 * 1024 * 1024, bebop_bemu::hart_capacity());
        let mut instances = cbus::instantiate(&device, &args).unwrap();
        assert_eq!(instances.len(), 1);
        let mut cpu = instances.pop().unwrap();
        cpu.cpu_mut().load_elf(&elf).unwrap();
        cpu.cpu_mut().init_firmware(None, None).unwrap();
        cpus.push(cpu);
    }
    for _ in 0..2_000_000 {
        if cpus.iter().all(|cpu| cpu.cpu().finished()) { break; }
        for cpu in &mut cpus { if !cpu.cpu().finished() { cpu.step(1).unwrap(); } }
    }
    for cpu in cpus { assert_eq!(cpu.cpu().exit_code(), Some(0)); }
}
