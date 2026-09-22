use bebop_bemu::{tile_topology, BemuInstance, SharedMemory, TraceConfig};
use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DRAM_SIZE: usize = 1 << 30;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 0)]
    tile_index: usize,
    #[arg(long)]
    elf: PathBuf,
    #[arg(long)]
    log_dir: PathBuf,
    #[arg(long)]
    pk: bool,
    #[arg(long)]
    disasm: bool,
    #[arg(long = "tool-profile")]
    tool_profile: bool,
    #[arg(long)]
    itrace: bool,
    #[arg(long)]
    mtrace: bool,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let elf = absolute(&args.elf)?;
    let log_dir = absolute(&args.log_dir)?;
    let topology = tile_topology(args.tile_index);
    eprintln!(
        "[INFO] Goban Chip BEMU: tile_index={} cores={} shared_dram={}MiB",
        args.tile_index,
        topology.cores.len(),
        DRAM_SIZE >> 20
    );

    let core_count = topology.cores.len();
    let virtual_bank_count = topology.virtual_bank_count;
    let memory = SharedMemory::new(
        DRAM_SIZE,
        core_count,
        topology.shared_physical_bank_count,
        topology.shared_bank_size,
        virtual_bank_count,
    );
    let mut harts = Vec::with_capacity(core_count);

    for (local_id, (core_name, core_index)) in topology.cores.into_iter().enumerate() {
        let hart_id = args.tile_index * core_count + local_id;
        let mut bemu = BemuInstance::new_with_core_hart(
            &log_dir.join(format!("hart-{hart_id}")),
            TraceConfig::new(args.itrace, args.mtrace),
            args.disasm,
            args.tool_profile,
            core_index,
            hart_id,
            Some(Arc::clone(&memory)),
            Some(virtual_bank_count),
        )
        .map_err(|error| error.to_string())?;
        bemu.load_elf(&elf).map_err(|error| error.to_string())?;
        bemu.init_hart(args.pk).map_err(|error| error.to_string())?;
        harts.push((hart_id, core_name, bemu, false));
    }

    loop {
        if let Some((hart_id, core_name, code)) = harts.iter().find_map(|(hart_id, core_name, bemu, _)| {
            bemu.finished().then(|| (*hart_id, core_name.clone(), bemu.exit_code().unwrap_or(1)))
        }) {
            for (_, _, bemu, _) in &mut harts {
                bemu.stop(code);
            }
            eprintln!("[INFO] stopped Core worker hart={hart_id} core={core_name} exit={code}");
            return if code == 0 {
                Ok(())
            } else {
                Err(format!("Core worker {hart_id}: guest exited with code {code}"))
            };
        }

        if harts.iter().all(|(_, _, _, waiting)| *waiting) {
            for (_, _, _, waiting) in &mut harts {
                *waiting = false;
            }
        }

        for (_, _, bemu, waiting) in &mut harts {
            if !*waiting {
                bemu.step(1).map_err(|error| error.to_string())?;
                *waiting = bemu.take_barrier();
            }
        }
    }
}

fn absolute(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(|error| format!("failed to resolve {}: {error}", path.display()))
}
