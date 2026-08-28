use clap::Parser;
use bebop_bemu::{tile_topology, BemuInstance, SharedMemory, TraceConfig};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

const DRAM_SIZE: usize = 1 << 30;

#[derive(Parser, Debug)]
struct Args {
    /// Tile index in the chip bundle.
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
}

struct StartGate {
    result: Mutex<Option<Result<(), String>>>,
    ready: Condvar,
}

impl StartGate {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn wait(&self) -> Result<(), String> {
        let mut result = self.result.lock().map_err(|_| "BEMU start gate poisoned".to_string())?;
        while result.is_none() {
            result = self.ready.wait(result).map_err(|_| "BEMU start gate poisoned".to_string())?;
        }
        result.as_ref().expect("BEMU start gate was set").clone()
    }

    fn release(&self, result: Result<(), String>) {
        *self.result.lock().expect("BEMU start gate poisoned") = Some(result);
        self.ready.notify_all();
    }
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
    if topology.cores.is_empty() {
        return Err(format!("tile {} has no Core instances", args.tile_index));
    }
    eprintln!(
        "[INFO] Multi-Rocket Chip BEMU: tile_index={} cores={} shared_dram={}MiB",
        args.tile_index,
        topology.cores.len(),
        DRAM_SIZE >> 20
    );

    let core_count = topology.cores.len();
    let virtual_bank_count = topology.virtual_bank_count;
    let memory = SharedMemory::new(DRAM_SIZE, core_count);
    let schedule = Arc::new(Mutex::new(()));
    let start = Arc::new(StartGate::new());
    let done = Arc::new(AtomicBool::new(false));
    let exit_code = Arc::new(AtomicI32::new(0));
    let (prepared_tx, prepared_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(topology.cores.len());

    for (hart_id, (core_name, core_index)) in topology.cores.into_iter().enumerate() {
        let elf = elf.clone();
        let worker_log = log_dir.join(format!("hart-{hart_id}"));
        let memory = Arc::clone(&memory);
        let schedule = Arc::clone(&schedule);
        let start = Arc::clone(&start);
        let done = Arc::clone(&done);
        let exit_code = Arc::clone(&exit_code);
        let prepared_tx = prepared_tx.clone();
        let pk = args.pk;
        let disasm = args.disasm;
        let tool_profile = args.tool_profile;
        workers.push(
            thread::Builder::new()
                .name(format!("core-{hart_id}-{core_name}"))
                .spawn(move || {
                    run_core(
                        hart_id,
                        &core_name,
                        core_index,
                        &elf,
                        &worker_log,
                        pk,
                        disasm,
                        tool_profile,
                        memory,
                        schedule,
                        start,
                        done,
                        exit_code,
                        prepared_tx,
                        virtual_bank_count,
                    )
                })
                .map_err(|error| format!("failed to spawn Core worker {hart_id}: {error}"))?,
        );
    }
    drop(prepared_tx);

    let mut preparation_error = None;
    for _ in 0..workers.len() {
        if let Err(error) = prepared_rx.recv().map_err(|_| "Core worker exited before initialization".to_string())? {
            preparation_error.get_or_insert(error);
        }
    }
    start.release(preparation_error.map_or(Ok(()), Err));

    let mut first_error = None;
    for (hart_id, worker) in workers.into_iter().enumerate() {
        match worker.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                first_error.get_or_insert(format!("Core worker {hart_id}: {error}"));
            }
            Err(_) => {
                first_error.get_or_insert(format!("Core worker {hart_id} panicked"));
            }
        };
    }
    first_error.map_or(Ok(()), Err)
}

#[allow(clippy::too_many_arguments)]
fn run_core(
    hart_id: usize,
    core_name: &str,
    core_index: usize,
    elf: &Path,
    log_dir: &Path,
    pk: bool,
    disasm: bool,
    tool_profile: bool,
    memory: Arc<SharedMemory>,
    schedule: Arc<Mutex<()>>,
    start: Arc<StartGate>,
    done: Arc<AtomicBool>,
    exit_code: Arc<AtomicI32>,
    prepared: mpsc::Sender<Result<(), String>>,
    virtual_bank_count: usize,
) -> Result<(), String> {
    eprintln!(
        "[INFO] starting Core worker hart={hart_id} core={core_name} core_index={core_index}"
    );
    let prepared_bemu = (|| {
        let _turn = schedule.lock().map_err(|_| "BEMU scheduler poisoned".to_string())?;
        let mut bemu = BemuInstance::new_with_core_hart(
            log_dir,
            TraceConfig::new(false, false),
            disasm,
            tool_profile,
            core_index,
            hart_id,
            Some(Arc::clone(&memory)),
            Some(virtual_bank_count),
        )
        .map_err(|error| error.to_string())?;
        bemu.load_elf(elf).map_err(|error| error.to_string())?;
        bemu.init_hart(pk).map_err(|error| error.to_string())?;
        Ok::<BemuInstance, String>(bemu)
    })();

    let mut bemu = match prepared_bemu {
        Ok(bemu) => {
            let _ = prepared.send(Ok(()));
            bemu
        }
        Err(error) => {
            let _ = prepared.send(Err(error.clone()));
            return Err(error);
        }
    };
    start.wait()?;

    loop {
        let barrier_hit = {
            let _turn = schedule.lock().map_err(|_| "BEMU scheduler poisoned".to_string())?;
            if done.load(Ordering::Acquire) {
                bemu.stop(exit_code.load(Ordering::Acquire));
                break;
            }
            if let Err(error) = bemu.step() {
                exit_code.store(1, Ordering::Release);
                done.store(true, Ordering::Release);
                memory.abort_barrier();
                return Err(error.to_string());
            }
            bemu.barrier_hit()
        };
        if barrier_hit {
            memory.wait_barrier(hart_id);
        }
        if bemu.finished() {
            let code = bemu.exit_code().unwrap_or(1);
            exit_code.store(code, Ordering::Release);
            done.store(true, Ordering::Release);
            memory.abort_barrier();
            break;
        }
    }

    let code = exit_code.load(Ordering::Acquire);
    eprintln!("[INFO] stopped Core worker hart={hart_id} core={core_name} exit={code}");
    if code == 0 {
        Ok(())
    } else {
        Err(format!("guest exited with code {code}"))
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
