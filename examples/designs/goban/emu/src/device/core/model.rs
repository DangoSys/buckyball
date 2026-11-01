use bebop_bemu::{
    ball::linkme, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, FromString, ModelOptions,
    NoRocc, ThreadGroup, Whatever, CORES,
};

use std::sync::{Arc, Mutex};
struct LocalMemory(Mutex<buckyball_core_shm::Storage>);
impl bebop_bemu::core::ThreadMemory for LocalMemory {
    fn transfer_cycles(&self, thread: usize, address: u64, rows: u64, write: bool) -> Option<u64> {
        self.0
            .lock()
            .unwrap()
            .transfer_cycles(thread, address, rows, write)
    }
    fn start(&self, thread: usize, task: u64) {
        self.0.lock().unwrap().start(thread, task);
    }
    fn control(&self, operation: u32, field: u64, value: u64) -> u64 {
        self.0.lock().unwrap().control(operation, field, value)
    }
    fn read(&self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        self.0.lock().unwrap().read(thread, address, output)
    }
    fn write(&self, thread: usize, address: u64, input: &[u8]) -> bool {
        self.0.lock().unwrap().write(thread, address, input)
    }
}

/// One RV64GC hart driving its matmul hardware threads (arch: GobanCore, behind TBus).
pub struct GobanCore {
    cpu: Cpu,
    threads: ThreadGroup,
}

impl CoreModel for GobanCore {
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    fn step(&mut self, count: u64) -> Result<(), Whatever> {
        self.threads.check().map_err(Whatever::without_source)?;
        self.cpu.step(count, &mut NoRocc)?;
        if self.cpu.finished() {
            self.threads.shutdown().map_err(Whatever::without_source)?;
        }
        Ok(())
    }
}

fn instantiate(
    device: &DeviceModel,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    cpu.state()
        .install_thread_memory(Arc::new(LocalMemory(Mutex::new(
            buckyball_core_shm::Storage::new(
                params.threads.len(),
                params.shared_physical_bank_count,
                params.shared_bank_size,
                buckyball_core_shm::Mesh {
                    columns: 2,
                    thread_nodes: vec![0, 1, 2, 3],
                    banks_per_node: params.shared_physical_bank_count / 4,
                },
            ),
        ))));
    let threads = ThreadGroup::new(cpu.state(), options)?;
    Ok(Box::new(GobanCore { cpu, threads }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static GOBAN: CoreFactory = CoreFactory {
    component: "goban",
    instantiate,
};
