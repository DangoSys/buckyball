use bebop_bemu::{
    ball::linkme, HartFactory, HartModel, CoreParams, Cpu, HLink, FromString, ModelOptions,
    NoRocc, ThreadGroup, Whatever, HARTS,
};

use std::sync::{Arc, RwLock};
struct LocalMemory(RwLock<buckyball_hart_shm::Storage>);
impl bebop_bemu::ThreadMemory for LocalMemory {
    fn span(&self, thread: usize, address: u64, write: bool) -> (bool, u64) {
        self.0.read().unwrap().span(thread, address, write)
    }
    fn contains(&self, thread: usize, address: u64, bytes: usize, write: bool) -> bool {
        self.0.read().unwrap().contains(thread, address, bytes, write)
    }
    fn transfer_cycles(&self, thread: usize, spans: &[(u64, usize)], write: bool) -> Option<u64> {
        self.0.read().unwrap().transfer_cycles(thread, spans, write)
    }
    fn start(&self, thread: usize, task: u64) {
        self.0.write().unwrap().start(thread, task);
    }
    fn control(&self, operation: u32, field: u64, value: u64) -> u64 {
        self.0.write().unwrap().control(operation, field, value)
    }
    fn read(&self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        self.0.read().unwrap().read(thread, address, output)
    }
    fn write(&self, thread: usize, address: u64, input: &[u8]) -> bool {
        self.0.write().unwrap().write(thread, address, input)
    }
}

/// One RV64GC hart driving its matmul hardware threads (arch: GobanHart, behind TBus).
pub struct GobanHart {
    cpu: Cpu,
    threads: ThreadGroup,
}

impl HartModel for GobanHart {
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
    device: &HLink,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn HartModel>, String> {
    let cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: true,
        channel_by_source: vec![0; params.threads.len() + 1],
    }, bebop_bemu::cpu_timing(params)).map_err(|error| error.to_string())?;
    cpu.state()
        .install_thread_memory(Arc::new(LocalMemory(RwLock::new(
            buckyball_hart_shm::Storage::new(
                params.threads.len(),
                params.shared_physical_bank_count,
                params.shared_bank_size,
                buckyball_hart_shm::Mesh {
                    columns: 2,
                    thread_nodes: vec![0, 1, 2, 3],
                    banks_per_node: params.shared_physical_bank_count / 4,
                },
            ),
        ))));
    let threads = ThreadGroup::new(
        &cpu.state().link,
        bebop_bemu::thread_workers(&cpu.state().link, bebop_bemu::ball::BUS, options)?,
    )?;
    Ok(Box::new(GobanHart { cpu, threads }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static GOBAN: HartFactory = HartFactory {
    component: "goban",
    instantiate,
};
