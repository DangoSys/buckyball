use super::shared_mem::SharedMem;
use bebop_bemu::{
    ball::linkme, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, FromString, ModelOptions,
    NoRocc, ThreadGroup, Whatever, CORES,
};
use std::sync::{Arc, Mutex};

/// A dense or action core: one RV64GC hart driving its hardware thread (arch: ComputeCore, behind TBus).
pub struct ComputeCore {
    cpu: Cpu,
    threads: ThreadGroup,
}

impl CoreModel for ComputeCore {
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

fn compute(
    device: &DeviceModel,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    cpu.state()
        .install_thread_memory(Arc::new(SharedMem(Mutex::new(
            buckyball_core_shm::Storage::new(
                params.threads.len(),
                params.shared_physical_bank_count,
                params.shared_bank_size,
                buckyball_core_shm::Mesh {
                    columns: 1,
                    thread_nodes: vec![0],
                    banks_per_node: params.shared_physical_bank_count,
                },
            ),
        ))));
    let threads = ThreadGroup::new(cpu.state(), options)?;
    Ok(Box::new(ComputeCore { cpu, threads }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static DENSE: CoreFactory = CoreFactory {
    component: "dense",
    instantiate: compute,
};

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static ACTION: CoreFactory = CoreFactory {
    component: "action",
    instantiate: compute,
};
