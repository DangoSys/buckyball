use crate::shared_mem::SharedMem;
use bebop_bemu::{
    ball::linkme, HartFactory, HartModel, CoreParams, Cpu, HLink, FromString, ModelOptions,
    NoRocc, ThreadGroup, Whatever, HARTS,
};
use std::sync::{Arc, RwLock};

/// A dense hart: one RV64GC hart driving its hardware thread (arch: DenseHart, behind TBus).
pub struct DenseHart {
    cpu: Cpu,
    threads: ThreadGroup,
}

impl HartModel for DenseHart {
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
    device: &HLink,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn HartModel>, String> {
    let cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: true,
        // Every thread has its own channel of the DDR's beat (arch: a Widen each before the hart's
        // port), so the threads move their rows at the same time and share only the DDR.
        channel_by_source: (0..=params.threads.len()).collect(),
    }, bebop_bemu::cpu_timing(params)).map_err(|error| error.to_string())?;
    // The SharedMem of DenseHart: twelve nodes of one bank, 3 x 4, thread i at node i.
    assert!(params.threads.len() == 12 && params.shared_physical_bank_count == 12);
    cpu.state()
        .install_thread_memory(Arc::new(SharedMem(RwLock::new(
            buckyball_hart_shm::Storage::new(
                12,
                12,
                params.shared_bank_size,
                buckyball_hart_shm::Mesh {
                    columns: 4,
                    thread_nodes: (0..12).collect(),
                    banks_per_node: 1,
                },
            ),
        ))));
    let threads = ThreadGroup::new(
        &cpu.state().link,
        bebop_bemu::thread_workers(&cpu.state().link, bebop_bemu::ball::BUS, options)?,
    )?;
    Ok(Box::new(DenseHart { cpu, threads }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static DENSE: HartFactory = HartFactory {
    component: "dense",
    instantiate: compute,
};
