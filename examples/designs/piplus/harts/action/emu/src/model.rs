use crate::shared_mem::SharedMem;
use bebop_bemu::{
    ball::linkme, HartFactory, HartModel, CoreParams, Cpu, HLink, FromString, ModelOptions,
    NoRocc, ThreadGroup, Whatever, HARTS,
};
use std::sync::{Arc, RwLock};

/// An action hart: one RV64GC hart driving its hardware thread (arch: ActionHart, behind TBus).
pub struct ActionHart {
    cpu: Cpu,
    threads: ThreadGroup,
}

impl HartModel for ActionHart {
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
    cpu.state()
        .install_thread_memory(Arc::new(SharedMem(RwLock::new(
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
    Ok(Box::new(ActionHart { cpu, threads }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static ACTION: HartFactory = HartFactory {
    component: "action",
    instantiate: compute,
};
