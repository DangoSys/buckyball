use bebop_bemu::{
    ball::linkme, Accelerator, BemuProfileReport, HartFactory, HartModel, CoreParams, Cpu, HLink, ModelOptions,
    Whatever, HARTS,
};
use std::time::Duration;

/// The reduce hart: one RV64 hart with a Ball accelerator on its RoCC port.
pub struct ReduceHart {
    cpu: Cpu,
    acc: Accelerator,
}

impl HartModel for ReduceHart {
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    fn step(&mut self, count: u64) -> Result<(), Whatever> {
        self.cpu.step(count, &mut self.acc)
    }
    fn take_barrier(&mut self) -> bool {
        self.acc.take_barrier()
    }
    fn profile_report(&self, total: Duration) -> Option<BemuProfileReport> {
        self.acc.profile_report(total, self.cpu.cpu_elapsed())
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
    let acc = Accelerator::attached(cpu.hart_id(), cpu.state().clone(), bebop_bemu::ball::BUS, options)?;
    Ok(Box::new(ReduceHart { cpu, acc }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static REDUCE: HartFactory = HartFactory {
    component: "reduce",
    instantiate,
};
