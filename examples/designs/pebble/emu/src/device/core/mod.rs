use bebop_bemu::{
    ball::linkme, Accelerator, BemuProfileReport, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, ModelOptions,
    Whatever, CORES,
};
use std::time::Duration;

/// One RV64GC hart with its local NPU on the RoCC port (arch: PebbleCore, Loom and Acc).
pub struct PebbleCore {
    cpu: Cpu,
    acc: Accelerator,
}

impl CoreModel for PebbleCore {
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
        self.acc.profile_report(total, &self.cpu)
    }
}

fn instantiate(device: &DeviceModel, params: &CoreParams, options: &ModelOptions) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    let acc = Accelerator::attached(&cpu, options)?;
    Ok(Box::new(PebbleCore { cpu, acc }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static PEBBLE: CoreFactory = CoreFactory { component: "pebble", instantiate };
