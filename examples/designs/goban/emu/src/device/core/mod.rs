use bebop_bemu::{
    ball::linkme, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, FromString, ModelOptions, NoRocc, ThreadGroup,
    Whatever, CORES,
};

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

fn instantiate(device: &DeviceModel, params: &CoreParams, options: &ModelOptions) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    let threads = ThreadGroup::new(cpu.state(), options)?;
    Ok(Box::new(GobanCore { cpu, threads }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static GOBAN: CoreFactory = CoreFactory { component: "goban", instantiate };
