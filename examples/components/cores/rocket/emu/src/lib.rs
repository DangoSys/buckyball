use bebop_bemu::{ball::linkme, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, ModelOptions, NoRocc, Whatever, CORES};

/// The Rocket core: one RV64GC hart with nothing on its RoCC port.
pub struct RocketCore {
    cpu: Cpu,
}

impl CoreModel for RocketCore {
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    fn step(&mut self, count: u64) -> Result<(), Whatever> {
        self.cpu.step(count, &mut NoRocc)
    }
}

fn instantiate(device: &DeviceModel, params: &CoreParams, options: &ModelOptions) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    Ok(Box::new(RocketCore { cpu }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static ROCKET: CoreFactory = CoreFactory { component: "rocket", instantiate };
