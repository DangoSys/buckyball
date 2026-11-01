use bebop_bemu::{ball::linkme, CoreFactory, CoreModel, CoreParams, Cpu, DeviceModel, ModelOptions, NoRocc, Whatever, CORES};

/// The control core: one RV64GC hart with nothing on its RoCC port (arch: ControlCore).
pub struct ControlCore {
    cpu: Cpu,
}

impl CoreModel for ControlCore {
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

fn control(device: &DeviceModel, params: &CoreParams, options: &ModelOptions) -> Result<Box<dyn CoreModel>, String> {
    let cpu = Cpu::new(device, params, options).map_err(|error| error.to_string())?;
    Ok(Box::new(ControlCore { cpu }))
}

#[linkme::distributed_slice(CORES)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static CONTROL: CoreFactory = CoreFactory {
    component: "control",
    instantiate: control,
};
