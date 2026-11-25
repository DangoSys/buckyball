use bebop_bemu::{
    ball::linkme, CoreParams, Cpu, HLink, HartFactory, HartModel, ModelOptions, NoRocc, Whatever, HARTS,
};

/// The control hart: one RV64GC hart with nothing on its RoCC port (arch: ControlHart, Loom).
pub struct ControlHart {
    cpu: Cpu,
}

impl HartModel for ControlHart {
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

fn control(device: &HLink, params: &CoreParams, options: &ModelOptions) -> Result<Box<dyn HartModel>, String> {
    let cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: false,
        channel_by_source: vec![0; params.threads.len() + 1],
    }, bebop_bemu::cpu_timing(params)).map_err(|error| error.to_string())?;
    Ok(Box::new(ControlHart { cpu }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static CONTROL: HartFactory = HartFactory {
    component: "control",
    instantiate: control,
};
