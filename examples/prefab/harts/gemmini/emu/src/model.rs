use crate::Gemmini;
use bebop_bemu::{
    ball::linkme, CoreParams, Cpu, HLink, HartFactory, HartModel, ModelOptions, Whatever, HARTS,
};

/// The Gemmini hart: a Rocket RV64GC CPU with Gemmini (DefaultGemmini) on its
/// RoCC port.
pub struct GemminiHart {
    cpu: Cpu,
    gemmini: Gemmini,
}

impl HartModel for GemminiHart {
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    fn step(&mut self, count: u64) -> Result<(), Whatever> {
        self.cpu.step(count, &mut self.gemmini)
    }
}

fn instantiate(
    device: &HLink,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn HartModel>, String> {
    let cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: false,
        channel_by_source: vec![0; params.threads.len() + 1],
    }, bebop_bemu::cpu_timing(params))
    .map_err(|error| error.to_string())?;
    Ok(Box::new(GemminiHart {
        cpu,
        gemmini: Gemmini::new(),
    }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static GEMMINI: HartFactory = HartFactory {
    component: "gemmini",
    instantiate,
};
