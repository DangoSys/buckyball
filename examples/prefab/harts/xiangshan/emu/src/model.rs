use bebop_bemu::{
    ball::linkme, HartFactory, HartModel, CoreParams, Cpu, HLink, ModelOptions, NoRocc, Whatever, HARTS,
};

/// The XiangShan hart: one RV64GC hart with nothing on its RoCC port. A first
/// version: the functional model is exact, its timing is the in-order Rocket
/// CPU timing of `bebop_bemu::cpu_timing`, not that of the out-of-order XiangShan
/// pipeline (issue width, ROB, branch prediction, its L2 and its prefetchers).
pub struct XiangShanCore {
    cpu: Cpu,
}

impl HartModel for XiangShanCore {
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

fn instantiate(
    device: &HLink,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn HartModel>, String> {
    let cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: false,
        channel_by_source: vec![0; params.threads.len() + 1],
    }, bebop_bemu::cpu_timing(params)).map_err(|error| error.to_string())?;
    Ok(Box::new(XiangShanCore { cpu }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static XIANGSHAN: HartFactory = HartFactory {
    component: "xiangshan",
    instantiate,
};
