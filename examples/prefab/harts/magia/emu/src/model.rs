use crate::Mesh;
use bebop_bemu::{
    ball::linkme, CoreParams, Cpu, HLink, HartFactory, HartModel, ModelOptions, Whatever, HARTS,
};

/// The MAGIA hart: Loom, an RV64GC process CPU, launches the mesh of tiles
/// with custom-0.
pub struct MagiaHart {
    cpu: Cpu,
    mesh: Mesh,
}

impl HartModel for MagiaHart {
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    fn step(&mut self, count: u64) -> Result<(), Whatever> {
        self.cpu.step(count, &mut self.mesh)
    }
}

fn instantiate(
    device: &HLink,
    params: &CoreParams,
    options: &ModelOptions,
) -> Result<Box<dyn HartModel>, String> {
    let mut cpu = Cpu::new(device, params, options, bebop_bemu::DmaLayout {
        read_row_cycles: 1,
        range_maintenance: false,
        channel_by_source: vec![0; params.threads.len() + 1],
    }, bebop_bemu::cpu_timing(params))
    .map_err(|error| error.to_string())?;
    // Loom's FPU has half precision: it converts the operands of the mesh.
    cpu.enable_half_precision();
    Ok(Box::new(MagiaHart {
        cpu,
        mesh: Mesh::new(),
    }))
}

#[linkme::distributed_slice(HARTS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static MAGIA: HartFactory = HartFactory {
    component: "magia",
    instantiate,
};
