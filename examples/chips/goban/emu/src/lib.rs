//===- lib.rs - Goban BEMU instruction set --------------------------------===//

pub use super::bank_matrix;
pub use super::decode;
pub use super::instruction;

#[path = "../../../../balls/transpose/emu/src/49_transpose.rs"]
pub mod f49_transpose;
#[path = "../../../../balls/matrix/emu/src/65_matrix.rs"]
pub mod f65_matrix;

use instruction::{ExecContext, Instruction};

macro_rules! register_instructions {
  ($($inst:path),* $(,)?) => {
    pub fn execute_known(
      funct: u32,
      xs1: u64,
      xs2: u64,
      ctx: &mut ExecContext,
    ) -> Option<u64> {
      match funct {
        $(
          <$inst as Instruction>::FUNCT => {
            Some(<$inst as Instruction>::exec(xs1, xs2, ctx))
          }
        )*
        _ => None,
      }
    }

    pub fn cycles_after_issue(funct: u32, xs1: u64, xs2: u64) -> u64 {
      match funct {
        $(
          <$inst as Instruction>::FUNCT => {
            <$inst as Instruction>::latency(xs1, xs2)
          }
        )*
        _ => 1,
      }
    }
  };
}

register_instructions! {
  super::f00_fence::Fence,
  super::f01_barrier::Barrier,
  super::f16_mvout::Mvout,
  super::f32_mset::Mset,
  super::f33_mvin::Mvin,
  super::f34_mmio_set::MmioSet,
  super::f35_mvin_mmio::MvinMmio,
  f49_transpose::Transpose,
  f65_matrix::Matrix,
}
