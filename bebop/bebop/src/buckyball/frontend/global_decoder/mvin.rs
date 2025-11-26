/// MVIN (Move In) instruction decoder
/// Moves data into the accelerator scratchpad via DMA
///
/// This instruction triggers a DMA read from DRAM to scratchpad memory.
///
/// Instruction format (from bb-tests/workloads/lib/bbhw/isa/24_mvin.c):
/// rs1[0:31]: base_dram_addr - DRAM start address
/// rs2[0:13]: base_sp_addr - scratchpad start address
/// rs2[14:23]: iter - iteration count
/// rs2[24:33]: col_stride - column stride

/// MVIN instruction configuration
#[derive(Debug, Clone, Default)]
pub struct MvinConfig {
  pub base_dram_addr: u32,  // DRAM start address
  pub base_sp_addr: u32,   // scratchpad start address (14 bits)
  pub iter: u32,       // iteration count (10 bits)
  pub col_stride: u32,     // column stride (10 bits)
}

impl MvinConfig {
  /// Parse configuration from instruction fields
  pub fn from_fields(xs1: u64, xs2: u64) -> Self {
  // rs1[0:31]: base_dram_addr
  let base_dram_addr = (xs1 & 0xFFFFFFFF) as u32;

  // rs2[0:13]: base_sp_addr (14 bits)
  let base_sp_addr = (xs2 & 0x3FFF) as u32;

  // rs2[14:23]: iter (10 bits)
  let iter = ((xs2 >> 14) & 0x3FF) as u32;

  // rs2[24:33]: col_stride (10 bits)
  let col_stride = ((xs2 >> 24) & 0x3FF) as u32;

  Self {
    base_dram_addr,
    base_sp_addr,
    iter,
    col_stride,
  }
  }
}

/// Process MVIN instruction
pub fn process(xs1: u64, xs2: u64) -> u64 {
  let config = MvinConfig::from_fields(xs1, xs2);

  println!("  -> MVIN: dram_addr=0x{:08x}, sp_addr=0x{:04x}, iter={}, col_stride={}",
  config.base_dram_addr,
  config.base_sp_addr,
  config.iter,
  config.col_stride
  );

  // DMA operation will be handled by simulator
  0
}
