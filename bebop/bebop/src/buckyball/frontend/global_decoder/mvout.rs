/// MVOUT (Move Out) instruction decoder
/// Moves data out of the accelerator scratchpad via DMA
///
/// This instruction triggers a DMA write from scratchpad to DRAM.
///
/// Instruction format (from bb-tests/workloads/lib/bbhw/isa/25_mvout.c):
/// rs1[0:31]: base_dram_addr - DRAM start address
/// rs2[0:13]: base_sp_addr - scratchpad start address
/// rs2[14:23]: iter - iteration count

/// MVOUT instruction configuration
#[derive(Debug, Clone, Default)]
pub struct MvoutConfig {
  pub base_dram_addr: u32,  // DRAM start address
  pub base_sp_addr: u32,     // scratchpad start address (14 bits)
  pub iter: u32,             // iteration count (10 bits)
}

impl MvoutConfig {
  /// Parse configuration from instruction fields
  pub fn from_fields(xs1: u64, xs2: u64) -> Self {
    // rs1[0:31]: base_dram_addr
    let base_dram_addr = (xs1 & 0xFFFFFFFF) as u32;

    // rs2[0:13]: base_sp_addr (14 bits)
    let base_sp_addr = (xs2 & 0x3FFF) as u32;

    // rs2[14:23]: iter (10 bits)
    let iter = ((xs2 >> 14) & 0x3FF) as u32;

    Self {
      base_dram_addr,
      base_sp_addr,
      iter,
    }
  }
}

/// Process MVOUT instruction
pub fn process(xs1: u64, xs2: u64) -> u64 {
  let config = MvoutConfig::from_fields(xs1, xs2);

  println!("  -> MVOUT: dram_addr=0x{:08x}, sp_addr=0x{:04x}, iter={}",
    config.base_dram_addr,
    config.base_sp_addr,
    config.iter
  );

  // DMA operation will be handled by simulator
  0
}
