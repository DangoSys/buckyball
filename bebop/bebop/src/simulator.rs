/// Accelerator simulator with state management
use crate::builtin::Module;
use crate::buckyball::memdomain::decoder::DmaOperation;
use crate::buckyball::Top;
use crate::config::NpuConfig;
use crate::socket::DmaClient;

/// Accelerator simulator that manages state
pub struct Simulator {
  /// NPU top-level module
  top: Top,
}

impl Simulator {
  pub fn new(config: NpuConfig) -> Self {
    Self {
      top: Top::new("npu_top", config.mem_size),
    }
  }

  /// Process an instruction from Spike
  /// Returns the instruction execution result
  pub fn process(&mut self, funct: u32, xs1: u64, xs2: u64, dma_client: &mut DmaClient) -> std::io::Result<u64> {
    // 1. Convert socket message to instruction and send to top-level module
    self.top.send_instruction(funct as u64, xs1, xs2);

    // 2. Run one clock cycle
    self.top.tick();

    // 3. Check if there's a DMA operation to execute
    if let Some(dma_op) = self.top.get_dma_operation() {
      self.execute_dma(&dma_op, dma_client)?;
    }

    // 4. Get result
    let result = self.top.get_mem_data() as u64;
    Ok(result)
  }

  /// Execute DMA operation
  fn execute_dma(&mut self, dma_op: &DmaOperation, dma_client: &mut DmaClient) -> std::io::Result<()> {
    match dma_op {
      DmaOperation::Mvin(config) => {
        println!("  [Simulator] Executing MVIN DMA operation");
        self.execute_mvin(config, dma_client)?;
      },
      DmaOperation::Mvout(config) => {
        println!("  [Simulator] Executing MVOUT DMA operation");
        self.execute_mvout(config, dma_client)?;
      },
    }
    Ok(())
  }

  /// Execute MVIN DMA operation - read from DRAM to scratchpad
  fn execute_mvin(
    &mut self,
    config: &crate::buckyball::frontend::MvinConfig,
    dma_client: &mut DmaClient,
  ) -> std::io::Result<()> {
    const DIM: u32 = 16; // Number of elements per row

    // Get bank configuration based on scratchpad address
    // bank 0, 1 (spad): elem_size = 1 byte, row_bytes = 16
    // bank 2 (acc): elem_size = 4 bytes, row_bytes = 64
    let bank_id = config.base_sp_addr >> 10; // Simplified: high bits are bank id
    let elem_size = if bank_id >= 1024 { 4u32 } else { 1u32 }; // acc uses 4 bytes, spad uses 1 byte
    let row_bytes = DIM * elem_size; // Bytes per row: 16 or 64

    println!(
      "  [Simulator] MVIN: bank={}, elem_size={}, row_bytes={}, iter={}",
      bank_id, elem_size, row_bytes, config.iter
    );

    // Iterate by row, one DMA transfer per row (128 bit or 512 bit)
    for i in 0..config.iter {
      // Calculate DRAM start address for current row
      let dram_addr = (config.base_dram_addr as u64) + ((i * row_bytes) as u64);

      // Read entire row data at once (16 bytes or 64 bytes)
      // Note: Need to split into multiple 8-byte reads because DMA protocol limits max 8 bytes per operation
      let num_dma_ops = (row_bytes + 7) / 8; // Round up

      for dma_idx in 0..num_dma_ops {
        let offset = dma_idx * 8;
        let remaining = row_bytes - offset;
        let dma_size = remaining.min(8); // Max 8 bytes

        let addr = dram_addr + (offset as u64);
        let data = dma_client.dma_read(addr, dma_size)?;

        // Write data to scratchpad (split by bytes)
        for byte_idx in 0..dma_size {
          let byte_val = ((data >> (byte_idx * 8)) & 0xFF) as u32;
          let spad_addr = config.base_sp_addr + i * DIM + (offset + byte_idx) / elem_size;
          self.top.write_spad(spad_addr as usize, byte_val);
        }
      }
    }

    Ok(())
  }

  /// Execute MVOUT DMA operation - write from scratchpad to DRAM
  fn execute_mvout(
    &mut self,
    config: &crate::buckyball::frontend::MvoutConfig,
    dma_client: &mut DmaClient,
  ) -> std::io::Result<()> {
    const DIM: u32 = 16; // Number of elements per row

    // Get bank configuration based on scratchpad address
    let bank_id = config.base_sp_addr >> 10;
    let elem_size = if bank_id >= 1024 { 4u32 } else { 1u32 };
    let row_bytes = DIM * elem_size;

    println!(
      "  [Simulator] MVOUT: bank={}, elem_size={}, row_bytes={}, iter={}",
      bank_id, elem_size, row_bytes, config.iter
    );

    // Iterate by row, one DMA transfer per row
    for i in 0..config.iter {
      // Calculate DRAM start address for current row
      let dram_addr = (config.base_dram_addr as u64) + ((i * row_bytes) as u64);

      // Write entire row data at once, split into multiple 8-byte DMA operations
      let num_dma_ops = (row_bytes + 7) / 8;

      for dma_idx in 0..num_dma_ops {
        let offset = dma_idx * 8;
        let remaining = row_bytes - offset;
        let dma_size = remaining.min(8);

        // Read data from scratchpad and assemble into 64-bit
        let mut data = 0u64;
        for byte_idx in 0..dma_size {
          let spad_addr = config.base_sp_addr + i * DIM + (offset + byte_idx) / elem_size;
          let byte_val = (self.top.read_spad(spad_addr as usize) & 0xFF) as u64;
          data |= byte_val << (byte_idx * 8);
        }

        let addr = dram_addr + (offset as u64);
        dma_client.dma_write(addr, data, dma_size)?;
      }
    }

    Ok(())
  }

  /// Reset simulator
  pub fn reset(&mut self) {
    self.top.reset();
  }

  /// Initialize memory data
  pub fn init_mem(&mut self, addr: usize, data: u32) {
    self.top.init_mem(addr, data);
  }
}

impl Default for Simulator {
  fn default() -> Self {
    Self::new(NpuConfig::default())
  }
}
