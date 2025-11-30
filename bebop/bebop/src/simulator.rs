/// Accelerator simulator with state management
use crate::builtin::Module;
use crate::buckyball::memdomain::DmaOperation;
use crate::buckyball::Top;
use crate::config::NpuConfig;
use crate::socket::DmaClient;
use std::io::{self, BufRead, Write};

/// Execution mode for the simulator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepMode {
  #[default]
  Run,
  Step,
}

/// Accelerator simulator - drives Top module
pub struct Simulator {
  top: Top,
  step_mode: StepMode,
  cycle: u64,
}

impl Simulator {
  pub fn new(config: NpuConfig, step_mode: StepMode) -> Self {
    Self {
      top: Top::new("npu_top", config.mem_size),
      step_mode,
      cycle: 0,
    }
  }

  /// Process an instruction from Spike
  pub fn process(&mut self, funct: u32, xs1: u64, xs2: u64, dma_client: &mut DmaClient) -> std::io::Result<u64> {
    // 1. Issue instruction to Top
    self.top.issue(funct as u64, xs1, xs2);

    // 2. Step one clock cycle
    self.step();

    // 3. Check DMA request and execute if needed
    if let Some(dma_op) = self.top.get_dma_req() {
      self.execute_dma(dma_op, dma_client)?;
    }

    // 4. Read result
    Ok(self.top.get_mem_data() as u64)
  }

  fn step(&mut self) {
    self.cycle += 1;
    self.top.tick();

    if self.step_mode == StepMode::Step {
      println!("[Cycle {}]", self.cycle);
      self.step_mode = self.wait_for_input();
    }
  }

  /// Execute DMA operation via socket
  fn execute_dma(&mut self, dma_op: DmaOperation, dma_client: &mut DmaClient) -> std::io::Result<()> {
    const DIM: u32 = 16;

    match dma_op {
      DmaOperation::Mvin(config) => {
        let bank_id = config.base_sp_addr >> 10;
        let elem_size = if bank_id >= 1024 { 4u32 } else { 1u32 };
        let row_bytes = DIM * elem_size;

        for i in 0..config.iter {
          let dram_addr = (config.base_dram_addr as u64) + ((i * row_bytes) as u64);
          let num_dma_ops = (row_bytes + 7) / 8;

          for dma_idx in 0..num_dma_ops {
            let offset = dma_idx * 8;
            let dma_size = (row_bytes - offset).min(8);
            let addr = dram_addr + (offset as u64);
            let data = dma_client.dma_read(addr, dma_size)?;

            for byte_idx in 0..dma_size {
              let byte_val = ((data >> (byte_idx * 8)) & 0xFF) as u32;
              let spad_addr = config.base_sp_addr + i * DIM + (offset + byte_idx) / elem_size;
              self.top.dma_write_spad(spad_addr as usize, byte_val);
            }
          }
        }
      }
      DmaOperation::Mvout(config) => {
        let bank_id = config.base_sp_addr >> 10;
        let elem_size = if bank_id >= 1024 { 4u32 } else { 1u32 };
        let row_bytes = DIM * elem_size;

        for i in 0..config.iter {
          let dram_addr = (config.base_dram_addr as u64) + ((i * row_bytes) as u64);
          let num_dma_ops = (row_bytes + 7) / 8;

          for dma_idx in 0..num_dma_ops {
            let offset = dma_idx * 8;
            let dma_size = (row_bytes - offset).min(8);

            let mut data = 0u64;
            for byte_idx in 0..dma_size {
              let spad_addr = config.base_sp_addr + i * DIM + (offset + byte_idx) / elem_size;
              let byte_val = (self.top.dma_read_spad(spad_addr as usize) & 0xFF) as u64;
              data |= byte_val << (byte_idx * 8);
            }

            let addr = dram_addr + (offset as u64);
            dma_client.dma_write(addr, data, dma_size)?;
          }
        }
      }
    }
    Ok(())
  }

  fn wait_for_input(&self) -> StepMode {
    print!("(s)tep, (r)un, (q)uit: ");
    io::stdout().flush().unwrap();

    let mut line = String::new();
    if io::stdin().lock().read_line(&mut line).is_ok() {
      match line.trim() {
        "r" => StepMode::Run,
        "q" => std::process::exit(0),
        _ => StepMode::Step,
      }
    } else {
      StepMode::Step
    }
  }

  pub fn reset(&mut self) {
    self.top = Top::new("npu_top", 1024 * 1024); // Reset by recreating
    self.cycle = 0;
  }

  pub fn init_mem(&mut self, addr: usize, data: u32) {
    self.top.init_mem(addr, data);
  }
}
