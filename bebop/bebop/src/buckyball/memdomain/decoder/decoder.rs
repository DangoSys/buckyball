/// Memory Domain Decoder - memory access instruction decoder
use crate::builtin::{Module, Wire};
use crate::buckyball::frontend::{DecoderOutput, MvinConfig, MvoutConfig};

/// MemDecoder input (from global Decoder output)
pub type MemDecoderInput = DecoderOutput;

use super::super::mem::bank::{ReadReq, WriteReq};

/// DMA operation type
#[derive(Debug, Clone)]
pub enum DmaOperation {
  Mvin(MvinConfig),   // Move In - DMA read from DRAM to scratchpad
  Mvout(MvoutConfig), // Move Out - DMA write from scratchpad to DRAM
}

/// MemDecoder output
#[derive(Clone, Default)]
pub struct MemDecoderOutput {
  pub read_req: Wire<ReadReq>,   // Read request signal
  pub write_req: Wire<WriteReq>, // Write request signal
  pub dma_op: Option<DmaOperation>, // DMA operation configuration
}

/// Memory Domain Decoder - memory access decoder
pub struct MemDecoder {
  name: String,

  // Input
  pub input: Wire<MemDecoderInput>,

  // Output
  pub output: Wire<MemDecoderOutput>,
}

impl MemDecoder {
  pub fn new(name: impl Into<String>) -> Self {
  Self {
    name: name.into(),
    input: Wire::default(),
    output: Wire::default(),
  }
  }
}

impl Module for MemDecoder {
  fn run(&mut self) {
  if !self.input.valid {
    self.output.clear();
    return;
  }

  let input = &self.input.value;

  let mut output = MemDecoderOutput::default();

  match input.funct {
    24 => {
    // MVIN - DMA read from DRAM to scratchpad
    let config = MvinConfig::from_fields(input.xs1, input.xs2);
    println!(
      "  [MemDecoder] MVIN: dram=0x{:08x}, spad=0x{:04x}, iter={}, col_stride={}",
      config.base_dram_addr, config.base_sp_addr, config.iter, config.col_stride
    );

    output.dma_op = Some(DmaOperation::Mvin(config));
    output.read_req.clear();
    output.write_req.clear();
    },
    25 => {
    // MVOUT - DMA write from scratchpad to DRAM
    let config = MvoutConfig::from_fields(input.xs1, input.xs2);
    println!(
      "  [MemDecoder] MVOUT: dram=0x{:08x}, spad=0x{:04x}, iter={}",
      config.base_dram_addr, config.base_sp_addr, config.iter
    );

    output.dma_op = Some(DmaOperation::Mvout(config));
    output.read_req.clear();
    output.write_req.clear();
    },
    _ => {
    println!("  [MemDecoder] UNKNOWN: funct={}", input.funct);
    output.dma_op = None;
    output.read_req.clear();
    output.write_req.clear();
    },
  }

  self.output.set(output);
  }

  fn reset(&mut self) {
  self.input = Wire::default();
  self.output = Wire::default();
  }

  fn name(&self) -> &str {
  &self.name
  }
}
