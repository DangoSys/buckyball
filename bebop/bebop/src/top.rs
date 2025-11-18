/// Top Module - top-level module, connects global Decoder and various Domains
use crate::builtin::Module;
use crate::global_decoder::{Decoder, DecoderInput};
use crate::memdomain::{decoder::DmaOperation, MemDomain};

/// Top - NPU top-level module
pub struct Top {
  name: String,

  // Global decoder
  pub global_decoder: Decoder,

  // Memory domain
  pub memdomain: MemDomain,
}

impl Top {
  pub fn new(name: impl Into<String>, mem_size: usize) -> Self {
    Self {
      name: name.into(),
      global_decoder: Decoder::new("global_decoder"),
      memdomain: MemDomain::new("memdomain", mem_size),
    }
  }

  /// Send instruction
  pub fn send_instruction(&mut self, funct: u64, xs1: u64, xs2: u64) {
    self.global_decoder.input.set(DecoderInput { funct, xs1, xs2 });
  }

  /// Get memory access result
  pub fn get_mem_data(&self) -> u32 {
    self.memdomain.get_data()
  }

  /// Initialize memory
  pub fn init_mem(&mut self, addr: usize, data: u32) {
    self.memdomain.init_write(addr, data);
  }

  /// Get DMA operation (if any)
  pub fn get_dma_operation(&self) -> Option<DmaOperation> {
    self.memdomain.get_dma_operation()
  }

  /// Write to scratchpad
  pub fn write_spad(&mut self, addr: usize, data: u32) {
    self.memdomain.write_spad(addr, data);
  }

  /// Read from scratchpad
  pub fn read_spad(&self, addr: usize) -> u32 {
    self.memdomain.read_spad(addr)
  }
}

impl Module for Top {
  fn run(&mut self) {
    // Run from back to front: first run downstream modules (read last cycle's input), then run upstream modules (generate this cycle's output)

    // 1. First run MemDomain (read input set by global_decoder in last cycle)
    self.memdomain.run();

    // 2. Then run global Decoder (generate this cycle's output)
    self.global_decoder.run();

    // 3. Wire update: this cycle's output -> next cycle's input (update registers)
    self.memdomain.decoder.input = self.global_decoder.output.clone();
  }

  fn reset(&mut self) {
    self.global_decoder.reset();
    self.memdomain.reset();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
