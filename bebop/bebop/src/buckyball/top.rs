/// Top Module - top-level module, connects GlobalDecoder, GlobalRS, MemDomain and BallDomain
use crate::builtin::Module;
use crate::buckyball::frontend::{GlobalDecoder, GlobalReservationStation, DecoderInput};
use crate::buckyball::memdomain::{decoder::DmaOperation, MemDomain};
use crate::buckyball::balldomain::BallDomain;

/// Top - NPU top-level module
/// Architecture: GlobalDecoder -> GlobalRS -> {MemDomain, BallDomain}
pub struct Top {
  name: String,

  // Frontend: Global decoder and reservation station
  pub global_decoder: GlobalDecoder,
  pub global_rs: GlobalReservationStation,

  // Memory domain
  pub memdomain: MemDomain,

  // Ball domain
  pub balldomain: BallDomain,
}

impl Top {
  pub fn new(name: impl Into<String>, mem_size: usize) -> Self {
    const ROB_SIZE: usize = 16; // Match RTL default
    Self {
      name: name.into(),
      global_decoder: GlobalDecoder::new("global_decoder"),
      global_rs: GlobalReservationStation::new("global_rs", ROB_SIZE),
      memdomain: MemDomain::new("memdomain", mem_size),
      balldomain: BallDomain::new("balldomain"),
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
    // Run from back to front following RTL architecture
    // Pipeline: GlobalDecoder -> GlobalRS -> {MemDomain, BallDomain}

    // 1. Run MemDomain (processes memory instructions from GlobalRS)
    self.memdomain.run();

    // 2. Run BallDomain (processes ball instructions from GlobalRS)
    self.balldomain.run();

    // 3. Run GlobalRS (issues to domains, receives completion)
    self.global_rs.run();

    // 4. Run GlobalDecoder (decodes incoming instructions)
    self.global_decoder.run();

    // 5. Wire updates: this cycle's output -> next cycle's input

    // GlobalDecoder -> GlobalRS
    self.global_rs.input = self.global_decoder.output.clone();

    // GlobalRS -> MemDomain (issue)
    self.memdomain.global_issue_i = self.global_rs.mem_issue.clone();

    // GlobalRS -> BallDomain (issue)
    self.balldomain.global_issue_i = self.global_rs.ball_issue.clone();

    // MemDomain -> GlobalRS (completion)
    self.global_rs.mem_complete = self.memdomain.global_complete_o.clone();

    // BallDomain -> GlobalRS (completion)
    self.global_rs.ball_complete = self.balldomain.global_complete_o.clone();
  }

  fn reset(&mut self) {
    self.global_decoder.reset();
    self.global_rs.reset();
    self.memdomain.reset();
    self.balldomain.reset();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
