use super::{decoder::DmaOperation, Bank, Controller, MemDecoder, MemLoader, MemStorer};
use super::rs::ReservationStation;
/// Memory Domain - connects Decoder, RS, Loader, Storer, Controller and Bank together
use crate::builtin::Module;
use crate::buckyball::frontend::{GlobalRsComplete, GlobalRsIssue};

/// Memory Domain - contains all memory subsystem components
pub struct MemDomain {
  name: String,

  decoder: MemDecoder,
  rs: ReservationStation,
  mem_loader: MemLoader,
  mem_storer: MemStorer,
  controller: Controller,
  bank: Bank,

  pub global_issue_i: crate::builtin::Wire<GlobalRsIssue>,
  pub global_complete_o: crate::builtin::Wire<GlobalRsComplete>,
}

impl MemDomain {
  pub fn new(name: impl Into<String>, bank_size: usize) -> Self {
  Self {
    name: name.into(),
    decoder: MemDecoder::new("mem_decoder"),
    rs: ReservationStation::new("mem_rs"),
    mem_loader: MemLoader::new("mem_loader"),
    mem_storer: MemStorer::new("mem_storer"),
    controller: Controller::new("ctrl"),
    bank: Bank::new("bank", bank_size),
    global_issue_i: crate::builtin::Wire::default(),
    global_complete_o: crate::builtin::Wire::default(),
  }
  }

  /// Write data to bank (for initialization, bypassing signal lines)
  pub fn init_write(&mut self, addr: usize, data: u32) {
  self.bank.init_write(addr, data);
  }

  /// Get last read data
  pub fn get_data(&self) -> u32 {
  self.controller.get_data()
  }

  /// Get DMA operation (if any)
  pub fn get_dma_operation(&self) -> Option<DmaOperation> {
  if self.decoder.output.valid {
    self.decoder.output.value.dma_op.clone()
  } else {
    None
  }
  }

  /// Write to scratchpad
  pub fn write_spad(&mut self, addr: usize, data: u32) {
  self.bank.init_write(addr, data);
  }

  /// Read from scratchpad
  pub fn read_spad(&self, addr: usize) -> u32 {
  self.bank.read_data(addr)
  }
}

impl Module for MemDomain {
  fn run(&mut self) {
  // Run from back to front following RTL architecture

  // 1. Run Bank (read last cycle's request)
  self.bank.run();

  // 2. Run Controller (read last cycle's bank response)
  self.controller.run();

  // 3. Run MemLoader and MemStorer (handle DMA operations)
  self.mem_loader.run();
  self.mem_storer.run();

  // 4. Run RS (issue to Loader/Storer, forward completion)
  self.rs.run();

  // 5. Run Decoder (decode incoming instruction from Global RS)
  self.decoder.run();

  // 6. Wire updates: this cycle's output -> next cycle's input

  // Global RS -> Decoder
  self.decoder.input.set(self.global_issue_i.value.cmd.clone());

  // Decoder -> RS (with rob_id from Global RS)
  self.rs.decode_input.set(self.global_issue_i.value.clone());

  // RS -> MemLoader/MemStorer
  self.mem_loader.cmd_req = self.rs.issue_output.ld.clone();
  self.mem_storer.cmd_req = self.rs.issue_output.st.clone();

  // MemLoader/MemStorer -> RS (completion)
  self.rs.commit_input.ld = self.mem_loader.cmd_resp.clone();
  self.rs.commit_input.st = self.mem_storer.cmd_resp.clone();

  // RS -> Global RS (completion)
  self.global_complete_o = self.rs.complete_output.clone();

  // Legacy wiring for Bank/Controller (kept for compatibility)
  self.bank.write_req = self.decoder.output.value.write_req.clone();
  self.controller.read_req = self.decoder.output.value.read_req.clone();
  self.bank.read_req = self.controller.req_out.clone();
  self.controller.resp_in = self.bank.read_resp.clone();
  }

  fn reset(&mut self) {
  self.decoder.reset();
  self.rs.reset();
  self.mem_loader.reset();
  self.mem_storer.reset();
  self.controller.reset();
  self.bank.reset();
  self.global_issue_i = crate::builtin::Wire::default();
  self.global_complete_o = crate::builtin::Wire::default();
  }

  fn name(&self) -> &str {
  &self.name
  }
}
