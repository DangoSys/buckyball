use super::{decoder::DmaOperation, Bank, Controller, MemDecoder};
/// Memory Domain - connects Decoder, Controller and Bank together
use crate::builtin::Module;

/// Memory Domain - contains decoder, controller and bank, and handles wiring between them
pub struct MemDomain {
  name: String,
  pub decoder: MemDecoder,
  pub controller: Controller,
  pub bank: Bank,
}

impl MemDomain {
  pub fn new(name: impl Into<String>, bank_size: usize) -> Self {
    Self {
      name: name.into(),
      decoder: MemDecoder::new("mem_decoder"),
      controller: Controller::new("ctrl"),
      bank: Bank::new("bank", bank_size),
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
    // Run from back to front

    // 1. First run Bank (read last cycle's request)
    self.bank.run();

    // 2. Then run Controller (read last cycle's bank response)
    self.controller.run();

    // 3. Then run Decoder (read last cycle's input)
    self.decoder.run();

    // 4. Wire update: this cycle's output -> next cycle's input
    // Write request: Decoder -> Bank
    self.bank.write_req = self.decoder.output.value.write_req.clone();

    // Read request: Decoder -> Controller -> Bank
    self.controller.read_req = self.decoder.output.value.read_req.clone();
    self.bank.read_req = self.controller.req_out.clone();

    // Bank -> Controller read response
    self.controller.resp_in = self.bank.read_resp.clone();

    // Pass Decoder input to Controller (for DMA operations)
    // Need to pass original DecoderInput from Top module
    // Temporarily use decoder's input
    // TODO: Need to pass original funct, xs1, xs2 from Top
  }

  fn reset(&mut self) {
    self.decoder.reset();
    self.controller.reset();
    self.bank.reset();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
