/// Memory bank with scratchpad
use crate::buckyball::builtin::{Module, Wire};

/// Read request
#[derive(Clone, Default)]
pub struct ReadReq {
  pub addr: u32,
}

/// Read response
#[derive(Clone, Default)]
pub struct ReadResp {
  pub data: u32,
}

/// Write request
#[derive(Clone, Default)]
pub struct WriteReq {
  pub addr: u32,
  pub data: u32,
}

/// Memory Bank - simple scratchpad storage
pub struct Bank {
  name: String,

  // Input: read request
  pub read_req: Wire<ReadReq>,

  // Input: write request
  pub write_req: Wire<WriteReq>,

  // Output: read response
  pub read_resp: Wire<ReadResp>,

  // Scratchpad storage
  spad: Vec<u32>,
}

impl Bank {
  pub fn new(name: impl Into<String>, size: usize) -> Self {
    Self {
      name: name.into(),
      read_req: Wire::default(),
      write_req: Wire::default(),
      read_resp: Wire::default(),
      spad: vec![0; size],
    }
  }

  /// Direct write data (for initialization, bypassing signal lines)
  pub fn init_write(&mut self, addr: usize, data: u32) {
    if addr < self.spad.len() {
      self.spad[addr] = data;
    }
  }

  /// Direct read data (for DMA, bypassing signal lines)
  pub fn read_data(&self, addr: usize) -> u32 {
    if addr < self.spad.len() {
      self.spad[addr]
    } else {
      0 // Out of bounds returns 0
    }
  }
}

impl Module for Bank {
  fn run(&mut self) {
    // Handle write request
    if self.write_req.valid {
      let addr = self.write_req.value.addr as usize;
      if addr < self.spad.len() {
        self.spad[addr] = self.write_req.value.data;
      }
    }

    // Handle read request
    if self.read_req.valid {
      let addr = self.read_req.value.addr as usize;
      let data = if addr < self.spad.len() {
        self.spad[addr]
      } else {
        0 // Out of bounds returns 0
      };
      self.read_resp.set(ReadResp { data });
    } else {
      self.read_resp.clear();
    }
  }

  fn reset(&mut self) {
    self.read_req = Wire::default();
    self.write_req = Wire::default();
    self.read_resp = Wire::default();
    self.spad.fill(0);
  }

  fn name(&self) -> &str {
    &self.name
  }
}
