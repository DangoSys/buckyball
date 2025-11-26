/// Memory controller
use crate::builtin::{Module, Wire};
use super::super::mem::bank::{ReadReq, ReadResp};

/// Memory Controller - sends read requests to bank
pub struct Controller {
  name: String,

  // Input: read request signal
  pub read_req: Wire<ReadReq>,

  // Output: read request
  pub req_out: Wire<ReadReq>,

  // Input: read response
  pub resp_in: Wire<ReadResp>,

  // Internal state
  last_data: u32,
}

impl Controller {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      read_req: Wire::default(),
      req_out: Wire::default(),
      resp_in: Wire::default(),
      last_data: 0,
    }
  }

  /// Get last read data
  pub fn get_data(&self) -> u32 {
    self.last_data
  }
}

impl Module for Controller {
  fn run(&mut self) {
    // Handle read request signal
    if self.read_req.valid {
      self.req_out.set(self.read_req.value.clone());
    } else {
      self.req_out.clear();
    }

    // If valid response received
    if self.resp_in.valid {
      self.last_data = self.resp_in.value.data;
    }
  }

  fn reset(&mut self) {
    self.read_req = Wire::default();
    self.req_out = Wire::default();
    self.resp_in = Wire::default();
    self.last_data = 0;
  }

  fn name(&self) -> &str {
    &self.name
  }
}
