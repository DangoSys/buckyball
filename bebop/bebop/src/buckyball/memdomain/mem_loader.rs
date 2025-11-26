/// MemLoader - Handles MVIN (load from DRAM to scratchpad) operations
use crate::builtin::{Module, Wire};
use crate::buckyball::frontend::{GlobalRsComplete, MvinConfig, RobId};

/// MemLoader command request
#[derive(Clone, Default)]
pub struct MemLoaderReq {
  pub rob_id: RobId,
  pub config: MvinConfig,
}

/// MemLoader - Load instruction handler
pub struct MemLoader {
  name: String,

  // Input: Load request
  pub cmd_req: Wire<MemLoaderReq>,

  // Output: Completion signal
  pub cmd_resp: Wire<GlobalRsComplete>,

  // Internal state
  current_config: Option<MvinConfig>,
  current_rob_id: RobId,
  busy: bool,
}

impl MemLoader {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      cmd_req: Wire::default(),
      cmd_resp: Wire::default(),
      current_config: None,
      current_rob_id: 0,
      busy: false,
    }
  }

  pub fn is_busy(&self) -> bool {
    self.busy
  }
}

impl Module for MemLoader {
  fn run(&mut self) {
    // If not busy and new request arrives, accept it
    if !self.busy && self.cmd_req.valid {
      self.current_config = Some(self.cmd_req.value.config.clone());
      self.current_rob_id = self.cmd_req.value.rob_id;
      self.busy = true;
      self.cmd_resp.clear();
    }
    // If busy, simulate completion (in real implementation, this would wait for DMA)
    else if self.busy {
      // Signal completion
      self.cmd_resp.set(GlobalRsComplete {
        rob_id: self.current_rob_id,
        data: 0, // MVIN doesn't return data
      });
      self.busy = false;
      self.current_config = None;
    } else {
      self.cmd_resp.clear();
    }
  }

  fn reset(&mut self) {
    self.cmd_req = Wire::default();
    self.cmd_resp = Wire::default();
    self.current_config = None;
    self.current_rob_id = 0;
    self.busy = false;
  }

  fn name(&self) -> &str {
    &self.name
  }
}
