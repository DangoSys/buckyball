use crate::log_backward;

pub struct Rob {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub rob_id: u64,
  pub is_fence: bool,
  pub pending_fence_rob_id: Option<u64>,
  committed_rob_id: u64,
}

impl Rob {
  pub fn new() -> Self {
    Self {
      funct: 0,
      xs1: 0,
      xs2: 0,
      rob_id: 0,
      is_fence: false,
      pending_fence_rob_id: None,
      committed_rob_id: 0,
    }
  }

  pub fn enter_rob(&mut self, funct: u32, xs1: u64, xs2: u64, is_fence: bool) {
    self.funct = funct;
    self.xs1 = xs1;
    self.xs2 = xs2;
    self.is_fence = is_fence;
    self.rob_id += 1;

    if is_fence {
      self.pending_fence_rob_id = Some(self.rob_id);
      log_backward!("Fence instruction entered ROB! rob_id={}", self.rob_id);
    } else {
      log_backward!("Inst enter rob!");
    }
  }



  pub fn is_empty(&self) -> bool {
    self.committed_rob_id >= self.rob_id
  }

  pub fn commit_instruction(&mut self) {
    self.committed_rob_id += 1;
  }

  pub fn check_fence_ready(&mut self) -> Option<u64> {
    if let Some(fence_rob_id) = self.pending_fence_rob_id {
      if self.is_empty() {
        log_backward!("Fence instruction ready! ROB is empty, fence_rob_id={}", fence_rob_id);
        self.pending_fence_rob_id = None;
        return Some(fence_rob_id);
      }
    }
    None
  }

  pub fn print_status(&self) {
    println!("    [Rob] funct={}, xs1=0x{:x}, xs2=0x{:x}, rob_id={}, is_fence={}, pending_fence={:?}, committed={}",
             self.funct, self.xs1, self.xs2, self.rob_id, self.is_fence, self.pending_fence_rob_id, self.committed_rob_id);
  }
}
