use crate::log_backward;

pub struct OutController {
  pub bank_id: u32,

  // Completion tracking
  pub is_load: bool,
  pub is_store: bool,
  pub mem_addr: u64,
  pub sp_bank_addr: u32,
  pub iter: u32,
  pub stride: u32,
  pub task_complete: bool,
  pub rob_id: u32,  // ROB ID for tracking instruction completion
}

impl OutController {
  pub fn new() -> Self {
    Self {
      bank_id: 0,
      is_load: false,
      is_store: false,
      mem_addr: 0,
      sp_bank_addr: 0,
      iter: 0,
      stride: 0,
      task_complete: false,
      rob_id: 0,
    }
  }

  pub fn dma_schedule(&mut self, bank_id: u32, is_load: bool, is_store: bool,
    mem_addr: u64, sp_bank_addr: u32, iter: u32, stride: u32) {

    self.bank_id = bank_id;

    self.is_load = is_load;
    self.is_store = is_store;

    self.mem_addr = mem_addr;
    self.sp_bank_addr = sp_bank_addr;
    self.iter = iter;
    self.stride = stride;
  }

  pub fn print_status(&self) {
    println!("    [OutController] bank_id={}, task_complete={}, rob_id={}, op={}, mem_addr=0x{:x}, sp_bank_addr=0x{:x}, iter={}, stride={}",
             self.bank_id, self.task_complete, self.rob_id,
             if self.is_load { "mvin" } else if self.is_store { "mvout" } else { "idle" },
             self.mem_addr, self.sp_bank_addr, self.iter, self.stride);
  }
}
