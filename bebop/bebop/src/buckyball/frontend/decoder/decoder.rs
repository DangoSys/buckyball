use crate::log_backward;

pub struct Decoder {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub is_fence: bool,
}

impl Decoder {
  pub fn new() -> Self {
    Self { funct: 0, xs1: 0, xs2: 0, is_fence: false }
  }

  pub fn decode_cmd(&mut self, funct: u32, xs1: u64, xs2: u64) {
    self.funct = funct;
    self.xs1 = xs1;
    self.xs2 = xs2;
    self.is_fence = funct == 31;
    if self.is_fence {
      log_backward!("Fence instruction decoded!");
    } else {
      log_backward!("Inst decode!");
    }
  }

  pub fn print_status(&self) {
    println!("    [Decoder] funct={}, xs1=0x{:x}, xs2=0x{:x}, is_fence={}",
             self.funct, self.xs1, self.xs2, self.is_fence);
  }
}
