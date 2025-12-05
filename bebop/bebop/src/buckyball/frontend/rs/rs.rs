use crate::log_backward;

pub struct Rs {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub rob_id: u64,
  pub domain_id: u64,
}

impl Rs {
  pub fn new() -> Self {
    Self { funct: 0, xs1: 0, xs2: 0, rob_id: 0, domain_id: 0 }
  }

  pub fn issue_cmd(&mut self, funct: u32, xs1: u64, xs2: u64, rob_id: u64, domain_id: u64) {
    self.funct = funct;
    self.xs1 = xs1;
    self.xs2 = xs2;
    self.rob_id = rob_id;
    self.domain_id = domain_id;
    log_backward!("Inst issue!");
  }

  pub fn print_status(&self) {
    println!("    [Rs] funct={}, xs1=0x{:x}, xs2=0x{:x}, rob_id={}, domain_id={}",
             self.funct, self.xs1, self.xs2, self.rob_id, self.domain_id);
  }
}
