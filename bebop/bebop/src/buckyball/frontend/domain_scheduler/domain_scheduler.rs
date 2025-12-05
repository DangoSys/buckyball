use crate::log_backward;

pub struct DomainScheduler {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub rob_id: u64,
  pub domain_id: u64,
}

impl DomainScheduler {
  pub fn new() -> Self {
    Self { funct: 0, xs1: 0, xs2: 0, rob_id: 0, domain_id: 0 }
  }

  pub fn dispatch_cmd(&mut self, funct: u32, xs1: u64, xs2: u64, rob_id: u64) {
    self.funct = funct;
    self.xs1 = xs1;
    self.xs2 = xs2;
    self.rob_id = rob_id;

    // Allocate domain_id based on instruction type:
    // - funct=24 (mvin) or funct=25 (mvout) -> domain_id=0 (MemDomain)
    // - Other instructions -> domain_id=1 (other domains)
    self.domain_id = match funct {
      0 => 0,       // unknown instruction
      24 | 25 => 1,  // mvin/mvout -> MemDomain
      _ => 2,        // other instructions
    };

    log_backward!("Inst dispatch (funct={}, domain_id={})!", funct, self.domain_id);
  }

  pub fn print_status(&self) {
    println!("    [DomainScheduler] funct={}, xs1=0x{:x}, xs2=0x{:x}, rob_id={}, domain_id={}",
             self.funct, self.xs1, self.xs2, self.rob_id, self.domain_id);
  }
}
