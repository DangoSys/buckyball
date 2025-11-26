/// Vector Ball ISA - instruction set for vector operations
/// Only visible within vector ball module

#[derive(Clone, Default, Debug)]
pub struct VecCmd {
  pub op1_addr: u32,
  pub op2_addr: u32,
  pub dst_addr: u32,
  pub iter: u32,
}

impl VecCmd {
  pub fn from_fields(xs1: u64, xs2: u64) -> Self {
    Self {
      op1_addr: (xs1 & 0x3FFF) as u32,
      op2_addr: ((xs1 >> 14) & 0x3FFF) as u32,
      dst_addr: (xs2 & 0x3FFF) as u32,
      iter: ((xs2 >> 14) & 0x3FF) as u32,
    }
  }
}
