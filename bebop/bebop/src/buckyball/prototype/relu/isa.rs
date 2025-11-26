/// ReLU Ball ISA
#[derive(Clone, Default, Debug)]
pub struct ReluCmd {
  pub op1_addr: u32,
  pub dst_addr: u32,
  pub iter: u32,
}

impl ReluCmd {
  pub fn from_fields(xs1: u64, xs2: u64) -> Self {
    Self {
      op1_addr: (xs1 & 0x3FFF) as u32,
      dst_addr: (xs2 & 0x3FFF) as u32,
      iter: ((xs2 >> 14) & 0x3FF) as u32,
    }
  }
}
