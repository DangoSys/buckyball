/// BallDomain - Ball 计算域顶层模块
use crate::builtin::Module;
use super::bbus::BBus;

pub struct BallDomain {
  name: String,
  bbus: BBus,
  pending: Option<(u64, u64, u64)>,
}

impl BallDomain {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      bbus: BBus::new("bbus"),
      pending: None,
    }
  }

  /// 发送指令
  pub fn issue(&mut self, funct: u64, xs1: u64, xs2: u64) {
    self.pending = Some((funct, xs1, xs2));
  }
}

impl Module for BallDomain {
  fn tick(&mut self) {
    if let Some((funct, xs1, xs2)) = self.pending.take() {
      self.bbus.issue(funct, xs1, xs2);
    }
    self.bbus.tick();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
