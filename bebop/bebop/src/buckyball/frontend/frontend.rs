/// Frontend - 全局指令前端
use crate::builtin::Module;
use super::decoder::{GlobalDecoder, DecodedInstr};
use super::rs::GlobalRS;

pub struct Frontend {
  name: String,
  decoder: GlobalDecoder,
  rs: GlobalRS,
}

impl Frontend {
  pub fn new(name: impl Into<String>, rob_size: usize) -> Self {
    Self {
      name: name.into(),
      decoder: GlobalDecoder::new("decoder"),
      rs: GlobalRS::new("rs", rob_size),
    }
  }

  /// 解码并分配指令
  pub fn issue(&mut self, funct: u64, xs1: u64, xs2: u64) -> Option<DecodedInstr> {
    // 检查 ROB 是否满
    if self.rs.is_full() {
      return None;
    }

    // 解码指令
    let decoded = self.decoder.decode(funct, xs1, xs2);

    // 分配 ROB 条目
    let _rob_id = self.rs.allocate(funct, xs1, xs2);

    Some(decoded)
  }

  /// 提交完成的指令
  pub fn commit(&mut self, rob_id: usize) {
    self.rs.commit(rob_id);
  }
}

impl Module for Frontend {
  fn tick(&mut self) {
    // Frontend 目前不需要 tick 逻辑
  }

  fn name(&self) -> &str {
    &self.name
  }
}
