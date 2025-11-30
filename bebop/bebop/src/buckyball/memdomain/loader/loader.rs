/// MemLoader - Handles MVIN (load from DRAM to scratchpad) operations
use crate::builtin::Module;
use super::super::MvinConfig;

/// MemLoader - Load instruction handler
pub struct MemLoader {
  name: String,
  pending: Option<MvinConfig>,
  current: Option<MvinConfig>,
}

impl MemLoader {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      pending: None,
      current: None,
    }
  }

  /// 发送 MVIN 请求
  pub fn issue(&mut self, config: MvinConfig) {
    self.pending = Some(config);
  }

  /// 获取当前正在处理的配置（用于 DMA）
  pub fn get_current(&self) -> Option<&MvinConfig> {
    self.current.as_ref()
  }

  /// 完成当前操作
  pub fn complete(&mut self) {
    self.current = None;
  }

  pub fn is_busy(&self) -> bool {
    self.current.is_some()
  }
}

impl Module for MemLoader {
  fn tick(&mut self) {
    // 如果没有正在处理的请求，接受新请求
    if self.current.is_none() {
      if let Some(config) = self.pending.take() {
        self.current = Some(config);
      }
    }
  }

  fn name(&self) -> &str {
    &self.name
  }
}
