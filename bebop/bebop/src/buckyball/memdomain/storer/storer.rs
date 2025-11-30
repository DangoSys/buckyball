/// MemStorer - Handles MVOUT (store from scratchpad to DRAM) operations
use crate::builtin::Module;
use super::super::MvoutConfig;

/// MemStorer - Store instruction handler
pub struct MemStorer {
  name: String,
  pending: Option<MvoutConfig>,
  current: Option<MvoutConfig>,
}

impl MemStorer {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      pending: None,
      current: None,
    }
  }

  /// 发送 MVOUT 请求
  pub fn issue(&mut self, config: MvoutConfig) {
    self.pending = Some(config);
  }

  /// 获取当前正在处理的配置（用于 DMA）
  pub fn get_current(&self) -> Option<&MvoutConfig> {
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

impl Module for MemStorer {
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
