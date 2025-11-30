/// MemDomain - 内存域顶层模块
use crate::builtin::Module;
use super::mem::Bank;
use super::loader::MemLoader;
use super::storer::MemStorer;
use super::{DmaOperation, MvinConfig, MvoutConfig};

pub struct MemDomain {
  name: String,
  bank: Bank,
  loader: MemLoader,
  storer: MemStorer,
}

impl MemDomain {
  pub fn new(name: impl Into<String>, bank_size: usize) -> Self {
    Self {
      name: name.into(),
      bank: Bank::new("bank", bank_size),
      loader: MemLoader::new("loader"),
      storer: MemStorer::new("storer"),
    }
  }

  /// 发送内存指令
  pub fn issue(&mut self, funct: u64, xs1: u64, xs2: u64) {
    match funct {
      24 => self.loader.issue(MvinConfig::from_fields(xs1, xs2)),
      25 => self.storer.issue(MvoutConfig::from_fields(xs1, xs2)),
      _ => {}
    }
  }

  /// 获取当前 DMA 操作
  pub fn get_dma_operation(&self) -> Option<DmaOperation> {
    if let Some(config) = self.loader.get_current() {
      return Some(DmaOperation::Mvin(config.clone()));
    }
    if let Some(config) = self.storer.get_current() {
      return Some(DmaOperation::Mvout(config.clone()));
    }
    None
  }

  /// DMA 写入 scratchpad
  pub fn write_spad(&mut self, addr: usize, data: u32) {
    self.bank.init_write(addr, data);
  }

  /// DMA 读取 scratchpad
  pub fn read_spad(&self, addr: usize) -> u32 {
    self.bank.read_data(addr)
  }

  /// 初始化内存
  pub fn init_write(&mut self, addr: usize, data: u32) {
    self.bank.init_write(addr, data);
  }

  /// 获取数据
  pub fn get_data(&self) -> u32 {
    0
  }

  /// 完成当前 DMA 操作
  pub fn complete_dma(&mut self) {
    self.loader.complete();
    self.storer.complete();
  }
}

impl Module for MemDomain {
  fn tick(&mut self) {
    self.loader.tick();
    self.storer.tick();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
