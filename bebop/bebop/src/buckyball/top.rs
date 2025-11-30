/// Top Module - connects Frontend, MemDomain and BallDomain
use crate::builtin::Module;
use crate::buckyball::frontend::{Frontend, TargetDomain};
use crate::buckyball::memdomain::{DmaOperation, MemDomain};
use crate::buckyball::balldomain::BallDomain;

/// Top - NPU top-level module
pub struct Top {
  name: String,
  frontend: Frontend,
  memdomain: MemDomain,
  balldomain: BallDomain,
}

impl Top {
  pub fn new(name: impl Into<String>, mem_size: usize) -> Self {
    Self {
      name: name.into(),
      frontend: Frontend::new("frontend", 16),
      memdomain: MemDomain::new("memdomain", mem_size),
      balldomain: BallDomain::new("balldomain"),
    }
  }

  /// 发送指令
  pub fn issue(&mut self, funct: u64, xs1: u64, xs2: u64) {
    // Frontend 解码并分配
    if let Some(decoded) = self.frontend.issue(funct, xs1, xs2) {
      // 分发到对应域
      match decoded.target {
        TargetDomain::MemDomain => self.memdomain.issue(funct, xs1, xs2),
        TargetDomain::BallDomain => self.balldomain.issue(funct, xs1, xs2),
        TargetDomain::Unknown => {}
      }
    }
  }

  /// DMA write to scratchpad
  pub fn dma_write_spad(&mut self, addr: usize, data: u32) {
    self.memdomain.write_spad(addr, data);
  }

  /// DMA read from scratchpad
  pub fn dma_read_spad(&self, addr: usize) -> u32 {
    self.memdomain.read_spad(addr)
  }

  /// Initialize memory
  pub fn init_mem(&mut self, addr: usize, data: u32) {
    self.memdomain.init_write(addr, data);
  }

  /// Get DMA request (if any)
  pub fn get_dma_req(&self) -> Option<DmaOperation> {
    self.memdomain.get_dma_operation()
  }

  /// Get memory data result
  pub fn get_mem_data(&self) -> u32 {
    self.memdomain.get_data()
  }
}

impl Module for Top {
  fn tick(&mut self) {
    self.frontend.tick();
    self.memdomain.tick();
    self.balldomain.tick();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
