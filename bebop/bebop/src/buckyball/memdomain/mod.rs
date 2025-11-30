/// MemDomain - 内存域模块
mod mem;
mod loader;
mod storer;
mod decoder;
mod memdomain;

pub use decoder::{MvinConfig, MvoutConfig};

/// DMA 操作类型
#[derive(Debug, Clone)]
pub enum DmaOperation {
  Mvin(MvinConfig),
  Mvout(MvoutConfig),
}

pub use memdomain::MemDomain;
