/// Buckyball - Core NPU modules
///
/// - `top` - 顶层模块
/// - `config` - NPU 配置
/// - `frontend` - 前端模块
/// - `memdomain` - 内存域模块

pub mod config;
pub mod frontend;
pub mod memdomain;
pub mod top;

pub use crate::builtin::Sim;
pub use config::NpuConfig;
pub use memdomain::MemDomain;
pub use top::Top;
