/// Buckyball - Core NPU modules
///
/// 模块结构：
/// - `top` - 顶层模块，连接 Frontend, MemDomain 和 BallDomain
/// - `frontend` - 全局 Decoder 和 ReservationStation
/// - `memdomain` - 内存域，处理 MVIN/MVOUT 指令
/// - `balldomain` - Ball 域，处理计算指令
/// - `prototype` - Ball 原型实现
pub mod frontend;
pub mod memdomain;
pub mod balldomain;
pub mod prototype;
pub mod top;

// 对外导出
pub use crate::builtin::Module;
pub use frontend::{Frontend, TargetDomain};
pub use memdomain::{DmaOperation, MemDomain, MvinConfig, MvoutConfig};
pub use balldomain::BallDomain;
pub use top::Top;
