#[path = "../../../../../../../../../../../../../bebop/src/nodes/bemu/src/core/riscv/trap.rs"]
mod trap;
#[path = "../../../../../../../../../../../../../bebop/src/nodes/bemu/src/core/riscv/compressed.rs"]
mod compressed;
pub use trap::Trap;
pub use compressed::decode as expand_compressed;
