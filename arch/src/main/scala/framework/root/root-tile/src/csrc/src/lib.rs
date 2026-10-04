// Reuse the actual leaf reference implementations, including their exported DPI API.
#[path = "../../../../../mem-core/coherence/src/csrc/src/memory.rs"]
mod line_memory;
#[path = "../../../../../mem-core/mmu/src/csrc/src/translation.rs"]
mod mmu;
#[path = "../../../../../mem-core/cpu_mem/src/csrc/src/access.rs"]
mod cpu_mem;

#[path = "../../../../root-core/src/csrc/src/memory.rs"]
mod core_memory;

mod consistency;

#[path = "../../../../../mem-core/ddr/src/csrc/src/memory.rs"]
mod ddr_memory;
