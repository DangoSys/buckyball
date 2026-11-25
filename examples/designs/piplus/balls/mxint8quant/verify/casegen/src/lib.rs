mod casegen;
mod config;
mod dpi;
// The cycle model of BEMU, checked against the RTL by every UVM test.
#[path = "../../../emu/src/latency.rs"]
mod latency;
