mod memory;
mod tile_peer;
pub use memory::*;
#[path = "../../../../../mem-core/ddr/src/csrc/src/lib.rs"]
mod ddr_reference;

#[no_mangle]
pub extern "C" fn admission_trace_init() {
    bebop_rtl_trace::init_trace(std::path::Path::new("build/admission_trace"),
        bebop_rtl_trace::TraceConfig { itrace: true, mtrace: true, pmctrace: true,
            ctrace: false, banktrace: false }).expect("initialize AdmissionSystem trace");
}
