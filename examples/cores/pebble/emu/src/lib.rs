pub const BEMU_TOP_CONFIG: &str = "../../core.toml";

pub fn bemu_top_config() -> &'static str {
    BEMU_TOP_CONFIG
}

mod chip;

include!("../../../../../bebop/src/nodes/bemu/src/lib.rs");
