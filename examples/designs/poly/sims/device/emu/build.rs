use std::{env, path::PathBuf};

fn main() {
    let entry = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    bebop_sim_build::balls(&entry.join("../../../output/generated/design.pb"));
    bebop_sim_build::runner_links();
}
