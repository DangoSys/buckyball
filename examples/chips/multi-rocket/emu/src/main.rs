use bebop_bemu::root::tile::run::{run, Args};
use clap::Parser;

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
