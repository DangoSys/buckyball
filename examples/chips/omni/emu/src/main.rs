use bebop_bemu::root::{system, tile::run::Args};
use clap::{CommandFactory, FromArgMatches, Parser};

#[derive(Parser)]
struct SystemArgs {
    #[command(flatten)]
    core: Args,
    #[arg(long, requires = "pk")]
    host_io: bool,
}

fn main() {
    let config: toml::Value =
        toml::from_str(include_str!("../../configs/system.toml")).expect("Omni system config");
    let chips = config["chips"].as_array().expect("Omni chips");
    let memory = chips[0]["memory_mib"].as_integer().expect("DDR capacity");
    for (id, chip) in chips.iter().enumerate() {
        assert_eq!(chip["id"].as_integer(), Some(id as i64));
        assert_eq!(chip["memory_mib"].as_integer(), Some(memory));
    }
    let memory = Box::leak(memory.to_string().into_boxed_str());
    let matches = SystemArgs::command()
        .mut_arg("memory_mib", |arg| arg.default_value(&*memory))
        .get_matches();
    let args = SystemArgs::from_arg_matches(&matches).expect("Omni arguments");
    let vm_chip = if config["vm"]["enabled"].as_bool().expect("VM enabled") {
        Some(
            config["vm"]["chip"]
                .as_integer()
                .expect("VM chip")
                .try_into()
                .expect("VM chip range"),
        )
    } else {
        None
    };
    if let Err(error) = system::run(args.core, chips.len(), vm_chip, args.host_io) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
