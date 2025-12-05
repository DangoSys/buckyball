/// Bebop - Accelerator Simulator
///
/// Main executable for running the Bebop accelerator simulator.
/// This program listens for custom instruction requests from Host
/// and simulates accelerator behavior.
use bebop::{log_info, Simulator, StepMode};
use std::env;

fn main() -> std::io::Result<()> {
  let args: Vec<String> = env::args().collect();
  let step_mode = if args.iter().any(|arg| arg == "--step" || arg == "-s") {
    StepMode::Step
  } else {
    StepMode::Run
  };

  if step_mode == StepMode::Step {
    log_info!("Bebop Accelerator Simulator (Step Mode)");
    log_info!("Commands: Enter=step, r=run, q=quit");
  }

  let simulator = Simulator::new("127.0.0.1", 9999, step_mode);
  simulator.run()
}
