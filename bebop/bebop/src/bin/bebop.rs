/// Bebop - Accelerator Simulator
///
/// Main executable for running the Bebop accelerator simulator.
/// This program listens for custom instruction requests from Spike
/// and simulates accelerator behavior.
use bebop::SocketServer;
use std::env;

fn main() -> std::io::Result<()> {
  let args: Vec<String> = env::args().collect();
  let step_mode = args.iter().any(|arg| arg == "--step" || arg == "-s");

  if step_mode {
    println!("Bebop Accelerator Simulator (Step Mode)");
    println!("========================================");
    println!("Commands: Enter=step, r=run, q=quit\n");
  }

  let server = SocketServer::new("127.0.0.1", 9999, step_mode);
  server.run()
}
