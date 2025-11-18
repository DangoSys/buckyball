/// Bebop - Accelerator Simulator
///
/// Main executable for running the Bebop accelerator simulator.
/// This program listens for custom instruction requests from Spike
/// and simulates accelerator behavior.
use bebop::SocketServer;

fn main() -> std::io::Result<()> {
  println!("Bebop Accelerator Simulator");
  println!("==========================\n");

  let server = SocketServer::default();
  server.run()
}
