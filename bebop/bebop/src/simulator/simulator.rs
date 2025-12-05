/// Accelerator simulator with state management
use super::server::SocketServer;
use crate::buckyball::Top;
use crate::builtin::Sim;
use std::thread;

/// Execution mode for the simulator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepMode {
  #[default]
  Run,
  Step,
}

/// Accelerator simulator - drives Top module
pub struct Simulator {
  host: String,
  port: u16,
  step_mode: StepMode,
}

impl Simulator {
  pub fn new(host: impl Into<String>, port: u16, step_mode: StepMode) -> Self {
    Self {
      host: host.into(),
      port,
      step_mode,
    }
  }

  /// Run the simulator server
  pub fn run(self) -> std::io::Result<()> {
    let step_mode = matches!(self.step_mode, StepMode::Step);
    let mut buckyball = Top::new("buckyball");
    let cmd_tx = buckyball.get_cmd_sender();
    let cmd_response_rx = buckyball.get_cmd_response_receiver();
    let (dma_req_rx, dma_resp_tx) = buckyball.get_dma_channels();

    // Start server in background thread (lower priority)
    let server = SocketServer::new(self.host, self.port, step_mode, cmd_tx, cmd_response_rx, dma_req_rx, dma_resp_tx);
    thread::spawn(move || {
      if let Err(e) = server.run() {
        log_error!("Server error: {}", e);
      }
    });

    // Run buckyball in main thread (high priority)
    log_info!("Buckyball main loop started");

    if step_mode {
      // Step mode: wait for user input (press Enter) before each tick
      log_info!("Running in STEP mode - press Enter to tick");
      use std::io::{self, BufRead};
      let stdin = io::stdin();
      let mut lines = stdin.lock().lines();

      loop {
        // Wait for Enter key
        if lines.next().is_some() {
          buckyball.forward();
          buckyball.backward();
        } else {
          break; // EOF or error
        }
      }
    } else {
      // Free-running mode: tick as fast as possible
      loop {
        buckyball.forward();
        buckyball.backward();
      }
    }

    Ok(())
  }
}
