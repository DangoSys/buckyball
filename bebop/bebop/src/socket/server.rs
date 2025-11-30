/// TCP server for accepting Spike connections
use std::net::TcpListener;

use super::handler::ConnectionHandler;

pub struct SocketServer {
  host: String,
  port: u16,
  step_mode: bool,
}

impl SocketServer {
  pub fn new(host: impl Into<String>, port: u16, step_mode: bool) -> Self {
    Self {
      host: host.into(),
      port,
      step_mode,
    }
  }

  /// Start the server and listen for connections
  pub fn run(&self) -> std::io::Result<()> {
    let addr = format!("{}:{}", self.host, self.port);
    let listener = TcpListener::bind(&addr)?;

    log_info!("Socket server listening on {}", addr);
    log_info!("Waiting for host simulator connections...");
    log_info!("Press Ctrl+C to exit\n");

    for stream in listener.incoming() {
      match stream {
        Ok(stream) => {
          let handler = ConnectionHandler::new(stream, self.step_mode);
          if let Err(e) = handler.handle() {
            log_error!("Error handling client: {}", e);
          }
        },
        Err(e) => {
          log_error!("Connection error: {}", e);
        },
      }
    }

    Ok(())
  }
}

impl Default for SocketServer {
  fn default() -> Self {
    Self::new("127.0.0.1", 9999, false)
  }
}
