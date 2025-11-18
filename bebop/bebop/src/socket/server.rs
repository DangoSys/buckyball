/// TCP server for accepting Spike connections
use std::net::TcpListener;

use super::handler::ConnectionHandler;

pub struct SocketServer {
  host: String,
  port: u16,
}

impl SocketServer {
  pub fn new(host: impl Into<String>, port: u16) -> Self {
    Self {
      host: host.into(),
      port,
    }
  }

  /// Start the server and listen for connections
  pub fn run(&self) -> std::io::Result<()> {
    let addr = format!("{}:{}", self.host, self.port);
    let listener = TcpListener::bind(&addr)?;

    println!("Socket server listening on {}", addr);
    println!("Waiting for Spike connections...");
    println!("Press Ctrl+C to exit\n");

    for stream in listener.incoming() {
      match stream {
        Ok(stream) => {
          let handler = ConnectionHandler::new(stream);
          if let Err(e) = handler.handle() {
            eprintln!("Error handling client: {}", e);
          }
        },
        Err(e) => {
          eprintln!("Connection error: {}", e);
        },
      }
    }

    Ok(())
  }
}

impl Default for SocketServer {
  fn default() -> Self {
    Self::new("127.0.0.1", 9999)
  }
}
