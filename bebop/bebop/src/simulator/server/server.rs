/// TCP server for accepting host connections
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Sender, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

use super::cmd::CmdHandler;
use super::dma::DmaHandler;
use super::socket::protocol::MsgType;
use crate::buckyball::top::{RoccCmd, DmaRequest, DmaResponse, CmdResponse};

pub struct SocketServer {
  host: String,
  port: u16,
  step_mode: bool,
  cmd_tx: Sender<RoccCmd>,
  cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>,
  dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
  dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
}

impl SocketServer {
  pub fn new(
    host: impl Into<String>,
    port: u16,
    step_mode: bool,
    cmd_tx: Sender<RoccCmd>,
    cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>,
    dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
    dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
  ) -> Self {
    Self {
      host: host.into(),
      port,
      step_mode,
      cmd_tx,
      cmd_response_rx,
      dma_req_rx,
      dma_resp_tx,
    }
  }

  /// Start the server and listen for connections
  pub fn run(&self) -> std::io::Result<()> {
    let addr = format!("{}:{}", self.host, self.port);

    log_info!("Socket server starting...");
    log_info!("  Listening on: {}", addr);
    log_info!("Waiting for host connections...\n");

    let listener = TcpListener::bind(&addr)?;
    log_info!("[Server] Listener ready on {}", addr);

    for stream in listener.incoming() {
      match stream {
        Ok(stream) => {
          let cmd_tx = self.cmd_tx.clone();
          let cmd_response_rx = self.cmd_response_rx.clone();
          let dma_req_rx = self.dma_req_rx.clone();
          let dma_resp_tx = self.dma_resp_tx.clone();

          thread::spawn(move || {
            if let Err(e) = Self::handle_connection(stream, cmd_tx, cmd_response_rx, dma_req_rx, dma_resp_tx) {
              log_error!("[Server] Connection handler error: {}", e);
            }
          });
        }
        Err(e) => {
          log_error!("[Server] Connection error: {}", e);
        }
      }
    }

    Ok(())
  }

  /// Handle a single connection, dispatching messages to CMD or DMA handlers
  fn handle_connection(
    mut stream: TcpStream,
    cmd_tx: Sender<RoccCmd>,
    cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>,
    dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
    dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
  ) -> std::io::Result<()> {
    use std::io::Read;

    let peer_addr = stream.peer_addr()?;
    log_info!("[Server] Connected from: {}", peer_addr);

    let mut cmd_handler = CmdHandler::new(cmd_tx, cmd_response_rx);
    let mut dma_handler = DmaHandler::new(dma_req_rx, dma_resp_tx);

    loop {
      // Read message from host (blocking)
      let mut msg_type_bytes = [0u8; 4];
      match stream.read_exact(&mut msg_type_bytes) {
        Ok(_) => {
          let msg_type_val = u32::from_le_bytes(msg_type_bytes);
          let msg_type = MsgType::from_u32(msg_type_val);

          match msg_type {
            Some(MsgType::CmdReq) => {
              cmd_handler.handle_cmd_request(&mut stream, msg_type_val, &mut dma_handler)?;
            }
            Some(MsgType::DmaReadResp) => {
              dma_handler.handle_dma_read_response(&mut stream, msg_type_val)?;
            }
            Some(MsgType::DmaWriteResp) => {
              dma_handler.handle_dma_write_response(&mut stream, msg_type_val)?;
            }
            _ => {
              log_error!("[Server] Unknown message type: {}", msg_type_val);
              return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown message type"));
            }
          }
        }
        Err(e) => {
          if e.kind() == std::io::ErrorKind::UnexpectedEof {
            log_info!("[Server] Client {} disconnected", peer_addr);
            return Ok(());
          } else {
            return Err(e);
          }
        }
      }
    }
  }
}
