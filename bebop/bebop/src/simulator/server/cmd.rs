/// CMD channel handler - receives RoCC commands from host and sends to Top
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::{Arc, Mutex};

use crate::buckyball::top::{RoccCmd, CmdResponse};
use super::socket::cmd::{CmdReq, CmdResp};
use super::dma::DmaHandler;

pub struct CmdHandler {
  cmd_tx: Sender<RoccCmd>,
  cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>,
}

impl CmdHandler {
  pub fn new(cmd_tx: Sender<RoccCmd>, cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>) -> Self {
    Self { cmd_tx, cmd_response_rx }
  }

  /// Handle a single CMD request
  pub fn handle_cmd_request(&mut self, stream: &mut TcpStream, msg_type_val: u32, dma_handler: &mut DmaHandler) -> std::io::Result<()> {
    // Read remaining CMD request bytes (already read 4 bytes for msg_type)
    let mut remaining_bytes = [0u8; CmdReq::SIZE - 4];
    stream.read_exact(&mut remaining_bytes)?;

    // Reconstruct full message
    let mut msg_bytes = [0u8; CmdReq::SIZE];
    msg_bytes[0..4].copy_from_slice(&msg_type_val.to_le_bytes());
    msg_bytes[4..].copy_from_slice(&remaining_bytes);

    // Parse CMD request
    let cmd_req = CmdReq::from_bytes(&msg_bytes);
    let funct = cmd_req.funct;
    let xs1 = cmd_req.xs1;
    let xs2 = cmd_req.xs2;
    log_ipc!("[CMD] Received: funct={}, xs1=0x{:016x}, xs2=0x{:016x}", funct, xs1, xs2);

    // Forward command to Top via channel
    let rocc_cmd = RoccCmd { funct, xs1, xs2 };
    if let Err(e) = self.cmd_tx.send(rocc_cmd) {
      log_error!("[CMD] Failed to send to Top: {}", e);
      return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, e));
    }

    // Wait for cmd_response from Frontend
    // While waiting, send any pending DMA requests
    let result = loop {
      // Try to send DMA requests (non-blocking)
      dma_handler.try_send_dma_request(stream)?;

      // Try to receive cmd_response (non-blocking)
      let response_rx = self.cmd_response_rx.lock().unwrap();
      match response_rx.try_recv() {
        Ok(cmd_response) => {
          log_ipc!("[CMD] Received cmd_response from Frontend: result=0x{:016x}", cmd_response.result);
          break cmd_response.result;
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {
          // No response yet, continue loop
          drop(response_rx);
          std::thread::sleep(std::time::Duration::from_micros(100));
          continue;
        }
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
          log_error!("[CMD] cmd_response channel disconnected");
          return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "cmd_response channel disconnected"));
        }
      }
    };

    // Send CMD response to host
    let cmd_resp = CmdResp::new(result);
    let resp_bytes = cmd_resp.to_bytes();
    stream.write_all(&resp_bytes)?;
    log_ipc!("[CMD] Sent response to host: result=0x{:016x}\n", result);

    Ok(())
  }
}
