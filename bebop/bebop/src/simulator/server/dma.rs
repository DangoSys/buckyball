/// DMA channel handler - processes DMA requests from MemDomain and sends to host via TCP
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::buckyball::top::{DmaRequest, DmaResponse};
use super::socket::read::{DmaReadReq, DmaReadResp};
use super::socket::write::{DmaWriteReq, DmaWriteResp};

pub struct DmaHandler {
  dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
  dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
}

impl DmaHandler {
  pub fn new(
    dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
    dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
  ) -> Self {
    Self {
      dma_req_rx,
      dma_resp_tx,
    }
  }

  /// Try to send a DMA request to host (non-blocking check)
  /// Returns true if a request was sent, false if no request available
  pub fn try_send_dma_request(&mut self, stream: &mut TcpStream) -> std::io::Result<bool> {
    // Non-blocking check for DMA request from MemDomain
    let rx = self.dma_req_rx.lock().unwrap();
    match rx.try_recv() {
      Ok(dma_req) => {
        drop(rx); // Release lock before I/O

        match dma_req {
          DmaRequest::Read { addr, size } => {
            println!("[DMA] Sending Read request: addr=0x{:x}, size={}", addr, size);
            let req = DmaReadReq::new(addr, size);
            let req_bytes = req.to_bytes();
            stream.write_all(&req_bytes)?;
            log_ipc!("[DMA] Read request sent: addr=0x{:x}, size={}", addr, size);
          }
          DmaRequest::Write { addr, data, size } => {
            println!("[DMA] Sending Write request: addr=0x{:x}, data=0x{:x}, size={}", addr, data, size);
            let req = DmaWriteReq::new(addr, data, size);
            let req_bytes = req.to_bytes();
            stream.write_all(&req_bytes)?;
            log_ipc!("[DMA] Write request sent: addr=0x{:x}, data=0x{:x}, size={}", addr, data, size);
          }
        }
        Ok(true)
      }
      Err(std::sync::mpsc::TryRecvError::Empty) => {
        Ok(false) // No request available
      }
      Err(std::sync::mpsc::TryRecvError::Disconnected) => {
        Err(std::io::Error::new(
          std::io::ErrorKind::BrokenPipe,
          "DMA request channel disconnected",
        ))
      }
    }
  }

  /// Handle DMA read response from host
  pub fn handle_dma_read_response(&mut self, stream: &mut TcpStream, msg_type_val: u32) -> std::io::Result<()> {
    // Read remaining DMA read response bytes
    let mut remaining_bytes = [0u8; DmaReadResp::SIZE - 4];
    stream.read_exact(&mut remaining_bytes)?;

    // Reconstruct full message
    let mut msg_bytes = [0u8; DmaReadResp::SIZE];
    msg_bytes[0..4].copy_from_slice(&msg_type_val.to_le_bytes());
    msg_bytes[4..].copy_from_slice(&remaining_bytes);

    let resp = DmaReadResp::from_bytes(&msg_bytes);
    let data = resp.data;
    log_ipc!("[DMA] Read response received: data=0x{:x}\n", data);

    // Send response back to MemDomain via channel
    let dma_resp = DmaResponse::ReadComplete { data };
    let tx = self.dma_resp_tx.lock().unwrap();
    if let Err(e) = tx.send(dma_resp) {
      log_error!("[DMA] Failed to send response to MemDomain: {}", e);
      return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, e));
    }

    Ok(())
  }

  /// Handle DMA write response from host
  pub fn handle_dma_write_response(&mut self, stream: &mut TcpStream, msg_type_val: u32) -> std::io::Result<()> {
    // Read remaining DMA write response bytes
    let mut remaining_bytes = [0u8; DmaWriteResp::SIZE - 4];
    stream.read_exact(&mut remaining_bytes)?;

    log_ipc!("[DMA] Write response received\n");

    // Send response back to MemDomain via channel
    let dma_resp = DmaResponse::WriteComplete;
    let tx = self.dma_resp_tx.lock().unwrap();
    if let Err(e) = tx.send(dma_resp) {
      log_error!("[DMA] Failed to send response to MemDomain: {}", e);
      return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, e));
    }

    Ok(())
  }
}
