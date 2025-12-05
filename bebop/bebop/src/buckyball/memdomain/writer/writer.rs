use crate::log_backward;
use crate::buckyball::top::DmaRequest;
use std::sync::mpsc::Sender;

pub struct Writer {
  pub mem_addr: u64,      // Memory address from xs1 (rs1[31:0])
  pub bank_id: u32,       // Bank ID extracted from sp_addr
  pub bank_addr: u32,  // Address within bank extracted from sp_addr
  pub iter: u32,          // Number of iterations (rs2[24:15])
  pub stride: u32,        // Stride/col_stride (rs2[33:24])

  dma_req_tx: Option<Sender<DmaRequest>>,
}

impl Writer {
  pub fn new() -> Self {
    Self {
      mem_addr: 0,
      bank_id: 0,
      bank_addr: 0,
      iter: 0,
      stride: 0,
      dma_req_tx: None,
    }
  }

  pub fn set_dma_sender(&mut self, sender: Sender<DmaRequest>) {
    self.dma_req_tx = Some(sender);
  }

  pub fn dma_write(&mut self, mem_addr: u64, iter: u32, stride: u32) {
    self.mem_addr = mem_addr;
    self.iter = iter;
    self.stride = stride;

    // Send DMA write request via channel
    if let Some(ref tx) = self.dma_req_tx {
      // For now, write dummy data (0x0) as 8 bytes (64-bit word)
      // In a real implementation, this would read data from the bank
      let req = DmaRequest::Write { addr: mem_addr, data: 0x0, size: 8 };
      if let Err(e) = tx.send(req) {
        eprintln!("[Writer] Failed to send DMA write request: {}", e);
      } else {
        println!("[Writer] Sent DMA write request: addr=0x{:x}, data=0x0, size=8", mem_addr);
      }
    }
  }

  pub fn print_status(&self) {
    println!("    [Writer] mem_addr=0x{:x}, bank_id={}, bank_addr=0x{:x}, iter={}, stride={}",
             self.mem_addr, self.bank_id, self.bank_addr, self.iter, self.stride);
  }
}
