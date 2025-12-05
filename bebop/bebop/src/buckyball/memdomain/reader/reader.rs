use crate::log_backward;
use crate::buckyball::top::DmaRequest;
use std::sync::mpsc::Sender;

pub struct Reader {
  pub mem_addr: u64,      // Memory address from xs1 (rs1[31:0])
  // pub bank_id: u32,       // Bank ID extracted from sp_addr
  // pub bank_addr: u32,  // Address within bank extracted from sp_addr
  pub iter: u32,          // Number of iterations (rs2[24:15])
  pub stride: u32,        // Stride/col_stride (rs2[33:24])

  dma_req_tx: Option<Sender<DmaRequest>>,
}

impl Reader {
  pub fn new() -> Self {
    Self {
      mem_addr: 0,
      iter: 0,
      stride: 0,
      dma_req_tx: None,
    }
  }

  pub fn set_dma_sender(&mut self, sender: Sender<DmaRequest>) {
    self.dma_req_tx = Some(sender);
  }

  pub fn dma_read(&mut self, mem_addr: u64, iter: u32, stride: u32) {
    self.mem_addr = mem_addr;
    self.iter = iter;
    self.stride = stride;

    println!("[Reader] dma_read called: mem_addr=0x{:x}, iter={}, stride={}", mem_addr, iter, stride);

    // Send DMA read request via channel
    if let Some(ref tx) = self.dma_req_tx {
      println!("[Reader] dma_req_tx channel exists, creating DmaRequest::Read");
      // For now, read 8 bytes (64-bit word)
      if mem_addr == 0 {
        eprintln!("[Reader] ERROR: mem_addr is 0!");
        return;
      }
      let req = DmaRequest::Read { addr: mem_addr, size: 8 };
      println!("[Reader] Sending DmaRequest::Read to channel...");
      if let Err(e) = tx.send(req) {
        eprintln!("[Reader] ERROR: Failed to send DMA read request: {}", e);
      } else {
        println!("[Reader] SUCCESS: Sent DMA read request to channel: addr=0x{:x}, size=8", mem_addr);
      }
    } else {
      println!("[Reader] ERROR: dma_req_tx channel is None!");
    }
  }

  pub fn print_status(&self) {
    println!("    [Reader] mem_addr=0x{:x}, iter={}, stride={}",
             self.mem_addr, self.iter, self.stride);
  }
}
