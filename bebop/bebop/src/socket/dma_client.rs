/// DMA client for sending DMA read/write requests to Spike
use std::io::{Read, Write};
use std::net::TcpStream;

use super::protocol::{DmaReadReq, DmaReadResp, DmaWriteReq, DmaWriteResp};

pub struct DmaClient<'a> {
  stream: &'a mut TcpStream,
}

impl<'a> DmaClient<'a> {
  pub fn new(stream: &'a mut TcpStream) -> Self {
    Self { stream }
  }

  /// Send DMA read request and wait for response
  pub fn dma_read(&mut self, addr: u64, size: u32) -> std::io::Result<u64> {
    // Send DMA read request
    let req = DmaReadReq::new(addr, size);
    let req_bytes = req.to_bytes();
    self.stream.write_all(&req_bytes)?;

    println!("  [DMA] Read request: addr=0x{:x}, size={}", addr, size);

    // Wait for DMA read response
    let mut resp_bytes = [0u8; DmaReadResp::SIZE];
    self.stream.read_exact(&mut resp_bytes)?;

    let resp = DmaReadResp::from_bytes(&resp_bytes);
    let data = resp.data;

    println!("  [DMA] Read response: data=0x{:x}", data);

    Ok(data)
  }

  /// Send DMA write request and wait for response
  pub fn dma_write(&mut self, addr: u64, data: u64, size: u32) -> std::io::Result<()> {
    // Send DMA write request
    let req = DmaWriteReq::new(addr, data, size);
    let req_bytes = req.to_bytes();
    self.stream.write_all(&req_bytes)?;

    println!(
      "  [DMA] Write request: addr=0x{:x}, data=0x{:x}, size={}",
      addr, data, size
    );

    // Wait for DMA write response
    let mut resp_bytes = [0u8; DmaWriteResp::SIZE];
    self.stream.read_exact(&mut resp_bytes)?;

    let _resp = DmaWriteResp::from_bytes(&resp_bytes);

    println!("  [DMA] Write response received");

    Ok(())
  }

  /// Read multiple 64-bit words from DRAM
  pub fn read_block(&mut self, base_addr: u64, count: usize) -> std::io::Result<Vec<u64>> {
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
      let addr = base_addr + (i as u64) * 8;
      let value = self.dma_read(addr, 8)?;
      data.push(value);
    }
    Ok(data)
  }

  /// Write multiple 64-bit words to DRAM
  pub fn write_block(&mut self, base_addr: u64, data: &[u64]) -> std::io::Result<()> {
    for (i, &value) in data.iter().enumerate() {
      let addr = base_addr + (i as u64) * 8;
      self.dma_write(addr, value, 8)?;
    }
    Ok(())
  }
}
