/// TCP connection handler for Spike communication
use std::io::{Read, Write};
use std::net::TcpStream;

use super::dma_client::DmaClient;
use super::protocol::{CmdReq, CmdResp};
use crate::config::NpuConfig;
use crate::simulator::Simulator;

pub struct ConnectionHandler {
  stream: TcpStream,
  simulator: Simulator,
}

impl ConnectionHandler {
  pub fn new(stream: TcpStream) -> Self {
    let config = NpuConfig::new();
    Self {
      stream,
      simulator: Simulator::new(config),
    }
  }

  /// Handle the client connection loop
  pub fn handle(mut self) -> std::io::Result<()> {
    let peer_addr = self.stream.peer_addr()?;
    println!("New connection from: {}", peer_addr);

    loop {
      // Read CMD request
      let mut msg_bytes = [0u8; CmdReq::SIZE];
      match self.stream.read_exact(&mut msg_bytes) {
        Ok(_) => {},
        Err(e) => {
          if e.kind() == std::io::ErrorKind::UnexpectedEof {
            println!("Client {} disconnected", peer_addr);
            return Ok(());
          }
          return Err(e);
        },
      }

      // Parse CMD request
      let cmd_req = CmdReq::from_bytes(&msg_bytes);

      // Copy fields to avoid packed struct alignment issues
      let funct = cmd_req.funct;
      let xs1 = cmd_req.xs1;
      let xs2 = cmd_req.xs2;

      println!("Received CMD: funct={}, xs1=0x{:016x}, xs2=0x{:016x}", funct, xs1, xs2);

      // Create DMA client for this request
      let mut dma_client = DmaClient::new(&mut self.stream);

      // Process instruction with DMA client
      let result = self.simulator.process(funct, xs1, xs2, &mut dma_client)?;

      // Send CMD response
      let cmd_resp = CmdResp::new(result);
      let resp_bytes = cmd_resp.to_bytes();
      self.stream.write_all(&resp_bytes)?;
      println!("Sent CMD response: result=0x{:016x}\n", result);
    }
  }
}
