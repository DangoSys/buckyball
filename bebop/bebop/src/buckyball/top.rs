/// Top Module - NPU top-level
use crate::builtin::Sim;
use crate::buckyball::frontend::Frontend;
use crate::buckyball::memdomain::MemDomain;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use crate::log_tpc;

/// RoCC command from host
pub struct RoccCmd {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
}

/// Command response notification (sent from Frontend to CmdHandler)
pub struct CmdResponse {
  pub result: u64,
}

/// Memory command sent from Frontend to MemDomain
#[derive(Debug, Clone)]
pub struct MemCmd {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub rob_id: u64,
  pub domain_id: u64,
}

/// DMA request types
#[derive(Debug, Clone)]
pub enum DmaRequest {
  Read { addr: u64, size: u32 },
  Write { addr: u64, data: u64, size: u32 },
}

/// DMA response types
#[derive(Debug, Clone)]
pub enum DmaResponse {
  ReadComplete { data: u64 },
  WriteComplete,
  Error(String),
}

/// Top - NPU top-level module
pub struct Top {
  name: String,
  frontend: Frontend,
  memdomain: MemDomain,

  // RoCC command channel (host -> Top)
  cmd_rx: Receiver<RoccCmd>,
  cmd_tx: Sender<RoccCmd>,

  // Memory command channel (Frontend -> MemDomain)
  #[allow(dead_code)]
  mem_cmd_tx: Sender<MemCmd>,
  mem_cmd_rx: Receiver<MemCmd>,

  // Command response channel (Frontend -> CmdHandler)
  pub cmd_response_rx: Arc<Mutex<Receiver<CmdResponse>>>,
  cmd_response_tx: Sender<CmdResponse>,

  // DMA request/response channels (MemDomain <-> ConnectionHandler)
  pub dma_req_rx: Arc<Mutex<Receiver<DmaRequest>>>,
  pub dma_resp_tx: Arc<Mutex<Sender<DmaResponse>>>,
}

impl Top {
  pub fn new(name: impl Into<String>) -> Self {
    let (cmd_tx, cmd_rx) = channel();
    let (mem_cmd_tx, mem_cmd_rx) = channel();
    let (cmd_response_tx, cmd_response_rx) = channel();
    let (dma_req_tx, dma_req_rx) = channel();
    let (dma_resp_tx, dma_resp_rx) = channel();

    let mut frontend = Frontend::new();
    frontend.set_mem_cmd_sender(mem_cmd_tx.clone());
    frontend.set_cmd_response_sender(cmd_response_tx.clone());

    let mut memdomain = MemDomain::new();
    memdomain.set_dma_channels(dma_req_tx, dma_resp_rx);

    Self {
      name: name.into(),
      frontend,
      memdomain,
      cmd_rx,
      cmd_tx,
      mem_cmd_tx,
      mem_cmd_rx,
      cmd_response_rx: Arc::new(Mutex::new(cmd_response_rx)),
      cmd_response_tx,
      dma_req_rx: Arc::new(Mutex::new(dma_req_rx)),
      dma_resp_tx: Arc::new(Mutex::new(dma_resp_tx)),
    }
  }

  pub fn frontend_cmd(&mut self, funct: u32, xs1: u64, xs2: u64) {
    self.frontend.funct = funct;
    self.frontend.xs1   = xs1;
    self.frontend.xs2   = xs2;
    self.frontend.rocc_cmd();
  }

  pub fn memdomain_cmd(&mut self, funct: u32, xs1: u64, xs2: u64, rob_id: u64, domain_id: u64) {
    self.memdomain.funct = funct;
    self.memdomain.xs1   = xs1;
    self.memdomain.xs2   = xs2;
    self.memdomain.mem_cmd();
  }

  /// Get a sender for sending RoCC commands
  pub fn get_cmd_sender(&self) -> Sender<RoccCmd> {
    self.cmd_tx.clone()
  }

  /// Get command response receiver for CmdHandler
  pub fn get_cmd_response_receiver(&self) -> Arc<Mutex<Receiver<CmdResponse>>> {
    self.cmd_response_rx.clone()
  }

  /// Get DMA channel handles for ConnectionHandler
  pub fn get_dma_channels(&self) -> (Arc<Mutex<Receiver<DmaRequest>>>, Arc<Mutex<Sender<DmaResponse>>>) {
    (self.dma_req_rx.clone(), self.dma_resp_tx.clone())
  }
}

impl Sim for Top {
  fn forward(&mut self) {
    // 1. Receive RoCC command from host
    if let Ok(rocc_cmd) = self.cmd_rx.try_recv() {
      self.frontend_cmd(rocc_cmd.funct, rocc_cmd.xs1, rocc_cmd.xs2);
    }

    // 2. Check if Frontend sent memory commands to MemDomain
    if let Ok(mem_cmd) = self.mem_cmd_rx.try_recv() {
      log_tpc!("[Top] Received MemCmd from Frontend: funct={}, xs1=0x{:x}, xs2=0x{:x}, rob_id={}, domain_id={}",
                mem_cmd.funct, mem_cmd.xs1, mem_cmd.xs2, mem_cmd.rob_id, mem_cmd.domain_id);
      self.memdomain_cmd(mem_cmd.funct, mem_cmd.xs1, mem_cmd.xs2, mem_cmd.rob_id, mem_cmd.domain_id);
    }

    // Debug: print queue status
    self.frontend.event_queue.print_status("Frontend");
    self.memdomain.event_queue.print_status("MemDomain");
  }

  fn backward(&mut self) {
    self.frontend.backward();
    self.memdomain.backward();

    // Print all module status
    println!("\n=== Module Status ===");
    self.frontend.print_status();
    self.memdomain.print_status();
    println!("====================\n");
  }

  fn module_name(&self) -> &str {
    &self.name
  }
}
