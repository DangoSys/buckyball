use crate::builtin::{Sim, EventQueue};
use super::{Decoder, Reader, Writer, Bank, OutController};
use crate::{log_forward};
use crate::buckyball::top::{DmaRequest, DmaResponse};
use std::sync::mpsc::{Sender, Receiver};

pub struct MemDomain {
  decoder: Decoder,
  reader: Reader,
  writer: Writer,
  bank: Bank,
  out_ctrl: OutController,
  pub event_queue: EventQueue<MemDomain>,

  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub rob_id: u32,  // ROB ID for tracking instruction completion

  // DMA channels
  dma_req_tx: Option<Sender<DmaRequest>>,
  dma_resp_rx: Option<Receiver<DmaResponse>>,
}

impl MemDomain {
  pub fn new() -> Self {
    Self {
      decoder: Decoder::new(),
      reader: Reader::new(),
      writer: Writer::new(),
      bank: Bank::new(),
      out_ctrl: OutController::new(),
      event_queue: EventQueue::new(),

      funct: 0,
      xs1: 0,
      xs2: 0,
      rob_id: 0,

      dma_req_tx: None,
      dma_resp_rx: None,
    }
  }

  pub fn set_dma_channels(&mut self, req_tx: Sender<DmaRequest>, resp_rx: Receiver<DmaResponse>) {
    self.dma_req_tx = Some(req_tx.clone());
    self.dma_resp_rx = Some(resp_rx);
    self.reader.set_dma_sender(req_tx.clone());
    self.writer.set_dma_sender(req_tx);
  }

  pub fn mem_cmd(&mut self) {
    let funct = self.funct;
    let xs1 = self.xs1;
    let xs2 = self.xs2;

    println!("[MemDomain] mem_cmd called: funct={}, xs1=0x{:x}, xs2=0x{:x}", funct, xs1, xs2);

    self.event_queue.push("MemCmd", move |memdomain: &mut MemDomain| {
      memdomain.decoder.decode_cmd(funct, xs1, xs2);
    });
    self.outside_schedule(funct);
  }

  pub fn outside_schedule(&mut self, funct: u32) {
    let mem_addr = self.decoder.mem_addr;
    let is_load = self.decoder.is_load;
    let is_store = self.decoder.is_store;
    let bank_id = self.decoder.sp_bank;
    let sp_bank_addr = self.decoder.sp_bank_addr;
    let iter = self.decoder.iter;
    let stride = self.decoder.stride;

    println!("[MemDomain] outside_schedule: is_load={}, is_store={}, mem_addr=0x{:x}", is_load, is_store, mem_addr);

    self.event_queue.push("ChooseDma", move |memdomain: &mut MemDomain| {
      memdomain.out_ctrl.dma_schedule( bank_id, is_load, is_store,
        mem_addr, sp_bank_addr, iter, stride);
    });
    log_forward!("MemDomain: Drive DMA (mem_addr=0x{:x}, iter={}, stride={})",
                 mem_addr, iter, stride);
    if funct == 24 {
      println!("[MemDomain] Calling dma_read()");
      self.dma_read();
    } else if funct == 25 {
      println!("[MemDomain] Calling dma_write()");
      self.dma_write();
    } else {
      println!("[MemDomain] Not Calling dma");
    }
  }

  pub fn dma_read(&mut self) {
    // MVIN: DMA read from memory
    let mem_addr = self.out_ctrl.mem_addr;
    let iter = self.out_ctrl.iter;
    let stride = self.out_ctrl.stride;

    println!("[MemDomain] dma_read: pushing DmaRead event to queue (mem_addr=0x{:x})", mem_addr);

    self.event_queue.push("DmaRead", move |memdomain: &mut MemDomain| {
      println!("[MemDomain] DmaRead event executing, calling reader.dma_read()");
      memdomain.reader.dma_read(mem_addr, iter, stride);
    });
    log_forward!("MemDomain: DMA read request");
    self.bank_write();
  }

  pub fn dma_write(&mut self) {
    // MVOUT: First read from bank, then DMA write to memory
    self.bank_read();

    // MVOUT: DMA write to memory after reading from bank
    let mem_addr = self.out_ctrl.mem_addr;
    let iter = self.out_ctrl.iter;
    let stride = self.out_ctrl.stride;

    self.event_queue.push("DmaWrite", move |memdomain: &mut MemDomain| {
      memdomain.writer.dma_write(mem_addr, iter, stride);
    });
    log_forward!("MemDomain: DMA write request");
  }

  pub fn bank_read(&mut self) {
    // MVOUT: Read data from bank
    let bank_id = self.out_ctrl.bank_id;
    let sp_bank_addr = self.out_ctrl.sp_bank_addr;
    let iter = self.out_ctrl.iter;

    self.event_queue.push("BankRead", move |memdomain: &mut MemDomain| {
      memdomain.bank.bank_read(bank_id, sp_bank_addr, iter);
    });
    log_forward!("MemDomain: Bank read for mvout");
  }

  pub fn bank_write(&mut self) {
    // MVIN: Write data to bank from DMA
    let bank_id = self.out_ctrl.bank_id;
    let sp_bank_addr = self.out_ctrl.sp_bank_addr;
    let iter = self.out_ctrl.iter;

    self.event_queue.push("BankWrite", move |memdomain: &mut MemDomain| {
      memdomain.bank.bank_write(bank_id, sp_bank_addr, iter);
    });
    log_forward!("MemDomain: Bank write for mvin");
  }

}

impl Sim for MemDomain {
  fn forward(&mut self) {
    // Forward phase is triggered by external command
    // Actual forward logic is in mem_cmd()
  }

  fn backward(&mut self) {
    // Process all events in the queue (LIFO)
    println!("[MemDomain] backward: processing event queue");
    let mut queue = std::mem::take(&mut self.event_queue);
    queue.process_all(self);
    self.event_queue = queue;
    println!("[MemDomain] backward: event queue processed");
  }

  fn module_name(&self) -> &str {
    "MemDomain"
  }

  fn print_status(&self) {
    println!("  [MemDomain] Module Status:");
    self.decoder.print_status();
    self.out_ctrl.print_status();
    self.reader.print_status();
    self.writer.print_status();
    self.bank.print_status();
  }
}
