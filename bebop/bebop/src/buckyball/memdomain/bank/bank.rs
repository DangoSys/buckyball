// Bank operation state
use crate::log_backward;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BankState {
  Idle,
  SramReq,     // Issuing SRAM request
  SramWait,    // Waiting for SRAM response
}

// Bank configuration (unified for all banks)
const NUM_BANKS: usize = 12;         // Total number of banks (e.g., 4 SP + 8 ACC)
const BANK_ENTRIES: usize = 4096;    // Entries per bank
const ENTRY_BYTES: usize = 16;       // 16 bytes per entry (128 bits)

pub struct Bank {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
  pub bank_id: u32,

  // State machine
  pub state: BankState,

  // Cached instruction parameters
  pub is_load: bool,
  pub is_store: bool,
  pub sp_bank: u32,
  pub sp_bank_addr: u32,
  pub iter: u32,

  // SRAM access tracking
  pub sram_count: u32,  // Current iteration counter
  pub data_buffer: Vec<u8>,  // Data buffer for streaming

  // Actual storage: Unified banks (NUM_BANKS x BANK_ENTRIES x ENTRY_BYTES)
  banks: Vec<Vec<Vec<u8>>>,
}

impl Bank {
  pub fn new() -> Self {
    // Initialize all banks: NUM_BANKS x BANK_ENTRIES x ENTRY_BYTES
    let banks = (0..NUM_BANKS)
      .map(|_| {
        (0..BANK_ENTRIES)
          .map(|_| vec![0u8; ENTRY_BYTES])
          .collect()
      })
      .collect();

    Self {
      funct: 0,
      xs1: 0,
      xs2: 0,
      bank_id: 0,
      state: BankState::Idle,
      is_load: false,
      is_store: false,
      sp_bank: 0,
      sp_bank_addr: 0,
      iter: 0,
      sram_count: 0,
      data_buffer: Vec::new(),
      banks,
    }
  }

  pub fn bank_read(&mut self, sp_bank: u32, sp_bank_addr: u32, iter: u32) {
    self.is_load = false;
    self.is_store = true;
    self.sp_bank = sp_bank;
    self.sp_bank_addr = sp_bank_addr;
    self.iter = iter;

    // Initialize state
    self.state = BankState::SramReq;
    self.sram_count = 0;
    self.bank_id += 1;

    // MVOUT: Read data from bank (to DMA)
    log_backward!(
      "Bank READ: mvout (bank_id={}, bank={}, addr=0x{:x}, iter={})",
      self.bank_id, self.sp_bank, self.sp_bank_addr, self.iter
    );
  }

  /// Read one entry from bank
  pub fn read_entry(&self, bank: u32, addr: u32) -> Vec<u8> {
    let bank_idx = bank as usize;
    let addr_idx = addr as usize;

    if bank_idx < NUM_BANKS && addr_idx < BANK_ENTRIES {
      self.banks[bank_idx][addr_idx].clone()
    } else {
      log_backward!("Bank READ ERROR: addr out of range (bank={}, addr={})", bank, addr);
      vec![0u8; ENTRY_BYTES]
    }
  }

  pub fn bank_write(&mut self, sp_bank: u32, sp_bank_addr: u32, iter: u32) {
    self.is_load = true;
    self.is_store = false;
    self.sp_bank = sp_bank;
    self.sp_bank_addr = sp_bank_addr;
    self.iter = iter;

    // Initialize state
    self.state = BankState::SramReq;
    self.sram_count = 0;
    self.bank_id += 1;

    // MVIN: Write data to bank (from DMA)
    log_backward!(
      "Bank WRITE: mvin (bank_id={}, bank={}, addr=0x{:x}, iter={})",
      self.bank_id, self.sp_bank, self.sp_bank_addr, self.iter
    );
  }

  /// Write one entry to bank
  pub fn write_entry(&mut self, bank: u32, addr: u32, data: Vec<u8>) {
    let bank_idx = bank as usize;
    let addr_idx = addr as usize;

    if bank_idx < NUM_BANKS && addr_idx < BANK_ENTRIES {
      // Ensure data is correct size
      if data.len() == ENTRY_BYTES {
        self.banks[bank_idx][addr_idx] = data;
        log_backward!("Bank WRITE: bank[{}][0x{:x}] = {} bytes", bank, addr, ENTRY_BYTES);
      } else {
        log_backward!("Bank WRITE ERROR: data size mismatch (expected {}, got {})", ENTRY_BYTES, data.len());
      }
    } else {
      log_backward!("Bank WRITE ERROR: addr out of range (bank={}, addr={})", bank, addr);
    }
  }

  pub fn print_status(&self) {
    println!("    [Bank] bank_id={}, state={:?}, sp_bank={}, sp_bank_addr=0x{:x}, iter={}",
             self.bank_id, self.state, self.sp_bank, self.sp_bank_addr, self.iter);
  }
}
