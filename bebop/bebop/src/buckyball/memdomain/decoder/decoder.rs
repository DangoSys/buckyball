use crate::log_backward;

pub struct Decoder {
  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,

  // Decoded instruction fields
  pub is_load: bool,
  pub is_store: bool,
  pub mem_addr: u64,      // Memory address from xs1 (rs1[31:0])
  pub sp_addr: u32,       // Scratchpad address (rs2[14:0]) - linear address
  pub sp_bank: u32,       // Bank ID extracted from sp_addr
  pub sp_bank_addr: u32,  // Address within bank extracted from sp_addr
  pub iter: u32,          // Number of iterations (rs2[24:15])
  pub stride: u32,        // Stride/col_stride (rs2[33:24])
}

impl Decoder {
  pub fn new() -> Self {
    Self {
      funct: 0,
      xs1: 0,
      xs2: 0,
      is_load: false,
      is_store: false,
      mem_addr: 0,
      sp_addr: 0,
      sp_bank: 0,
      sp_bank_addr: 0,
      iter: 0,
      stride: 0,
    }
  }

  pub fn decode_cmd(&mut self, funct: u32, xs1: u64, xs2: u64) {
    self.funct = funct;
    self.xs1 = xs1;
    self.xs2 = xs2;

    // Decode instruction type
    self.is_load = funct == 24;   // mvin
    self.is_store = funct == 25;  // mvout

    // Parse xs1: memory address (DRAM address)
    // rs1[31:0] = base_dram_addr
    self.mem_addr = xs1 & 0xFFFFFFFF;

    // Parse xs2 fields (matching C implementation):
    // rs2[14:0]  = base_sp_addr (15 bits)
    // rs2[24:15] = iter (10 bits)
    // rs2[33:24] = col_stride/stride (10 bits)
    self.sp_addr = (xs2 & 0x7FFF) as u32;           // bits [14:0]
    self.iter = ((xs2 >> 15) & 0x3FF) as u32;       // bits [24:15]
    self.stride = ((xs2 >> 24) & 0x3FF) as u32;     // bits [33:24]

    // Parse sp_addr into bank and bank_addr
    // Assuming: 12 banks total, 4096 entries per bank
    // sp_bank_addr_bits = log2(4096) = 12
    // sp_bank_bits = log2(12) = 4 (rounded up)
    const SP_BANK_ADDR_BITS: u32 = 12;
    self.sp_bank_addr = self.sp_addr & ((1 << SP_BANK_ADDR_BITS) - 1);  // bits [11:0]
    self.sp_bank = self.sp_addr >> SP_BANK_ADDR_BITS;                    // bits [14:12]

    // funct=24: mvin, funct=25: mvout
    match funct {
      24 => log_backward!(
        "MemDomain decode: mvin (mem_addr=0x{:x}, sp_addr=0x{:x} [bank={}, addr=0x{:x}], iter={}, col_stride={})",
        self.mem_addr, self.sp_addr, self.sp_bank, self.sp_bank_addr, self.iter, self.stride
      ),
      25 => log_backward!(
        "MemDomain decode: mvout (mem_addr=0x{:x}, sp_addr=0x{:x} [bank={}, addr=0x{:x}], iter={}, stride={})",
        self.mem_addr, self.sp_addr, self.sp_bank, self.sp_bank_addr, self.iter, self.stride
      ),
      _ => log_backward!("MemDomain decode: unknown funct={}", funct),
    }
  }

  pub fn print_status(&self) {
    println!("    [Decoder] funct={}, mem_addr=0x{:x}, sp_addr=0x{:x} [bank={}, addr=0x{:x}], iter={}, stride={}, is_load={}, is_store={}",
             self.funct, self.mem_addr, self.sp_addr, self.sp_bank, self.sp_bank_addr, self.iter, self.stride, self.is_load, self.is_store);
  }
}
