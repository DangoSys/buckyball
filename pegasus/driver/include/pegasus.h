#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// Pegasus FPGA Simulation Framework — Public API
// ============================================================

// ---- SCU register offsets (BAR0 MMIO) ----
#define SCU_CTRL_OFFSET     0x0000U  // W: [0]=run, [1]=enter step mode
#define SCU_STEP_N_OFFSET   0x0004U  // W: write N → execute N DUT cycles, auto-stop
#define SCU_STATUS_OFFSET   0x0008U  // R: [0]=idle, [1]=step_done
#define SCU_CYCLE_LO_OFFSET 0x000CU  // R: cycle counter bits [31:0]
#define SCU_CYCLE_HI_OFFSET 0x0010U  // R: cycle counter bits [63:32]
#define SCU_RESET_OFFSET    0x0014U  // W: [0]=assert DUT reset (1=reset, 0=run)

// ---- HBM2 address space (as seen by XDMA DMA engine) ----
#define HBM2_BASE           0x000000000ULL  // FPGA-internal HBM2 base address
#define HBM2_SIZE           0x100000000ULL  // 4 GB (single pseudo-channel)

// ---- SoC memory map (Rocket core physical address space) ----
#define SOC_DRAM_BASE       0x080000000ULL  // SoC DRAM starts here (physical)

// Address translation: SoC physical → HBM2 DMA address
#define SOC_TO_HBM2(paddr)  ((paddr) - SOC_DRAM_BASE + HBM2_BASE)

// ---- XDMA device file paths ----
// UART data comes from C2H channel 1 (channel 0 used for DMA)
#define XDMA_H2C_CHAN    "/dev/xdma0_h2c_0"
#define XDMA_C2H_CHAN    "/dev/xdma0_c2h_0"
#define XDMA_USER_BAR    "/dev/xdma0_user"
#define XDMA_UART_C2H    "/dev/xdma0_c2h_1"

// ============================================================
// XDMA device handle
// ============================================================
struct PegasusXDMA {
    int      h2c_fd;      // H2C DMA file descriptor (write to FPGA)
    int      c2h_fd;      // C2H DMA file descriptor (read from FPGA)
    int      uart_fd;     // C2H channel 1 (UART stream)
    int      bar_fd;      // BAR0 mmap file descriptor
    void    *bar0_base;   // mmap'd BAR0 base pointer
    uint32_t bar0_size;   // BAR0 size in bytes
};

// ============================================================
// ELF information extracted from ELF file
// ============================================================
struct ELFSegment {
    uint64_t paddr;   // SoC physical load address
    uint64_t vaddr;   // Virtual address (unused)
    uint64_t filesz;  // Bytes to copy from file
    uint64_t memsz;   // Bytes in memory (filesz ≤ memsz, zero-fill rest)
    uint64_t offset;  // Offset in ELF file
};

struct ELFInfo {
    std::string            path;
    uint64_t               entry;         // Entry point (SoC physical)
    uint64_t               tohost_paddr;  // SoC physical address of tohost symbol
    uint64_t               fromhost_paddr; // SoC physical address of fromhost symbol
    std::vector<ELFSegment> segments;     // PT_LOAD segments
};

// ============================================================
// XDMA device open/close
// ============================================================

// Open XDMA device.
// dev_prefix: e.g. "/dev/xdma0" — appended with "_h2c_0", "_c2h_0", etc.
// Returns nullptr on failure.
PegasusXDMA *xdma_open(const char *dev_prefix = "/dev/xdma0");
void xdma_close(PegasusXDMA *xdma);

// ============================================================
// BAR0 MMIO (32-bit registers)
// ============================================================
void     mmio_write32(PegasusXDMA *xdma, uint64_t offset, uint32_t val);
uint32_t mmio_read32(PegasusXDMA *xdma, uint64_t offset);

// ============================================================
// DMA read/write (pread/pwrite on /dev/xdma0_h2c_0 and c2h_0)
// ============================================================

// Write buf[0..len) to FPGA HBM2 at fpga_addr.
// fpga_addr is the DMA address (e.g., 0x0 for start of HBM2).
ssize_t dma_write(PegasusXDMA *xdma, uint64_t fpga_addr, const void *buf, size_t len);

// Read len bytes from FPGA HBM2 at fpga_addr into buf.
ssize_t dma_read(PegasusXDMA *xdma, uint64_t fpga_addr, void *buf, size_t len);

// ============================================================
// SCU control
// ============================================================

// Assert DUT reset for ~10ms, then deassert.
void scu_reset(PegasusXDMA *xdma);

// Start free-running DUT clock (CTRL[0]=1, CTRL[1]=0).
void scu_run(PegasusXDMA *xdma);

// Halt DUT clock (CTRL[0]=0).
void scu_halt(PegasusXDMA *xdma);

// Execute exactly n_cycles DUT clock cycles, then stop.
// Blocks until STATUS[1] (step_done) is set.
void scu_step(PegasusXDMA *xdma, uint32_t n_cycles);

// Read current DUT cycle counter (64-bit).
uint64_t scu_read_cycles(PegasusXDMA *xdma);

// ============================================================
// UART polling
// ============================================================

// Non-blocking read from UART FIFO (XDMA C2H channel 1).
// Writes received bytes to stdout. Call repeatedly in a loop.
void uart_poll(PegasusXDMA *xdma);

// ============================================================
// ELF loading
// ============================================================

// Parse ELF file: extract LOAD segments and tohost/fromhost symbol addresses.
// Exits on fatal error (file not found, not a valid RISC-V ELF, etc.).
ELFInfo elf_parse(const char *elf_path);

// Load all PT_LOAD segments from the parsed ELF into HBM2 via DMA.
// Address translation: SoC physical paddr → HBM2 DMA addr via SOC_TO_HBM2().
void elf_load(PegasusXDMA *xdma, const ELFInfo *info);
