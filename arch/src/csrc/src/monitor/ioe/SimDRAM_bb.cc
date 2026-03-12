// SimDRAM_bb.cc — Override memory_init from testchipip's SimDRAM.cc
// Replaces fesvr load_elf with libelf-based ELF loader.
// This file is compiled into the simulation and takes precedence over
// the version embedded in the SimDRAM Verilog blackbox resource.
//
// Globals shared with testchipip/csrc/mm_dramsim2.cc (included via linkage):
//   extern bool use_dramsim;
//   extern std::string loadmem_file;
//   extern std::vector<std::map<long long int, backing_data_t>> backing_mem_data;

#include <vpi_user.h>
#include <svdpi.h>
#include <stdint.h>
#include <cassert>
#include <sys/mman.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <vector>

// elf.h is available in glibc, no extra library needed for parsing
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "mm_dramsim2.h"

// Global state — defined here since testchipip's SimDRAM.cc is excluded from build.
// memory_tick (not overridden) accesses these via external linkage from mm_dramsim2.cc.
bool use_dramsim = false;
std::string ini_dir = "dramsim2_ini";
std::vector<std::map<long long int, backing_data_t>> backing_mem_data = {};

// ELF file path, set via +elf= plusarg
static std::string elf_file = "";

// ---------------------------------------------------------------------------
// load_elf_to_mem: parse ELF64 and copy PT_LOAD segments into backing data
// ---------------------------------------------------------------------------
static void load_elf_to_mem(const char *path, uint8_t *data,
                             uint64_t mem_base, uint64_t mem_size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "[SimDRAM_bb] Cannot open ELF: %s\n", path);
    abort();
  }

  struct stat st;
  fstat(fd, &st);
  size_t file_size = st.st_size;

  uint8_t *file_buf = (uint8_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (file_buf == MAP_FAILED) {
    fprintf(stderr, "[SimDRAM_bb] mmap failed for ELF: %s\n", path);
    abort();
  }

  Elf64_Ehdr *ehdr = (Elf64_Ehdr *)file_buf;
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    fprintf(stderr, "[SimDRAM_bb] Not a valid ELF file: %s\n", path);
    abort();
  }
  if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
    fprintf(stderr, "[SimDRAM_bb] Only ELF64 supported\n");
    abort();
  }

  Elf64_Phdr *phdrs = (Elf64_Phdr *)(file_buf + ehdr->e_phoff);
  size_t loaded = 0;
  for (int i = 0; i < ehdr->e_phnum; i++) {
    Elf64_Phdr *ph = &phdrs[i];
    if (ph->p_type != PT_LOAD) continue;
    if (ph->p_filesz == 0) continue;

    uint64_t vaddr = ph->p_paddr; // use physical address
    if (vaddr < mem_base || vaddr + ph->p_memsz > mem_base + mem_size) {
      fprintf(stderr,
        "[SimDRAM_bb] Segment paddr=0x%lx size=0x%lx outside mem [0x%lx, 0x%lx)\n",
        vaddr, ph->p_memsz, mem_base, mem_base + mem_size);
      abort();
    }
    uint64_t offset = vaddr - mem_base;
    // Copy file content
    memcpy(data + offset, file_buf + ph->p_offset, ph->p_filesz);
    // Zero BSS (memsz > filesz)
    if (ph->p_memsz > ph->p_filesz) {
      memset(data + offset + ph->p_filesz, 0, ph->p_memsz - ph->p_filesz);
    }
    loaded += ph->p_filesz;
  }

  munmap(file_buf, file_size);
  printf("[SimDRAM_bb] Loaded ELF '%s': %zu bytes\n", path, loaded);
}

// ---------------------------------------------------------------------------
// memory_init override — called from SimDRAM.v via DPI-C
// ---------------------------------------------------------------------------
extern "C" void *memory_init(
    int chip_id,
    long long int mem_size,
    long long int word_size,
    long long int line_size,
    long long int id_bits,
    long long int clock_hz,
    long long int mem_base)
{
  mm_t *mm;
  s_vpi_vlog_info info;

  std::string memory_ini = "DDR3_micron_64M_8B_x4_sg15.ini";
  std::string system_ini = "system.ini";
  std::string local_ini_dir = "dramsim2_ini";

  if (!vpi_get_vlog_info(&info))
    abort();

  // Parse plusargs: +elf=<path>, +dramsim, +dramsim_ini_dir=<dir>
  for (int i = 1; i < info.argc; i++) {
    std::string arg(info.argv[i]);
    if (arg.find("+elf=") == 0)
      elf_file = arg.substr(strlen("+elf="));
    if (arg == "+dramsim")
      use_dramsim = true;
    if (arg.find("+dramsim_ini_dir=") == 0)
      local_ini_dir = arg.substr(strlen("+dramsim_ini_dir="));
  }

  while (chip_id >= (int)backing_mem_data.size()) {
    backing_mem_data.push_back(std::map<long long int, backing_data_t>());
  }

  if (backing_mem_data[chip_id].find(mem_base) != backing_mem_data[chip_id].end()) {
    assert(backing_mem_data[chip_id][mem_base].size == (size_t)mem_size);
  } else {
    uint8_t *data = (uint8_t *)mmap(NULL, mem_size,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
      fprintf(stderr, "[SimDRAM_bb] mmap for backing store failed\n");
      abort();
    }
    memset(data, 0, mem_size);

    // Load ELF if +elf= was provided
    if (!elf_file.empty()) {
      load_elf_to_mem(elf_file.c_str(), data, (uint64_t)mem_base, (uint64_t)mem_size);
    }

    backing_mem_data[chip_id][mem_base] = {data, (size_t)mem_size};
  }

  if (use_dramsim) {
    mm = (mm_t *)(new mm_dramsim2_t(
        mem_base, mem_size, word_size, line_size,
        backing_mem_data[chip_id][mem_base],
        memory_ini, system_ini, local_ini_dir,
        1 << id_bits, clock_hz));
  } else {
    mm = (mm_t *)(new mm_magic_t(
        mem_base, mem_size, word_size, line_size,
        backing_mem_data[chip_id][mem_base]));
  }

  return mm;
}

// ---------------------------------------------------------------------------
// MMIO device state for UART interception
// SiFive UART TX register: base 0x10020000, offset 0 = txdata
// AXI write is split: aw channel (address) and w channel (data) may arrive
// in different cycles. We track pending aw/w separately.
// ---------------------------------------------------------------------------

#define UART_TX_ADDR 0x10020000LL

struct mmio_state_t {
  bool     aw_pending = false;
  long long aw_addr   = 0;
  int      aw_id      = 0;
  bool     b_fire     = false;  // b response ready to send
  int      b_id_val   = 0;
} mmio_state;

static void mmio_uart_tick(
    unsigned char reset,
    unsigned char aw_valid, unsigned char *aw_ready,
    long long int aw_addr_in, int aw_id_in,
    unsigned char w_valid, unsigned char *w_ready,
    long long w_data,
    unsigned char *b_valid, unsigned char b_ready, int *b_id, int *b_resp)
{
  if (reset) {
    mmio_state = mmio_state_t{};
    *aw_ready = 1; *w_ready = 1; *b_valid = 0;
    return;
  }

  // Accept aw
  *aw_ready = !mmio_state.aw_pending;
  if (aw_valid && *aw_ready) {
    mmio_state.aw_pending = true;
    mmio_state.aw_addr    = aw_addr_in;
    mmio_state.aw_id      = aw_id_in;
  }

  // Accept w when aw is pending
  *w_ready = mmio_state.aw_pending && !mmio_state.b_fire;
  if (w_valid && *w_ready) {
    if (mmio_state.aw_addr == UART_TX_ADDR) {
      char ch = (char)(w_data & 0xFF);
      putchar(ch);
      fflush(stdout);
    }
    mmio_state.aw_pending = false;
    mmio_state.b_fire     = true;
    mmio_state.b_id_val   = mmio_state.aw_id;
  }

  // Send b response
  *b_valid = mmio_state.b_fire;
  *b_id    = mmio_state.b_id_val;
  *b_resp  = 0;
  if (mmio_state.b_fire && b_ready)
    mmio_state.b_fire = false;
}

// ---------------------------------------------------------------------------
// memory_tick — DPI-C from SimDRAM.v / SimAXIMem.v
// MMIO region (base != DRAM base) is dispatched to mmio_uart_tick.
// ---------------------------------------------------------------------------
extern "C" void memory_tick(
    void *channel,
    unsigned char reset,
    unsigned char ar_valid, unsigned char *ar_ready,
    long long int ar_addr, int ar_id, int ar_size, int ar_len,
    unsigned char aw_valid, unsigned char *aw_ready,
    long long int aw_addr, int aw_id, int aw_size, int aw_len,
    unsigned char w_valid, unsigned char *w_ready,
    int w_strb, long long w_data, unsigned char w_last,
    unsigned char *r_valid, unsigned char r_ready,
    int *r_id, int *r_resp, long long *r_data, unsigned char *r_last,
    unsigned char *b_valid, unsigned char b_ready,
    int *b_id, int *b_resp)
{
  mm_t *mm = (mm_t *)channel;

  // Route MMIO channel (mem_base < 0x80000000) to our MMIO handler.
  // DRAM is always at 0x80000000+; MMIO (SiFive peripherals) is below.
  if (mm->get_base() < 0x80000000ULL) {
    *ar_ready = 0; *r_valid = 0; *r_last = 0; *r_id = 0; *r_resp = 0; *r_data = 0;
    mmio_uart_tick(reset,
                   aw_valid, aw_ready, aw_addr, aw_id,
                   w_valid, w_ready, w_data,
                   b_valid, b_ready, b_id, b_resp);
    return;
  }

  mm->tick(reset,
           ar_valid, ar_addr, ar_id, ar_size, ar_len,
           aw_valid, aw_addr, aw_id, aw_size, aw_len,
           w_valid, w_strb, &w_data, w_last,
           r_ready, b_ready);
  *ar_ready = mm->ar_ready();
  *aw_ready = mm->aw_ready();
  *w_ready  = mm->w_ready();
  *r_valid  = mm->r_valid();
  *r_id     = mm->r_id();
  *r_resp   = mm->r_resp();
  *r_data   = *((long *)mm->r_data());
  *r_last   = mm->r_last();
  *b_valid  = mm->b_valid();
  *b_id     = mm->b_id();
  *b_resp   = mm->b_resp();
}
