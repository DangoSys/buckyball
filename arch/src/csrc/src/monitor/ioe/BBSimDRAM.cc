// BBSim-owned DPI memory backend for BBSimDRAM.v.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdint.h>
#include <string>
#include <svdpi.h>
#include <sys/mman.h>
#include <vector>
#include <vpi_user.h>

#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ioe/mm_dramsim2.h"

static bool use_dramsim = false;
static std::vector<std::map<long long int, backing_data_t>> mem_data = {};
static std::string elf_file = "";

static void load_elf_to_mem(const char *path, uint8_t *data, uint64_t mem_base,
                            uint64_t mem_size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "[BBSimDRAM] Cannot open ELF: %s\n", path);
    abort();
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    fprintf(stderr, "[BBSimDRAM] fstat failed for ELF: %s\n", path);
    abort();
  }

  size_t file_size = st.st_size;
  uint8_t *file_buf =
      (uint8_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (file_buf == MAP_FAILED) {
    fprintf(stderr, "[BBSimDRAM] mmap failed for ELF: %s\n", path);
    abort();
  }

  Elf64_Ehdr *ehdr = (Elf64_Ehdr *)file_buf;
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    fprintf(stderr, "[BBSimDRAM] Not a valid ELF file: %s\n", path);
    abort();
  }
  if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
    fprintf(stderr, "[BBSimDRAM] Only ELF64 supported\n");
    abort();
  }

  Elf64_Phdr *phdrs = (Elf64_Phdr *)(file_buf + ehdr->e_phoff);
  size_t loaded = 0;
  for (int i = 0; i < ehdr->e_phnum; i++) {
    Elf64_Phdr *ph = &phdrs[i];
    if (ph->p_type != PT_LOAD)
      continue;
    if (ph->p_filesz == 0)
      continue;

    uint64_t vaddr = ph->p_paddr;
    if (vaddr < mem_base || vaddr + ph->p_memsz > mem_base + mem_size) {
      fprintf(stderr,
              "[BBSimDRAM] Segment paddr=0x%lx size=0x%lx outside mem [0x%lx, "
              "0x%lx)\n",
              vaddr, ph->p_memsz, mem_base, mem_base + mem_size);
      abort();
    }

    uint64_t offset = vaddr - mem_base;
    memcpy(data + offset, file_buf + ph->p_offset, ph->p_filesz);
    if (ph->p_memsz > ph->p_filesz)
      memset(data + offset + ph->p_filesz, 0, ph->p_memsz - ph->p_filesz);
    loaded += ph->p_filesz;
  }

  munmap(file_buf, file_size);
  printf("[BBSimDRAM] Loaded ELF '%s': %zu bytes\n", path, loaded);
}

extern "C" void *
bbsim_memory_init(int chip_id, long long int mem_size, long long int word_size,
                  long long int line_size, long long int id_bits,
                  long long int clock_hz, long long int mem_base) {
  mm_t *mm;
  s_vpi_vlog_info info;

  std::string memory_ini = "DDR3_micron_64M_8B_x4_sg15.ini";
  std::string system_ini = "system.ini";
  std::string local_ini_dir = "dramsim2_ini";

  if (!vpi_get_vlog_info(&info))
    abort();

  for (int i = 1; i < info.argc; i++) {
    std::string arg(info.argv[i]);
    if (arg.find("+elf=") == 0)
      elf_file = arg.substr(strlen("+elf="));
    if (arg == "+dramsim")
      use_dramsim = true;
    if (arg.find("+dramsim_ini_dir=") == 0)
      local_ini_dir = arg.substr(strlen("+dramsim_ini_dir="));
  }

  while (chip_id >= (int)mem_data.size())
    mem_data.push_back(std::map<long long int, backing_data_t>());

  if (mem_data[chip_id].find(mem_base) != mem_data[chip_id].end()) {
    assert(mem_data[chip_id][mem_base].size == (size_t)mem_size);
  } else {
    uint8_t *data = (uint8_t *)mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
      fprintf(stderr, "[BBSimDRAM] mmap for backing store failed\n");
      abort();
    }
    memset(data, 0, mem_size);

    if (!elf_file.empty())
      load_elf_to_mem(elf_file.c_str(), data, (uint64_t)mem_base,
                      (uint64_t)mem_size);

    mem_data[chip_id][mem_base] = {data, (size_t)mem_size};
  }

  if (use_dramsim) {
    mm = (mm_t *)(new mm_dramsim2_t(
        mem_base, mem_size, word_size, line_size, mem_data[chip_id][mem_base],
        memory_ini, system_ini, local_ini_dir, 1 << id_bits, clock_hz));
  } else {
    mm = (mm_t *)(new mm_magic_t(mem_base, mem_size, word_size, line_size,
                                 mem_data[chip_id][mem_base]));
  }

  return mm;
}

extern "C" void bbsim_memory_tick(
    void *channel, unsigned char reset, unsigned char ar_valid,
    unsigned char *ar_ready, long long int ar_addr, int ar_id, int ar_size,
    int ar_len, unsigned char aw_valid, unsigned char *aw_ready,
    long long int aw_addr, int aw_id, int aw_size, int aw_len,
    unsigned char w_valid, unsigned char *w_ready, int w_strb, long long w_data,
    unsigned char w_last, unsigned char *r_valid, unsigned char r_ready,
    int *r_id, int *r_resp, long long *r_data, unsigned char *r_last,
    unsigned char *b_valid, unsigned char b_ready, int *b_id, int *b_resp) {
  mm_t *mm = (mm_t *)channel;
  mm->tick(reset, ar_valid, ar_addr, ar_id, ar_size, ar_len, aw_valid, aw_addr,
           aw_id, aw_size, aw_len, w_valid, w_strb, &w_data, w_last, r_ready,
           b_ready);
  *ar_ready = mm->ar_ready();
  *aw_ready = mm->aw_ready();
  *w_ready = mm->w_ready();
  *r_valid = mm->r_valid();
  *r_id = mm->r_id();
  *r_resp = mm->r_resp();
  *r_data = *((long *)mm->r_data());
  *r_last = mm->r_last();
  *b_valid = mm->b_valid();
  *b_id = mm->b_id();
  *b_resp = mm->b_resp();
}
