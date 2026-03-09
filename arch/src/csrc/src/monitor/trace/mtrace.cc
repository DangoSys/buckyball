#include "monitor/trace.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>

// Global log file pointer (shared with monitor.cc)
extern const char *log_path;
static FILE *mtrace_fp = NULL;

// Initialize mtrace logging
static void init_mtrace()
{
  if (mtrace_fp == NULL && log_path != NULL)
  {
    mtrace_fp = fopen(log_path, "a");
    if (mtrace_fp == NULL)
    {
      panic("Failed to open mtrace log file: %s", log_path);
    }
  }
}

// DPI-C function for memory trace (mtrace)
// Called when MemBackend performs read/write operations
extern "C" void dpi_mtrace(unsigned char is_write, // 1 = write, 0 = read
                           unsigned char is_shared,
                           unsigned int channel, unsigned long long hart_id,
                           unsigned int vbank_id,
                           unsigned int group_id, unsigned int addr,
                           unsigned long long data_lo,
                           unsigned long long data_hi)
{
  init_mtrace();

  if (mtrace_fp)
  {
    if (is_write)
    {
      fprintf(mtrace_fp,
              "[MTRACE] WRITE ch=%u hart=%llu shared=%u vbank=%u group=%u "
              "addr=0x%08x "
              "data=0x%016llx%016llx\n",
              channel, hart_id, is_shared, vbank_id, group_id, addr, data_hi,
              data_lo);
    }
    else
    {
      fprintf(mtrace_fp,
              "[MTRACE] READ  ch=%u hart=%llu shared=%u vbank=%u group=%u "
              "addr=0x%08x\n",
              channel, hart_id, is_shared, vbank_id, group_id, addr);
    }
    fflush(mtrace_fp);
  }
}
