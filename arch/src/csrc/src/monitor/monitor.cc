#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "utils/welcome.cc"

// Define global path variables
const char *log_path = nullptr;
const char *fst_path = nullptr;
const char *stdout_path = nullptr;
uint32_t bdb_trace_mask = BDB_TR_ALL;

// Raw stdout fd saved before dup2 redirect — used by UART putchar for real-time
// display.
int raw_stdout_fd = -1;

static int parse_args(int argc, char *argv[]) {
  auto parse_trace_list = [](const char *list) {
    if (list == nullptr || *list == '\0') {
      return;
    }
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "%s", list);
    Assert(n >= 0 && n < (int)sizeof(buf), "trace option too long: %s", list);
    char *tok = strtok(buf, ",");
    while (tok != NULL) {
      if (strcmp(tok, "all") == 0) {
        bdb_trace_mask = BDB_TR_ALL;
      } else if (strcmp(tok, "none") == 0) {
        bdb_trace_mask = 0;
      } else if (strcmp(tok, "itrace") == 0) {
        bdb_trace_mask |= BDB_TR_ITRACE;
      } else if (strcmp(tok, "mtrace") == 0) {
        bdb_trace_mask |= BDB_TR_MTRACE;
      } else if (strcmp(tok, "pmctrace") == 0) {
        bdb_trace_mask |= BDB_TR_PMCTRACE;
      } else if (strcmp(tok, "ctrace") == 0) {
        bdb_trace_mask |= BDB_TR_CTRACE;
      } else if (strcmp(tok, "banktrace") == 0) {
        bdb_trace_mask |= BDB_TR_BANKTRACE;
      } else {
        panic("Unknown +trace item: %s", tok);
      }
      tok = strtok(NULL, ",");
    }
  };

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "+fst=", 5) == 0) {
      fst_path = argv[i] + 5;
    } else if (strncmp(argv[i], "+log=", 5) == 0) {
      log_path = argv[i] + 5;
    } else if (strncmp(argv[i], "+stdout=", 8) == 0) {
      stdout_path = argv[i] + 8;
    } else if (strncmp(argv[i], "+trace_mask=", 12) == 0) {
      char *end = NULL;
      unsigned long v = strtoul(argv[i] + 12, &end, 0);
      Assert(end && *end == '\0', "Invalid +trace_mask value: %s",
             argv[i] + 12);
      bdb_trace_mask = (uint32_t)v;
    } else if (strncmp(argv[i], "+trace=", 7) == 0) {
      bdb_trace_mask = 0;
      parse_trace_list(argv[i] + 7);
    } else if (strcmp(argv[i], "+batch") == 0) {
      bdb_set_batch_mode();
    } else if (strcmp(argv[i], "+help") == 0) {
      printf("\t+batch            run with batch mode\n");
      printf("\t+elf=<path>       specify ELF binary to load into DRAM\n");
      printf("\t+log=<path>       specify log file path\n");
      printf("\t+stdout=<path>    specify UART output file path\n");
      printf("\t+fst=<path>       specify FST waveform file path\n");
      printf("\t+trace=<items>    trace list: "
             "none|all|itrace,mtrace,pmctrace,ctrace,banktrace\n");
      printf("\t+trace_mask=<n>   bitfield itrace=1 mtrace=2 pmctrace=4 "
             "ctrace=8 banktrace=16\n");
      printf("\n");
      exit(0);
    }
    // +elf= is parsed by SimDRAM_bb.cc via vpi_get_vlog_info (Verilator
    // plusargs)
  }

  Assert(log_path, "Log file path is required. Use +log=<path> to specify.");
  Assert(fst_path, "FST file path is required. Use +fst=<path> to specify.");
  return 0;
}

static void init_log(const char *log_file) {
  if (log_file != NULL) {
    // Keep a dedicated fd for UART real-time display.
    // Do not redirect stdout to trace file, otherwise NDJSON is polluted.
    raw_stdout_fd = dup(STDOUT_FILENO);
    Assert(raw_stdout_fd >= 0, "dup(STDOUT_FILENO) failed");
    FILE *fp = fopen(log_file, "w");
    Assert(fp, "Can not open '%s'", log_file);
    fclose(fp); // truncate/create file; trace writers append NDJSON later
  }
  if (log_file) {
    fprintf(stderr, "NDJSON trace is written to %s\n", log_file);
    fprintf(stderr,
            "Trace mask=0x%X [itrace=%d mtrace=%d pmctrace=%d ctrace=%d "
            "banktrace=%d]\n",
            bdb_trace_mask, bdb_trace_on(BDB_TR_ITRACE),
            bdb_trace_on(BDB_TR_MTRACE), bdb_trace_on(BDB_TR_PMCTRACE),
            bdb_trace_on(BDB_TR_CTRACE), bdb_trace_on(BDB_TR_BANKTRACE));
  } else {
    fprintf(stderr, "NDJSON trace path is not set\n");
  }
}

static void init_io() {
  fflush(stdout);
  fflush(stderr);

  struct termios tty;
  if (tcgetattr(STDIN_FILENO, &tty) == 0) {
    tty.c_lflag |= ECHO;
    tty.c_lflag |= ICANON;
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
  }
}

void init_monitor(int argc, char *argv[]) {
  parse_args(argc, argv);
  init_log(log_path);
  init_io();
  welcome();
}
