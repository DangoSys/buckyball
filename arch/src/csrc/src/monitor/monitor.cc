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
const char *log_path    = nullptr;
const char *fst_path    = nullptr;
const char *stdout_path = nullptr;

// Raw stdout fd saved before dup2 redirect — used by UART putchar for real-time display.
int raw_stdout_fd = -1;

static int parse_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "+fst=", 5) == 0) {
      fst_path = argv[i] + 5;
    } else if (strncmp(argv[i], "+log=", 5) == 0) {
      log_path = argv[i] + 5;
    } else if (strncmp(argv[i], "+stdout=", 8) == 0) {
      stdout_path = argv[i] + 8;
    } else if (strcmp(argv[i], "+batch") == 0) {
      bdb_set_batch_mode();
    } else if (strcmp(argv[i], "+help") == 0) {
      printf("\t+batch            run with batch mode\n");
      printf("\t+elf=<path>       specify ELF binary to load into DRAM\n");
      printf("\t+log=<path>       specify log file path\n");
      printf("\t+stdout=<path>    specify UART output file path\n");
      printf("\t+fst=<path>       specify FST waveform file path\n");
      printf("\n");
      exit(0);
    }
    // +elf= is parsed by SimDRAM_bb.cc via vpi_get_vlog_info (Verilator plusargs)
  }

  Assert(log_path, "Log file path is required. Use +log=<path> to specify.");
  Assert(fst_path, "FST file path is required. Use +fst=<path> to specify.");
  return 0;
}

static void init_log(const char *log_file) {
  if (log_file != NULL) {
    // Save original stdout fd for UART real-time display
    raw_stdout_fd = dup(STDOUT_FILENO);
    FILE *fp = fopen(log_file, "w");
    Assert(fp, "Can not open '%s'", log_file);
    // Redirect stdout to bdb.log for Log() / DPI-C printf output
    fflush(stdout);
    dup2(fileno(fp), STDOUT_FILENO);
    fclose(fp);
  }
  Log("Log is written to %s", log_file ? log_file : "stdout");
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
