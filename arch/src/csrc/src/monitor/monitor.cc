#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <termios.h>
#include <unistd.h>

#include "ioe/mem.cc"
#include "utils/welcome.cc"

// 定义全局VCD路径变量
const char *vcd_path = nullptr;

// 定义全局日志路径变量
const char *log_path = nullptr;
const char *fst_path = nullptr;

static int parse_args(int argc, char *argv[]) {
  // const struct option table[] = {
  //   {"batch"  , no_argument      , NULL, 'b'},
  //   {"vcd"    , required_argument, NULL, 'v'},
  //   {"log"    , required_argument, NULL, 'l'},
  //   {"help"   , no_argument      , NULL, 'h'},
  // };
  // int o;
  // while ( (o = getopt_long(argc, argv, "bv:l:", table, NULL)) != -1) {
  //   switch (o) {
  //   case 'b': bdb_set_batch_mode(); break;
  //   case 'v': vcd_path = optarg; break;
  //   case 'l': log_path = optarg; break;
  //   default:
  //     printf("\t-b,--batch        run with batch mode\n");
  //     printf("\t-v,--vcd <path>   specify VCD file path\n");
  //     printf("\t-l,--log <path>   specify log file path\n");

  // 手动解析Verilator风格的参数 (+vcd=path, +log=path)
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "+vcd=", 5) == 0) {
      vcd_path = argv[i] + 5; // 跳过 "+vcd="
    } else if (strncmp(argv[i], "+fst=", 5) == 0) {
      fst_path = argv[i] + 5; // 跳过 "+fst="
    } else if (strncmp(argv[i], "+log=", 5) == 0) {
      log_path = argv[i] + 5; // 跳过 "+log="
    } else if (strcmp(argv[i], "+batch") == 0) {
      bdb_set_batch_mode();
    } else if (strcmp(argv[i], "+help") == 0) {
      printf("\t+batch            run with batch mode\n");
      printf("\t+vcd=<path>       specify VCD file path\n");
      printf("\t+log=<path>       specify log file path\n");
      printf("\t+fst=<path>       specify FST file path\n");
      printf("\n");
      exit(0);
    }
  }

  // Assert(vcd_path, "VCD file path is required. Use +vcd=<path> to specify.");
  Assert(log_path, "Log file path is required. Use +log=<path> to specify.");
  Assert(fst_path, "FST file path is required. Use +fst=<path> to specify.");
  return 0;
}

static void init_log(const char *log_file) {
  FILE *log_fp = NULL;
  log_fp = stdout;
  if (log_file != NULL) {
    FILE *fp = fopen(log_file, "w");
    Assert(fp, "Can not open '%s'", log_file);
    log_fp = fp;
  }
  Log("Log is written to %s", log_file ? log_file : "stdout");
}

static void init_io() {
  // 强制刷新所有输出缓冲
  fflush(stdout);
  fflush(stderr);

  // 恢复终端回显功能
  struct termios tty;
  if (tcgetattr(STDIN_FILENO, &tty) == 0) {
    tty.c_lflag |= ECHO;   // 启用回显
    tty.c_lflag |= ICANON; // 启用行缓冲模式
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    Log("Terminal echo restored");
  }
}

void init_monitor(int argc, char *argv[]) {
  parse_args(argc, argv);
  init_log(log_path);
  init_io();
  // long img_size =
  // load_image("/home/mio/Code/buckyball/arch/hello-baremetal");
  welcome();
}
