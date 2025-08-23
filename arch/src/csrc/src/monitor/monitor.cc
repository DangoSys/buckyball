#include "bdb.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/macro.h"
#include "utils/debug.h"

#include <termios.h>
#include <unistd.h>

#include "utils/welcome.cc"
#include "ioe/mem.cc"

static int parse_args(int argc, char *argv[]) {
  const struct option table[] = {
    {"batch"  , no_argument    , NULL, 'b'},
    {"help"   , no_argument    , NULL, 'h'},
  };
  int o;
  while ( (o = getopt_long(argc, argv, "b", table, NULL)) != -1) {
    switch (o) {
    case 'b': bdb_set_batch_mode(); break;
    default:
      printf("\t-b,--batch        run with batch mode\n");
      printf("\n");
      exit(0);
    }
  }
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
    tty.c_lflag |= ECHO;      // 启用回显
    tty.c_lflag |= ICANON;    // 启用行缓冲模式
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    Log("Terminal echo restored");
  }
  
}

void init_monitor(int argc, char *argv[]) {
  parse_args(argc, argv);
  init_log(TOSTRING(CONFIG_LOG_PATH));
  init_io();
  // long img_size = load_image("/home/mio/Code/buckyball/arch/hello-baremetal");
  welcome();
}
