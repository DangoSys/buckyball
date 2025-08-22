#include "bdb.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/macro.h"
#include "utils/debug.h"

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


void init_monitor(int argc, char *argv[]) {
  parse_args(argc, argv);
  init_log(TOSTRING(CONFIG_LOG_PATH));
  // long img_size = load_image("/home/mio/Code/buckyball/arch/hello-baremetal");
  welcome();
}
