#include "tet.h"
#include <getopt.h>
#include "utils/macro.h"
#include "utils/debug.h"


static void welcome() {
  	Log("Build time: %s, %s", __TIME__, __DATE__);
  	printf("Welcome to %s!\n", ASNI_FMT(str(TET), ASNI_FG_YELLOW ASNI_BG_RED));
  	printf("For help, type \"help\"\n");
}

static int parse_args(int argc, char *argv[]) {
	const struct option table[] = {
		{"batch"    , no_argument      , NULL, 'b'},
		{"help"     , no_argument      , NULL, 'h'},
	};
	int o;
	while ( (o = getopt_long(argc, argv, "b", table, NULL)) != -1) {
		switch (o) {
		case 'b': sdb_set_batch_mode(); break;
		default:
			printf("\t-b,--batch              run with batch mode\n");
			printf("\n");
			exit(0);
		}
	}
	return 0;
}


void init_monitor(int argc, char *argv[]) {
	parse_args(argc, argv);
	welcome();
}
