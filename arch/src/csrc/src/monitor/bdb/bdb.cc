#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"
#include <readline/history.h>
#include <readline/readline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int bb_step;

static int is_batch_mode = false;

static char *rl_gets() {
  static char *line_read = NULL;
  if (line_read) {
    free(line_read);
    line_read = NULL;
  }
  line_read = readline("(bdb) ");

  if (line_read && *line_read) {
    add_history(line_read);
  }

  return line_read;
}

static int cmd_help(char *args);
static int cmd_c(char *args);
static int cmd_q(char *args);
static int cmd_si(char *args);

static struct {
  const char *name;
  const char *description;
  int (*handler)(char *);
} cmd_table[] = {
    {"help", "Display informations about all supported commands", cmd_help},
    {"c", "Continue the execution of the program", cmd_c},
    {"q", "Exit Simulation", cmd_q},
    {"si", "Execute the program in n steps\n\t\t-n nsteps(1~1000000)", cmd_si},
    // {"x", "scan the rom", cmd_x }
};

static int bdb_exec_once(int step) {
  while (step--) {
    ball_exec_once();
    bb_step++;
  }
  return 0;
}

#define NR_CMD ARRLEN(cmd_table)

static int cmd_help(char *args) {
  /* extract the first argument */
  char *arg = strtok(NULL, " ");
  int i;

  if (arg == NULL) {
    /* no argument given */
    printf("Common options:\n");
    for (i = 0; i < NR_CMD; i++) {
      printf("\t%-4s - %s\n", cmd_table[i].name, cmd_table[i].description);
    }
  } else {
    for (i = 0; i < NR_CMD; i++) {
      if (strcmp(arg, cmd_table[i].name) == 0) {
        printf("%s - %s\n", cmd_table[i].name, cmd_table[i].description);
        return 0;
      }
    }
    printf("Unknown command '%s'\n", arg);
  }
  return 0;
}

static int cmd_c(char *args) {
  while (1) {
    bdb_exec_once(1);
  }
  return 0;
}

static int cmd_q(char *args) { return -1; }

static int cmd_si(char *args) {
  if (args == NULL) {
    bdb_exec_once(1);
    return 0;
  }
  int step = atoi(strtok(NULL, " "));
  if (strtok(NULL, " ") != NULL) {
    printf("Too Many Parameters.\n");
    return 0;
  }
  if (step <= 0 || step > 1000000) {
    printf("Parameter Out of Range.\n");
    return 0;
  }
  bdb_exec_once(step);
  return 0;
}

void bdb_set_batch_mode() { is_batch_mode = true; }

void bdb_mainloop() {
  if (is_batch_mode) {
    cmd_c(NULL);
    return;
  }

  for (char *str; (str = rl_gets()) != NULL;) { // rl_gets读取（bdb）开始命令行
    char *str_end = str + strlen(str);

    char *cmd = strtok(str, " "); // strtok: 分解字符串为一组字符串
    if (cmd == NULL) {
      continue;
    }

    char *args = cmd + strlen(cmd) + 1;
    if (args >= str_end) {
      args = NULL;
    }

    int i;
    for (i = 0; i < NR_CMD; i++) {
      if (strcmp(cmd, cmd_table[i].name) == 0) {
        if (cmd_table[i].handler(args) < 0) {
          return;
        }
        break;
      }
    }
    if (i == NR_CMD) {
      printf("Unknown command '%s'\n", cmd);
    }
  }
}
