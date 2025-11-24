#include "utils/debug.h"
#include <stdio.h>

static void print_logo() {
  const char *logo[] = {
      "     __  __                               __          ____  ",
      "    / / / /___ __   _____      ____ _    / /_  ____ _/ / / ",
      "   / /_/ / __ `/ | / / _ \\    / __ `/   / __ \\/ __ `/ / /   ",
      "  / __  / /_/ /| |/ /  __/   / /_/ /   / /_/ / /_/ / / /   ",
      " /_/ /_/\\__,_/ |___/\\___/    \\__,_/   /_.___/\\__,_/_/_/     "};
  int n = sizeof(logo) / sizeof(logo[0]);
  for (int i = 0; i < n; i++) {
    printf("%s\n", logo[i]);
  }
}

static void welcome() {
  Log("Build time: %s, %s", __TIME__, __DATE__);
  print_logo();
  printf("Welcome to %s!\n",
         ASNI_FMT(str(Buckyball), ASNI_FG_YELLOW ASNI_BG_RED));
  printf("For help, type \"help\"\n");
}
