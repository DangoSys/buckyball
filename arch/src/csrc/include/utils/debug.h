#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "utils/macro.h"
#include <stdio.h>

#define ASNI_FG_RED "\33[1;31m"
#define ASNI_FG_GREEN "\33[1;32m"
#define ASNI_FG_YELLOW "\33[1;33m"
#define ASNI_FG_BLUE "\33[1;34m"
#define ASNI_BG_RED "\33[1;41m"
#define ASNI_NONE "\33[0m"

#define ASNI_FMT(str, fmt) fmt str ASNI_NONE

#define Log(format, ...)                                                       \
  printf(ASNI_FMT("[%s:%d %s] " format, ASNI_FG_BLUE) "\n", __FILE__,          \
         __LINE__, __func__, ##__VA_ARGS__)

#define Assert(cond, format, ...)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fflush(stdout);                                                          \
      fprintf(stderr, ASNI_FMT(format, ASNI_FG_RED) "\n", ##__VA_ARGS__);      \
    }                                                                          \
  } while (0)

#define panic(format, ...) Assert(0, format, ##__VA_ARGS__)

#endif
