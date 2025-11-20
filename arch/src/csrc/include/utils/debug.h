#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "utils/macro.h"
#include <stdio.h>

// Without formatting
// #define COLOR(a, b) "\033[" #b "m" a "\033[0m"
// #define GREEN(a) COLOR(a, 32)
// #define RED(a) COLOR(a, 31)
// #define BLUE(a) COLOR(a, 34)

// Printf parameters for convenience, default is FG
#define BLACK "\33[1;30m"
#define RED "\33[1;31m"
#define GREEN "\33[1;32m"
#define YELLOW "\33[1;33m"
#define BLUE "\33[1;34m"
#define MAGENTA "\33[1;35m"
#define CYAN "\33[1;36m"
#define WHITE "\33[1;37m"

// Interface with original Debug
// Foreground color
#define ASNI_FG_BLACK "\33[1;30m"
#define ASNI_FG_RED "\33[1;31m"
#define ASNI_FG_GREEN "\33[1;32m"
#define ASNI_FG_YELLOW "\33[1;33m"
#define ASNI_FG_BLUE "\33[1;34m"
#define ASNI_FG_MAGENTA "\33[1;35m"
#define ASNI_FG_CYAN "\33[1;36m"
#define ASNI_FG_WHITE "\33[1;37m"
#define ASNI_BG_BLACK "\33[1;40m"
#define ASNI_BG_RED "\33[1;41m"
#define ASNI_BG_GREEN "\33[1;42m"
#define ASNI_BG_YELLOW "\33[1;43m"
#define ASNI_BG_BLUE "\33[1;44m"
#define ASNI_BG_MAGENTA "\33[1;35m"
#define ASNI_BG_CYAN "\33[1;46m"
#define ASNI_BG_WHITE "\33[1;47m"
// Reset color to "\033[0m"
#define ASNI_NONE "\33[0m"

#define ASNI_FMT(str, fmt) fmt str ASNI_NONE

#define log_write(...)                                                         \
  IFDEF(                                                                       \
      CONFIG_TARGET_NATIVE_ELF, do {                                           \
        extern FILE *log_fp;                                                   \
        extern bool log_enable();                                              \
        if (log_enable()) {                                                    \
          fprintf(log_fp, __VA_ARGS__);                                        \
          fflush(log_fp);                                                      \
        }                                                                      \
      } while (0))

#define _Log(...)                                                              \
  do {                                                                         \
    printf(__VA_ARGS__);                                                       \
    log_write(__VA_ARGS__);                                                    \
  } while (0)

#define Log(format, ...)                                                       \
  _Log(ASNI_FMT("[%s:%d %s] " format, ASNI_FG_BLUE) "\n", __FILE__, __LINE__,  \
       __func__, ##__VA_ARGS__)

// Printf usage: first parameter is color, subsequent parameters (if any) follow
// in order
#define Printf(format, color, ...) _Log(ASNI_FMT(format, color), ##__VA_ARGS__)

// #define assert(cond) \
//   do { \
//     if (!(cond)) { \
//       printf("Assertion fail at %s:%d\n", __FILE__, __LINE__); \
//       halt(0); \
//     } \
//   } while (0)

#define Assert(cond, format, ...)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fflush(stdout),                                                          \
          fprintf(stderr, ASNI_FMT(format, ASNI_FG_RED) "\n", ##__VA_ARGS__);  \
    }                                                                          \
  } while (0)

// #define Assert(cond, format, ...) \
//   do { \
//     if (!(cond)) { \
//       printf(ASNI_FMT(format, ASNI_FG_RED) "\n", ## __VA_ARGS__), \
//       fflush(stdout), fprintf(stderr, ASNI_FMT(format, ASNI_FG_RED) "\n", ##  __VA_ARGS__); \
//       extern FILE* log_fp; fflush(log_fp); \
//       extern void assert_fail_msg(); \
//       assert_fail_msg(); \
//       assert(cond); \
//     } \
//   } while (0)

#define panic(format, ...) Assert(0, format, ##__VA_ARGS__)

#endif
