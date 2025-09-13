#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// 颜色定义
#define ANSI_FG_BLACK "\33[1;30m"
#define ANSI_FG_RED "\33[1;31m"
#define ANSI_FG_GREEN "\33[1;32m"
#define ANSI_FG_YELLOW "\33[1;33m"
#define ANSI_FG_BLUE "\33[1;34m"
#define ANSI_FG_MAGENTA "\33[1;35m"
#define ANSI_FG_CYAN "\33[1;36m"
#define ANSI_FG_WHITE "\33[1;37m"
#define ANSI_NONE "\33[0m"

// 日志宏
#define Log(format, ...)                                                       \
  printf(ANSI_FG_BLUE "[LOG] " ANSI_NONE format "\n", ##__VA_ARGS__)

#define Info(format, ...)                                                      \
  printf(ANSI_FG_GREEN "[INFO] " ANSI_NONE format "\n", ##__VA_ARGS__)

#define Warn(format, ...)                                                      \
  printf(ANSI_FG_YELLOW "[WARN] " ANSI_NONE format "\n", ##__VA_ARGS__)

#define Error(format, ...)                                                     \
  printf(ANSI_FG_RED "[ERROR] " ANSI_NONE format "\n", ##__VA_ARGS__)

// 断言宏
#define Assert(cond, format, ...)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      Error("断言失败: %s:%d %s", __FILE__, __LINE__, #cond);                  \
      Error(format, ##__VA_ARGS__);                                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// 调试宏
#ifdef DEBUG
#define Debug(format, ...)                                                     \
  printf(ANSI_FG_MAGENTA "[DEBUG] " ANSI_NONE format "\n", ##__VA_ARGS__)
#else
#define Debug(format, ...)
#endif

#endif
