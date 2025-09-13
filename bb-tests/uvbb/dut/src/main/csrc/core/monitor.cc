#include "core/core.h"
#include "utils/debug.h"
#include <getopt.h>

// 全局配置变量
const char *vcd_path = nullptr;
const char *log_path = nullptr;

static int parse_args(int argc, char *argv[]) {
  const struct option table[] = {
      {"vcd", required_argument, NULL, 'v'},
      {"log", required_argument, NULL, 'l'},
      {"help", no_argument, NULL, 'h'},
  };

  int o;
  while ((o = getopt_long(argc, argv, "v:l:h", table, NULL)) != -1) {
    switch (o) {
    case 'v':
      vcd_path = optarg;
      break;
    case 'l':
      log_path = optarg;
      break;
    default:
      printf("用法: %s [选项]\n", argv[0]);
      printf("  -v, --vcd <path>   VCD文件路径\n");
      printf("  -l, --log <path>   日志文件路径\n");
      return -1;
    }
  }

  // 暂时简化参数检查
  if (!vcd_path)
    vcd_path = "default.vcd";
  if (!log_path)
    log_path = "default.log";
  return 0;
}

static void init_log(const char *log_file) {
  FILE *log_fp = nullptr;
  if (log_file) {
    FILE *fp = fopen(log_file, "w");
    Assert(fp, "无法打开日志文件: '%s'", log_file);
    log_fp = fp;
  } else {
    log_fp = stdout;
  }
  Log("日志写入至: %s", log_file ? log_file : "stdout");
}

extern "C" void init_monitor(int argc, char *argv[]) {
  if (parse_args(argc, argv) != 0) {
    exit(1);
  }
  init_log(log_path);
}
