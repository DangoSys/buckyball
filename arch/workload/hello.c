#include <stdio.h>

// 启动入口
// __attribute__((section(".text.start")))
// void _start() {
//     asm volatile("li a0, 0\n"
//                  "li a7, 93\n"
//                  "ecall\n");
// }

int main() {
  printf("Hello, World!\n");
  return 0;
}