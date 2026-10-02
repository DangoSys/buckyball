#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_u8_random_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = rand() % 128;
  }
}

void init_u32_random_matrix(result_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = rand() % 256;
  }
}
int compare_u8_matrices(elem_t *a, elem_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}
int compare_u32_matrices(result_t *a, result_t *b, int rows, int cols) {
  for (int i = 0; i <= rows * cols - 1; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}

int compare_i8_matrices(elem_t *a, elem_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}
int compare_i32_matrices(result_t *a, result_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}

void clear_u32_matrix(result_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(result_t));
}
void clear_u8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}
void clear_i8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}

// Transpose matrix
void transpose_u8_matrix(elem_t *src, elem_t *dst, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}

result_t gemmini_in_shift(result_t v, int shift) {
  if (shift <= 0)
    return v;
  if (shift >= 32)
    return 0;

  uint32_t s = (uint32_t)shift;
  uint32_t bits = (uint32_t)v;
  uint32_t point_five = (bits >> (s - 1)) & 1;
  uint32_t zeros = (s <= 1) ? 0 : ((bits & ((1u << (s - 1)) - 1)) != 0);
  uint32_t ones_digit = (bits >> s) & 1;
  uint32_t r = point_five & (zeros | ones_digit);
  return (v >> s) + (result_t)r;
}

// CPU matrix multiplication (used to generate expected results)
void cpu_matmul(elem_t *a, elem_t *b, result_t *c, int rows, int cols,
                int inner) {
  clear_u32_matrix(c, rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = 0; k < inner; k++) {
        c[i * cols + j] += a[i * inner + k] * b[k * cols + j];
      }
    }
  }
}
// MMIO stubs are for baremetal/BBSim only.
// Linux user-mode tests (`*-linux` under BEMU user-mode execution) must use
// libc/syscall exit path.
#if !defined(__linux__)
// MMIO address map (BBSimHarness, WithDefaultMMIOPort base=0x6000_0000):
//   0x6000_0000 : simulation exit  — write triggers sim_exit()
//   0x6002_0000 : UART0 TX         — write low byte → putchar in C++
#define MMIO_SIM_EXIT ((volatile uint32_t *)0x60000000UL)
#define MMIO_UART_TX ((volatile uint32_t *)0x60020000UL)

// _write: route stdout/stderr through MMIO UART so printf works in simulation.
// nosys.specs provides a weak _write stub; we override it here.
int _write(int fd, const char *buf, int len) {
  (void)fd;
  for (int i = 0; i < len; i++) {
    *MMIO_UART_TX = (uint32_t)(unsigned char)buf[i];
  }
  return len;
}

// _exit: write exit code to MMIO sim-exit register; C++ mmio_tick() detects
// this and calls sim_exit().
void __attribute__((noreturn)) _exit(int code) {
  *MMIO_SIM_EXIT = (uint32_t)code;
  while (1) {
  } // wait for C++ to process the MMIO write and call sim_exit()
}
#endif
