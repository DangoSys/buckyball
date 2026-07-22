#include "buddy_cycle_trace.h"

#define BUDDY_TRACE_MAX_ID 4096
#define BUDDY_TRACE_MAX_DEPTH 4
#define MMIO_UART_TX ((volatile uint32_t *)0x60020000UL)

static uint64_t start_cycles[BUDDY_TRACE_MAX_ID];

static void trace_fail(void) {
  __asm__ volatile("ebreak");
  while (1) {
  }
}

static uint64_t read_cycle(void) {
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static void uart_putc(char ch) { *MMIO_UART_TX = (uint32_t)(unsigned char)ch; }

static void uart_put_u64(uint64_t value) {
  char digits[20];
  unsigned int count = 0;

  if (value == 0) {
    uart_putc('0');
    return;
  }

  while (value != 0) {
    digits[count++] = (char)('0' + (value % 10));
    value /= 10;
  }
  while (count != 0) {
    uart_putc(digits[--count]);
  }
}

static void uart_put_i64(int64_t value) {
  if (value < 0) {
    uart_putc('-');
    uart_put_u64((uint64_t)(-(value + 1)) + 1);
    return;
  }
  uart_put_u64((uint64_t)value);
}

static void validate_id(int64_t id) {
  if (id < 0 || id >= BUDDY_TRACE_MAX_ID) {
    trace_fail();
  }
}

static void validate_path(int64_t depth,
                          const int64_t path[BUDDY_TRACE_MAX_DEPTH]) {
  if (depth <= 0 || depth > BUDDY_TRACE_MAX_DEPTH) {
    trace_fail();
  }
  for (int64_t i = 0; i < depth; ++i) {
    if (path[i] < 0) {
      trace_fail();
    }
  }
}

static void emit_cycle_record(int64_t depth,
                              const int64_t path[BUDDY_TRACE_MAX_DEPTH],
                              uint64_t start, uint64_t end) {
  static const char prefix[] = "@BCT,";
  for (unsigned int i = 0; i < sizeof(prefix) - 1; ++i) {
    uart_putc(prefix[i]);
  }

  uart_put_i64(depth);
  for (int i = 0; i < BUDDY_TRACE_MAX_DEPTH; ++i) {
    uart_putc(',');
    uart_put_i64(path[i]);
  }
  uart_putc(',');
  uart_put_u64(start);
  uart_putc(',');
  uart_put_u64(end);
  uart_putc('\n');
}

void _mlir_ciface_buddyTraceCycleStart(int64_t id) {
  validate_id(id);
  start_cycles[id] = read_cycle();
}

void _mlir_ciface_buddyTraceCycleEnd(int64_t id) {
  const int64_t path[BUDDY_TRACE_MAX_DEPTH] = {id, -1, -1, -1};
  validate_id(id);
  emit_cycle_record(1, path, start_cycles[id], read_cycle());
}

void _mlir_ciface_buddyTraceCycleStartPath(int64_t id, int64_t depth,
                                           int64_t path0, int64_t path1,
                                           int64_t path2, int64_t path3) {
  const int64_t path[BUDDY_TRACE_MAX_DEPTH] = {path0, path1, path2, path3};
  validate_id(id);
  validate_path(depth, path);
  start_cycles[id] = read_cycle();
}

void _mlir_ciface_buddyTraceCycleEndPath(int64_t id, int64_t depth,
                                         int64_t path0, int64_t path1,
                                         int64_t path2, int64_t path3) {
  const int64_t path[BUDDY_TRACE_MAX_DEPTH] = {path0, path1, path2, path3};
  validate_id(id);
  validate_path(depth, path);
  emit_cycle_record(depth, path, start_cycles[id], read_cycle());
}
