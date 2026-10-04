#pragma once
#include <stdint.h>

// Chip device map, matching framework.system.device.DeviceParams.
#define CLINT_BASE 0x02000000UL
#define PLIC_BASE 0x0C000000UL

#define CLINT_MSIP(h) (*(volatile uint32_t *)(CLINT_BASE + 4 * (h)))
#define CLINT_MTIMECMP(h)                                                      \
  (*(volatile uint64_t *)(CLINT_BASE + 0x4000 + 8 * (h)))
#define CLINT_MTIME (*(volatile uint64_t *)(CLINT_BASE + 0xbff8))

#define PLIC_PRIORITY(s) (*(volatile uint32_t *)(PLIC_BASE + 4 * (s)))
#define PLIC_PENDING (*(volatile uint32_t *)(PLIC_BASE + 0x1000))
#define PLIC_ENABLE(c) (*(volatile uint32_t *)(PLIC_BASE + 0x2000 + 0x80 * (c)))
#define PLIC_THRESHOLD(c)                                                      \
  (*(volatile uint32_t *)(PLIC_BASE + 0x200000 + 0x1000 * (c)))
#define PLIC_CLAIM(c)                                                          \
  (*(volatile uint32_t *)(PLIC_BASE + 0x200004 + 0x1000 * (c)))

#define MIP_MSIP (1UL << 3)
#define MIP_MTIP (1UL << 7)
#define MIP_MEIP (1UL << 11)
#define MSTATUS_MIE (1UL << 3)
#define MCAUSE_INTERRUPT (1UL << 63)

#define read_csr(r)                                                            \
  ({                                                                           \
    uint64_t _v;                                                               \
    asm volatile("csrr %0, " #r : "=r"(_v));                                   \
    _v;                                                                        \
  })
#define write_csr(r, v) asm volatile("csrw " #r ", %0" ::"r"((uint64_t)(v)))
#define set_csr(r, v) asm volatile("csrs " #r ", %0" ::"r"((uint64_t)(v)))
#define clear_csr(r, v) asm volatile("csrc " #r ", %0" ::"r"((uint64_t)(v)))
