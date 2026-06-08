/*
Copyright 2018 Embedded Microprocessor Benchmark Consortium (EEMBC)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original Author: Shay Gal-on
*/

#include "coremark.h"

#if (MEM_METHOD == MEM_MALLOC)
#error "Buckyball CoreMark workload uses MEM_STATIC."
#endif

#if (SEED_METHOD == SEED_VOLATILE)
#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PROFILE_RUN
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif
volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = 0;
#endif

#define NSECS_PER_SEC 1000000000
#define EE_TIMER_TICKER_RATE 1000
#define TIMER_RES_DIVIDER 1
#define EE_TICKS_PER_SEC (NSECS_PER_SEC / TIMER_RES_DIVIDER)

static inline clock_t rdcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, cycle" : "=r"(v));
  return v;
}

static clock_t start_time_val;
static clock_t stop_time_val;

void start_time(void) { start_time_val = rdcycle(); }

void stop_time(void) { stop_time_val = rdcycle(); }

CORE_TICKS get_time(void) { return stop_time_val - start_time_val; }

secs_ret time_in_secs(CORE_TICKS ticks) {
  return ((secs_ret)ticks) / (secs_ret)EE_TICKS_PER_SEC;
}

ee_u32 default_num_contexts = MULTITHREAD;

void portable_init(core_portable *p, int *argc, char *argv[]) {
  (void)argc;
  (void)argv;
  if (sizeof(ee_ptr_int) != sizeof(ee_u8 *)) {
    ee_printf(
        "ERROR! Please define ee_ptr_int to a type that holds a pointer!\n");
  }
  if (sizeof(ee_u32) != 4) {
    ee_printf("ERROR! Please define ee_u32 to a 32b unsigned type!\n");
  }
  p->portable_id = 1;
}

void portable_fini(core_portable *p) { p->portable_id = 0; }
