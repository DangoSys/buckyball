#include <bbsw/test.h>
#include <stdint.h>
#include <stdio.h>

#include "boardsupport.h"

typedef void (*init_fn)(void);
typedef void (*warm_fn)(int);
typedef int (*bench_fn)(void);
typedef int (*verify_fn)(int);

typedef struct {
  const char *name;
  init_fn init;
  warm_fn warm;
  bench_fn bench;
  verify_fn verify;
} bench_case;

#define DECL_BENCH(prefix)                                                     \
  void prefix##_initialise_benchmark(void);                                    \
  void prefix##_warm_caches(int heat);                                         \
  int prefix##_benchmark(void);                                                \
  int prefix##_verify_benchmark(int result)

DECL_BENCH(aha_mont64);
DECL_BENCH(crc32);
DECL_BENCH(edn);
DECL_BENCH(huffbench);
DECL_BENCH(matmult_int);
DECL_BENCH(minver);
DECL_BENCH(nbody);
DECL_BENCH(nettle_aes);
DECL_BENCH(nettle_sha256);
DECL_BENCH(nsichneu);
DECL_BENCH(picojpeg);
DECL_BENCH(qrduino);
DECL_BENCH(sglib_combined);
DECL_BENCH(slre);
DECL_BENCH(st);
DECL_BENCH(statemate);
DECL_BENCH(ud);
DECL_BENCH(wikisort);

#define BENCH_ENTRY(name, prefix)                                              \
  {name, prefix##_initialise_benchmark, prefix##_warm_caches,                  \
   prefix##_benchmark, prefix##_verify_benchmark}

static const bench_case benches[] = {
    BENCH_ENTRY("aha-mont64", aha_mont64),
    BENCH_ENTRY("crc32", crc32),
    BENCH_ENTRY("edn", edn),
    BENCH_ENTRY("huffbench", huffbench),
    BENCH_ENTRY("matmult-int", matmult_int),
    BENCH_ENTRY("minver", minver),
    BENCH_ENTRY("nbody", nbody),
    BENCH_ENTRY("nettle-aes", nettle_aes),
    BENCH_ENTRY("nettle-sha256", nettle_sha256),
    BENCH_ENTRY("nsichneu", nsichneu),
    BENCH_ENTRY("picojpeg", picojpeg),
    BENCH_ENTRY("qrduino", qrduino),
    BENCH_ENTRY("sglib-combined", sglib_combined),
    BENCH_ENTRY("slre", slre),
    BENCH_ENTRY("st", st),
    BENCH_ENTRY("statemate", statemate),
    BENCH_ENTRY("ud", ud),
    BENCH_ENTRY("wikisort", wikisort),
};

static inline uint64_t rdcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, cycle" : "=r"(v));
  return v;
}

int main(void) {
#ifdef MULTICORE
  unsigned long hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != 0) {
    while (1) {
      __asm__ volatile("wfi");
    }
  }
#endif

  int failures = 0;
  uint64_t total = 0;

  initialise_board();
  printf("Embench top: %lu benchmarks\n",
         (unsigned long)(sizeof(benches) / sizeof(benches[0])));

  for (unsigned long i = 0; i < sizeof(benches) / sizeof(benches[0]); ++i) {
    const bench_case *b = &benches[i];

    b->init();
    b->warm(WARMUP_HEAT);

    uint64_t start = rdcycle();
    int result = b->bench();
    uint64_t cycles = rdcycle() - start;
    int ok = b->verify(result);

    total += cycles;
    printf("embench %-15s cycles=%lu result=%d %s\n", b->name,
           (unsigned long)cycles, result, ok ? "PASS" : "FAIL");

    if (!ok) {
      failures++;
    }
  }

  printf("Embench top total cycles=%lu failures=%d\n", (unsigned long)total,
         failures);

  if (failures == 0) {
    bb_test_pass();
    while (1) {
    }
  }

  bb_test_fail();
}
