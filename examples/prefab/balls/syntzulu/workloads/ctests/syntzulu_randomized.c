#include <isa/syntzulu.h>
#include <stdio.h>
#ifndef USE_SHARED_WEIGHTS
#define USE_SHARED_WEIGHTS 1
#endif
#ifndef FRAME_DELAY
#define FRAME_DELAY 0
#endif
// Compile twice with USE_SHARED_WEIGHTS=0/1 and compare all TRACE words.
// Narrow each signed 8-bit weight to 0..3 for the passing differential run.
// Set RANDOM_WEIGHT_MASK to UINT64_MAX for the full-range signed-byte regression.
#ifndef RANDOM_WEIGHT_MASK
#define RANDOM_WEIGHT_MASK UINT64_C(0x0303030303030303)
#endif
#ifndef RANDOM_SEED
#define RANDOM_SEED UINT32_C(0x82519437)
#endif
static volatile uint64_t weights[128] __attribute__((aligned(128)));
static volatile uint32_t voltages[8][4] __attribute__((aligned(128)));
static volatile uint32_t mask_row[4] __attribute__((aligned(128)));
static volatile uint32_t spike_row[4] __attribute__((aligned(128)));
static uint64_t cycle(void) { uint64_t n; asm volatile("rdcycle %0":"=r"(n)); return n; }
static void report(const char *label,uint64_t v) {
  char buf[17]; buf[16]=0;
  for(int i=15;i>=0;i--){buf[i]="0123456789abcdef"[v&15];v>>=4;}
  puts(label);puts(buf);
}
int main(void) {

  uint32_t rng=RANDOM_SEED;
  for(int b=0;b<3;b++) bb_mem_alloc(b,1,1);
  for(unsigned i=0;i<128;i++){
    rng=rng*1664525u+1013904223u;
    weights[i]=((uint64_t)rng<<32);
    rng=rng*1664525u+1013904223u;
    weights[i]|=rng;
    weights[i]&=RANDOM_WEIGHT_MASK;
  }
  asm volatile("fence rw,rw":::"memory");
  if(USE_SHARED_WEIGHTS) bb_mvin((uintptr_t)weights,0,64,1);
  else for(unsigned i=0;i<128;i++){
    syntzulu_weight(2,0,i,weights[i]); syntzulu_weight(2,1,i,weights[i]>>32);
  }
  for(unsigned t=0;t<68;t++){
    rng=rng*1664525u+1013904223u;
    unsigned frame=t<64?(rng>>8)&65535:0;
    unsigned delay=(rng>>29)&3;
    syntzulu_run_frame(0,1,frame,delay,USE_SHARED_WEIGHTS);
    bb_mvout((uintptr_t)voltages,1,8,1);
    asm volatile("fence rw,rw":::"memory");
    for(unsigned i=0;i<8;i++) report("TRACE",voltages[i][0]);
    syntzulu_store_spikes(2); bb_mvout((uintptr_t)mask_row,2,1,1);
    asm volatile("fence rw,rw":::"memory");
    report("TRACE",mask_row[0]);
  }
  puts("SYNtzulu randomized PASS");
  return 0;
}
