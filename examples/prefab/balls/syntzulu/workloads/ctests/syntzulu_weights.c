#include <isa/syntzulu.h>
#include <stdio.h>
#ifndef USE_SHARED_WEIGHTS
#define USE_SHARED_WEIGHTS 1
#endif
#ifndef FRAME_DELAY
#define FRAME_DELAY 0
#endif
static volatile uint64_t weights[128] __attribute__((aligned(128)));
static volatile uint32_t voltages[8][4] __attribute__((aligned(128)));
static volatile uint32_t mask_row[4] __attribute__((aligned(128)));
static uint64_t cycle(void) { uint64_t n; asm volatile("rdcycle %0":"=r"(n)); return n; }
static void report(const char *label,uint64_t v) {
  char buf[17]; buf[16]=0;
  for(int i=15;i>=0;i--){buf[i]="0123456789abcdef"[v&15];v>>=4;}
  puts(label);puts(buf);
}
int main(void) {
  const uint16_t frames[]={1,0,0x8000,0xffff};
  const uint16_t masks[]={0,0,0xa200,0x0820};
  uint16_t previous[16]={0};
  for(int b=0;b<3;b++) bb_mem_alloc(b,1,1);
  for(unsigned i=0;i<128;i++) weights[i]=i<96 ? UINT64_C(0x0102040810203040)
    : UINT64_C(0x0102030405060708)*(1+((i>>2)&7));
  asm volatile("fence rw,rw":::"memory");
  uint64_t start=cycle();
  if(USE_SHARED_WEIGHTS) bb_mvin((uintptr_t)weights,0,64,1);
  else for(unsigned i=0;i<128;i++){
    syntzulu_weight(2,0,i,weights[i]); syntzulu_weight(2,1,i,weights[i]>>32);
  }
  // Force the destination's previous writer to complete before ending configuration timing.
  bb_mvout((uintptr_t)mask_row,USE_SHARED_WEIGHTS?0:2,1,1);
  asm volatile("fence rw,rw":::"memory");
  uint64_t config=cycle()-start,total=0;
  uint64_t per_frame[4+FRAME_DELAY];
  for(unsigned t=0;t<4+FRAME_DELAY;t++){
    unsigned input=t<4?frames[t]:0;
    start=cycle();
    syntzulu_run_frame(0,1,input,FRAME_DELAY,USE_SHARED_WEIGHTS);
    bb_mvout((uintptr_t)voltages,1,8,1);
    asm volatile("fence rw,rw":::"memory");
    per_frame[t]=cycle()-start;
    total+=per_frame[t];
    syntzulu_store_spikes(2); bb_mvout((uintptr_t)mask_row,2,1,1);
    asm volatile("fence rw,rw":::"memory");
    int src=(int)t-FRAME_DELAY;
    if(mask_row[0]!=(src<0?0:masks[src])) return 20+t;
    for(unsigned i=0;i<16;i++){
      unsigned value=previous[i]*3672u/4096u;
      if(src>=0 && frames[src]) value+=(i%2?40:104)*(1+i/2);
      previous[i]=value>=1024?0:value;
      if((uint16_t)(voltages[i/2][0]>>(16*(i%2)))!=previous[i])return 40+t;
    }
  }
#ifdef HOST_RESET_TEST
  // The host pulses reset at this marker and restarts the workload once.
  // A pending event must disappear before the restarted run's silent second frame.
  syntzulu_run_frame(0,1,0xffff,1,USE_SHARED_WEIGHTS);
  bb_mvout((uintptr_t)voltages,1,8,1);
  asm volatile("fence rw,rw":::"memory");
  puts("RESET_HERE");
#endif
  report("CONFIG",config);report("INFERENCE",total);
  for(unsigned t=0;t<4+FRAME_DELAY;t++) report("FRAME_CYCLES",per_frame[t]);puts("SYNtzulu weight bank PASS");
  return 0;
}
