#ifndef _MMIO_H_
#define _MMIO_H_

// mmio_tick: called at posedge (before 2nd eval) to drive ready/valid signals.
// mmio_tick_post: called after 2nd eval to observe RTL's b_ready/r_ready and
//   immediately clear b_pending/r_pending if handshake completed in this cycle.
//
// Address map:
//   0x6000_0000 : simulation exit  — write triggers sim_exit()
//   0x6002_0000 : UART0 TX         — write low byte → putchar
void mmio_tick();
void mmio_tick_post();

#endif // _MMIO_H_
