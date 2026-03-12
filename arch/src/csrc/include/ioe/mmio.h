#ifndef _MMIO_H_
#define _MMIO_H_

// mmio_tick: called once per posedge after eval().
// io_mmio_fire is a 1-cycle register pulse; addr/data are stable latched
// registers.
//
// Address map:
//   0x6000_0000 : simulation exit  — write triggers sim_exit()
//   0x6002_0000 : UART0 TX         — write low byte → putchar
void mmio_tick();

#endif // _MMIO_H_
