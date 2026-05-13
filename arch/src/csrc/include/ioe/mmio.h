#ifndef _MMIO_H_
#define _MMIO_H_

// SCU DPI-C interface for Verilator simulation.
// These functions are called from the RTL via DPI-C when software writes to
// the SCU registers (0x6000_0000 for sim_exit, 0x6002_0000 for UART).
//
// The SCU (System Control Unit) is instantiated per-tile inside BBTile and
// intercepts MMIO writes before they reach the system bus.
//
// DPI-C functions (implemented in mmio.cc):
//   - scu_uart_write(hart_id, ch) : called when software writes to UART TX
//   - scu_sim_exit(hart_id, code) : called when software writes to sim exit

#endif // _MMIO_H_
