#ifndef MONITOR_HALT_H_
#define MONITOR_HALT_H_

#ifdef __cplusplus
extern "C" {
#endif

// DPI-C function called by HaltDPI.sv when ebreak is detected
// Triggers bdb sim_exit() from within the simulation loop
void dpi_sim_halt(void);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_HALT_H_
