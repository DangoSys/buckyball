// tsi_stub.cc — stub for SimTSI DPI-C symbol.
// SimTSI is instantiated by chipyard's AbstractConfig via
// WithSimTSIOverSerialTL, but we don't use TSI. This stub satisfies the linker
// without pulling in fesvr.
extern "C" int tsi_tick(int chip_id, unsigned char out_valid,
                        unsigned char *out_ready, int out_bits,
                        unsigned char *in_valid, unsigned char in_ready,
                        int *in_bits) {
  *out_ready = 0;
  *in_valid = 0;
  return 0;
}
