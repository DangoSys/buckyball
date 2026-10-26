`include "coherence_config.svh"
`define CF(K, F) `COH_``K``_``F``_OFFSET+:`COH_``K``_``F``_WIDTH
package home_pkg;
  `include "coherence/body.svh"
  `include "agent/transport.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
