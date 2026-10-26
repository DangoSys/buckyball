`include "parameters.svh"
`define FI(P, N) P[`FETCH_IN_``N``_OFFSET+:`FETCH_IN_``N``_WIDTH]
`define FO(P, N) P[`FETCH_OUT_``N``_OFFSET+:`FETCH_OUT_``N``_WIDTH]
`define PK(P,
           N) P[`FETCH_OUT_PACKET_BITS_``N``_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_``N``_WIDTH]
package fetch_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function longint unsigned fetch_ref_word(
    input longint unsigned address,
    input int unsigned version
  );
  import "DPI-C" function int unsigned fetch_ref_group(
    input longint unsigned address,
    input int unsigned version
  );
  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "env/model.svh"
  `include "agent/peer.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
