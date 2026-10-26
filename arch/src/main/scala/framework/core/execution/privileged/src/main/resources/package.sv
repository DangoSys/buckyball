`include "parameters.svh"
`define PIN(P, N) P[`PRIV_IN_``N``_OFFSET +: `PRIV_IN_``N``_WIDTH]
`define POUT(P, N) P[`PRIV_OUT_``N``_OFFSET +: `PRIV_OUT_``N``_WIDTH]
package privileged_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle privileged_create(input int unsigned address_bits);
  import "DPI-C" function void privileged_destroy(input chandle model);
  import "DPI-C" function void privileged_reset(input chandle model);
  import "DPI-C" function void privileged_evaluate(
    input chandle model,
    input int unsigned address,
    input longint unsigned operand,
    input int unsigned operation,
    input byte unsigned wr,
    commit,
    trap,
    input longint unsigned pc,
    cause,
    value,
    input byte unsigned returning,
    input int unsigned return_mode,
    input longint unsigned hart,
    interrupts,
    input byte unsigned retired,
    floating,
    input int unsigned flags
  );
  import "DPI-C" function longint unsigned privileged_value(
    input chandle model,
    input int unsigned index
  );
  import "DPI-C" function longint unsigned privileged_pmp(
    input chandle model,
    input int unsigned index,
    field
  );
  import "DPI-C" function byte unsigned privileged_permitted(
    input chandle model,
    input longint unsigned address,
    input int unsigned size,
    input byte unsigned rd,
    wr,
    execute,
    input int unsigned privilege
  );
  import "DPI-C" function void privileged_advance(input chandle model);
  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "env/model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
