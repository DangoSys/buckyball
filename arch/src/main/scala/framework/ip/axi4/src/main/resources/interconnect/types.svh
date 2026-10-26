`include "parameters.svh"
package axi_types;
  localparam int M = `AXI_MASTERS;
  localparam int D = `AXI_DATA_BITS;
  localparam int I = `AXI_ID_BITS;
  typedef struct packed {
    logic [I-1:0] id;
    logic [`AXI_ADDRESS_BITS-1:0] addr;
    logic [7:0] len;
    logic [2:0] size;
    logic [1:0] burst;
    logic lock;
    logic [3:0] cache;
    logic [2:0] prot;
    logic [3:0] qos;
    logic [3:0] region;
  } address;
  typedef struct packed {
    logic [D-1:0] data;
    logic [D/8-1:0] strb;
    logic last;
  } write_data;
  typedef struct packed {
    logic [I-1:0] id;
    logic [D-1:0] data;
    logic [1:0] resp;
    logic last;
  } read_data;
  typedef struct packed {
    logic [I-1:0] id;
    logic [1:0]   resp;
  } response;
  typedef struct packed {
    logic aw_valid, aw_ready, ar_valid, ar_ready, w_valid, w_ready;
    logic r_valid, r_ready, b_valid, b_ready;
    address aw, ar;
    write_data w;
    read_data r;
    response b;
  } port_signals;
  function automatic logic [D-1:0] read_pattern(address addr, int beat);
    logic [D-1:0] value;
    for (int b = 0; b < D / 8; b++) value[b*8+:8] = (addr.addr >> 3) ^ (beat * 17) ^ b;
    return value;
  endfunction
endpackage
