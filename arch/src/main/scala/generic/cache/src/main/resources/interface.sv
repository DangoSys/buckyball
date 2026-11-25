`include "cache_config.svh"
interface cache_request_if (
    input logic clock
);
  logic reset, valid, ready;
  logic [`CACHE_ID_BITS-1:0] id;
  logic [2:0] op;
  logic [`CACHE_ADDR_BITS-1:0] addr;
  logic [`CACHE_WAY_BITS-1:0] way;
  logic [`CACHE_LINE_BITS-1:0] data;
  logic [`CACHE_LINE_BYTES-1:0] mask;
  logic [`CACHE_META_BITS-1:0] metadata;
  logic [`CACHE_WAYS-1:0] eligible;
  clocking cb @(posedge clock);
    default input #1step output #0;
    input ready;
    output reset, valid, id, op, addr, way, data, mask, metadata, eligible;
  endclocking
  assert property (@(posedge clock) disable iff (reset) valid && !ready |=> valid && $stable(
      {id, op, addr, way, data, mask, metadata, eligible}
  ))
  else $fatal(1, "Cache request changed under backpressure");
endinterface

interface cache_response_if (
    input logic clock
);
  logic reset, valid, ready;
  logic [`CACHE_ID_BITS-1:0] id;
  logic hit, available, entry_valid;
  logic [ `CACHE_WAY_BITS-1:0] way;
  logic [`CACHE_ADDR_BITS-1:0] addr;
  logic [`CACHE_LINE_BITS-1:0] data;
  logic [`CACHE_META_BITS-1:0] metadata;
  clocking cb @(posedge clock);
    default input #1step output #0;
    input valid, id, hit, available, entry_valid, way, addr, data, metadata;
    output ready;
  endclocking
  assert property (@(posedge clock) disable iff (reset) valid && !ready |=> valid && $stable(
      {id, hit, available, entry_valid, way, addr, data, metadata}
  ))
  else $fatal(1, "Cache response changed under backpressure");
endinterface
