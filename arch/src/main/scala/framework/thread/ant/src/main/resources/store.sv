// A host or thread port of an Ant store (TLS, TSS), checked against the SPM
// reference model of the IP's DPI library.
package ant_store_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle spm_create(
    input longint unsigned base,
    input int unsigned bytes
  );
  import "DPI-C" function void spm_destroy(input chandle model);
  import "DPI-C" function int unsigned spm_access(
    input chandle model,
    input longint unsigned address,
    input int unsigned size,
    input byte unsigned write,
    input int unsigned mask,
    input longint unsigned low,
    high,
    input byte unsigned read_only,
    output longint unsigned out_low,
    out_high
  );
  class response_item extends uvm_sequence_item;
    logic [127:0] data;
    logic error;
    `uvm_object_utils_begin(response_item)
      `uvm_field_int(data, UVM_ALL_ON)
      `uvm_field_int(error, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "response_item");
      super.new(name);
    endfunction
  endclass
endpackage

interface spm_if (
    input logic clock,
    reset
);
  import uvm_pkg::*;
  import verification_pkg::*;
  import ant_store_pkg::*;
  `include "uvm_macros.svh"
  logic valid = 0, ready, write = 0;
  logic [ 63:0] address = 0;
  logic [  2:0] size = 0;
  logic [127:0] data = 0;
  logic [ 15:0] mask = 0;
  logic response_valid, response_ready = 0, error;
  logic [127:0] result;
  bit cancel = 0, read_only = 0;
  chandle model;
  result_board #(response_item, 0) scoreboard;
  bit enabled = 0, held = 0;
  logic [128:0] held_response;

  always @(posedge clock)
    if (enabled) begin
      if (reset || cancel) begin
        scoreboard.cancel_pending();
        held = 0;
        if (response_valid) `uvm_fatal("CANCEL", "response survived reset/cancel")
      end else begin
        if (held && (!response_valid || {error, result} !== held_response))
          `uvm_fatal("HOLD", "response changed under backpressure")
        held = response_valid && !response_ready;
        held_response = {error, result};
        if (valid && ready) begin
          response_item item;
          longint unsigned low, high;
          int status;
          item = new;
          status = spm_access(model, address, size, write, mask, data[63:0], data[127:64],
                              read_only, low, high);
          if (status == 2) `uvm_fatal("REFERENCE", "read before explicit SRAM initialization")
          item.data  = {high, low};
          item.error = status != 0;
          scoreboard.write_expected(item);
        end
        if (response_valid && response_ready) begin
          response_item item;
          item = new;
          item.data = result;
          item.error = error;
          scoreboard.write_actual(item);
        end
      end
    end
  task send(input logic [63:0] addr, input logic [2:0] width, input bit wr,
            input logic [127:0] value = 0, input logic [15:0] keep = 0);
    @(negedge clock);
    address = addr;
    size = width;
    write = wr;
    data = value;
    mask = keep;
    valid = 1;
    do @(posedge clock); while (!ready);
    @(negedge clock);
    valid = 0;
  endtask
  task take(output logic [127:0] value);
    @(negedge clock);
    response_ready = 1;
    do @(posedge clock); while (!response_valid);
    value = result;
    @(negedge clock);
    response_ready = 0;
  endtask
  task access (input logic [63:0] addr, input logic [2:0] width, input bit wr,
               input logic [127:0] value = 0, input logic [15:0] keep = 0);
    logic [127:0] ignored;
    send(addr, width, wr, value, keep);
    take(ignored);
  endtask
endinterface
