`include "parameters.svh"
package hart_dma_pkg;
  class coverage;
    covergroup behavior with function sample (int kind);
      path: coverpoint kind {
        bins nonresident_read = {0};
        bins concurrent_write = {1};
        bins resident_maintenance = {2};
        bins denied = {3};
        bins bus_error = {4};
      }
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage
module hart_dma_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  logic [ `HD_INPUT_BITS-1:0] inputs = '0;
  wire  [`HD_OUTPUT_BITS-1:0] outputs;
  localparam longint unsigned READ_ADDR = `HD_DDR_BASE + 'h8100, WRITE_ADDR = `HD_DDR_BASE + 'h8200;
  scenario_if control (clock);
  hart_dma_pkg::coverage cov = new();
  DmaMemory dut (
      .clock(clock),
      .reset(reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  task automatic issue_read(input int id, input int beats);
    `HD_IO_CLIENTS_0_READ_BITS_VIRTUALADDRESS = READ_ADDR;
    `HD_IO_CLIENTS_0_READ_BITS_BEATS = beats;
    `HD_IO_CLIENTS_0_READ_BITS_BYTES = beats * 16;
    `HD_IO_CLIENTS_0_READ_BITS_ID = id;
    `HD_IO_CLIENTS_0_READ_VALID = 1;
    do @(posedge clock); while (!`HD_IO_CLIENTS_0_READ_READY);
    @(negedge clock);
    `HD_IO_CLIENTS_0_READ_VALID = 0;
  endtask
  task automatic check_cpu(input longint unsigned address, input bit write_access,
                           input bit allowed);
    `HD_IO_SERVICES_CPUQUERY_VALID = 1;
    `HD_IO_SERVICES_CPUQUERY_PADDR = address;
    `HD_IO_SERVICES_CPUQUERY_WRITE = write_access;
    @(negedge clock);
    evidence::check(`HD_IO_SERVICES_CPUALLOW === allowed,
                    "Incorrect CPU interlock during a live DMA");
  endtask
  task automatic read_data(input logic [127:0] data, input bit last, input int resp);
    `HD_IO_AXI_0_R_VALID = 1;
    `HD_IO_AXI_0_R_BITS_ID = 3;
    `HD_IO_AXI_0_R_BITS_LAST = last;
    `HD_IO_AXI_0_R_BITS_RESP = resp;
    `HD_IO_AXI_0_R_BITS_DATA = data;
    do @(posedge clock); while (!`HD_IO_AXI_0_R_READY);
    evidence::check(
        `HD_IO_CLIENTS_0_READDATA_VALID && `HD_IO_CLIENTS_0_READDATA_BITS_TID===3 &&
       `HD_IO_CLIENTS_0_READDATA_BITS_TDATA===data && `HD_IO_CLIENTS_0_READDATA_BITS_TLAST===last &&
       `HD_IO_CLIENTS_0_READDATA_BITS_TKEEP===16'hffff && `HD_IO_CLIENTS_0_READDATA_BITS_TUSER===resp,
        "Wrong DMA result, mask, terminal marker or AXI response");
    @(negedge clock);
    `HD_IO_AXI_0_R_VALID = 0;
  endtask
  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    `HD_IO_LAUNCH_0_VALID = 1;
    `HD_IO_LAUNCH_0_BITS_EFFECTIVEPRIVILEGE = 3;
    `HD_IO_LAUNCH_1_VALID = 1;
    `HD_IO_LAUNCH_1_BITS_EFFECTIVEPRIVILEGE = 3;
    @(negedge clock);
    `HD_IO_LAUNCH_0_VALID = 0;
    `HD_IO_LAUNCH_1_VALID = 0;
    `HD_IO_SERVICES_DRAINED = 1;
    `HD_IO_PAGE_RESIDENT = 0;
    issue_read(3, 2);
    wait (`HD_IO_AXI_0_AR_VALID);
    repeat (2) begin
      @(negedge clock);
      evidence::check(
          !`HD_IO_SERVICES_MAINTENANCE_VALID && `HD_IO_AXI_0_AR_BITS_ADDR===READ_ADDR &&
        `HD_IO_AXI_0_AR_BITS_LEN===1 && `HD_IO_AXI_0_AR_BITS_SIZE===4 && `HD_IO_AXI_0_AR_BITS_ID===3,
          "Bare translation, burst geometry or nonresident maintenance");
    end
    `HD_IO_AXI_0_AR_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `HD_IO_AXI_0_AR_READY = 0;
    check_cpu(READ_ADDR, 0, 1);
    check_cpu(READ_ADDR + 16, 1, 0);
    check_cpu(READ_ADDR + 64, 1, 1);
    cov.behavior.sample(0);
    `HD_IO_SERVICES_DRAINED = 0;
    `HD_IO_PAGE_RESIDENT = 1;
    `HD_IO_CLIENTS_1_WRITE_BITS_VIRTUALADDRESS = WRITE_ADDR;
    `HD_IO_CLIENTS_1_WRITE_BITS_BEATS = 1;
    `HD_IO_CLIENTS_1_WRITE_BITS_BYTES = 16;
    `HD_IO_CLIENTS_1_WRITE_BITS_ID = 5;
    `HD_IO_CLIENTS_1_WRITE_VALID = 1;
    do @(posedge clock); while (!`HD_IO_CLIENTS_1_WRITE_READY);
    @(negedge clock);
    `HD_IO_CLIENTS_1_WRITE_VALID = 0;
    repeat (20) begin
      @(negedge clock);
      evidence::check(!`HD_IO_AXI_1_AW_VALID && !`HD_IO_SERVICES_MAINTENANCE_VALID,
                      "Resident write bypassed drain");
    end
    `HD_IO_SERVICES_DRAINED = 1;
    wait (`HD_IO_SERVICES_MAINTENANCE_VALID);
    evidence::check(
        `HD_IO_SERVICES_MAINTENANCE_BITS_FIRSTLINE===WRITE_ADDR &&
                    `HD_IO_SERVICES_MAINTENANCE_BITS_LASTLINE===WRITE_ADDR,
        "Wrong cache maintenance range");
    `HD_IO_SERVICES_MAINTENANCE_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `HD_IO_SERVICES_MAINTENANCE_READY = 0;
    read_data(128'h0123456789abcdef, 0, 0);
    evidence::check(!`HD_IO_AXI_1_AW_VALID, "Write posted before maintenance acknowledgement");
    `HD_IO_SERVICES_MAINTAINED_VALID   = 1;
    `HD_IO_SERVICES_MAINTAINED_BITS_OK = 1;
    do @(posedge clock); while (!`HD_IO_SERVICES_MAINTAINED_READY);
    @(negedge clock);
    `HD_IO_SERVICES_MAINTAINED_VALID = 0;
    cov.behavior.sample(2);
    wait (`HD_IO_AXI_1_AW_VALID);
    evidence::check(
        `HD_IO_AXI_1_AW_BITS_ADDR===WRITE_ADDR && `HD_IO_AXI_1_AW_BITS_ID===5 &&
                    `HD_IO_AXI_1_AW_BITS_LEN===0,
        "Write routed to wrong channel");
    check_cpu(WRITE_ADDR, 0, 0);
    check_cpu(WRITE_ADDR + 64, 0, 1);
    `HD_IO_AXI_1_AW_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `HD_IO_AXI_1_AW_READY = 0;
    `HD_IO_CLIENTS_1_WRITEDATA_VALID = 1;
    `HD_IO_CLIENTS_1_WRITEDATA_BITS_TDATA = 128'hbadc010203040506;
    `HD_IO_CLIENTS_1_WRITEDATA_BITS_TKEEP = 16'h03ff;
    `HD_IO_CLIENTS_1_WRITEDATA_BITS_TLAST = 1;
    @(negedge clock);
    `HD_IO_CLIENTS_1_WRITEDATA_VALID = 0;
    wait (`HD_IO_AXI_1_W_VALID);
    evidence::check(
        `HD_IO_AXI_1_W_BITS_DATA===128'hbadc010203040506 && `HD_IO_AXI_1_W_BITS_STRB===16'h03ff &&
                   `HD_IO_AXI_1_W_BITS_LAST,
        "Write data, strobe or last did not survive buffering");
    `HD_IO_AXI_1_W_READY = 1;
    @(posedge clock);
    evidence::check(`HD_IO_CLIENTS_1_WRITEDATA_CREDIT, "Write credit not returned");
    @(negedge clock);
    `HD_IO_AXI_1_W_READY = 0;
    cov.behavior.sample(1);
    read_data(128'hff9876543210, 1, 2);
    `HD_IO_AXI_1_B_VALID = 1;
    `HD_IO_AXI_1_B_BITS_ID = 5;
    `HD_IO_AXI_1_B_BITS_RESP = 2;
    do @(posedge clock); while (!`HD_IO_AXI_1_B_READY);
    @(negedge clock);
    `HD_IO_AXI_1_B_VALID = 0;
    wait (`HD_IO_CLIENTS_1_COMPLETED_VALID);
    evidence::check(
        `HD_IO_CLIENTS_1_COMPLETED_BITS_ID === 5 && `HD_IO_CLIENTS_1_COMPLETED_BITS_ERROR,
        "AXI write error did not reach the originating client");
    `HD_IO_CLIENTS_1_COMPLETED_READY = 1;
    @(negedge clock);
    `HD_IO_CLIENTS_1_COMPLETED_READY = 0;
    check_cpu(WRITE_ADDR, 1, 1);
    cov.behavior.sample(4);
    `HD_IO_LAUNCH_0_VALID = 1;
    `HD_IO_LAUNCH_0_BITS_EFFECTIVEPRIVILEGE = 0;
    @(negedge clock);
    `HD_IO_LAUNCH_0_VALID = 0;
    `HD_IO_PAGE_RESIDENT  = 0;
    issue_read(2, 1);
    do begin
      @(posedge clock);
      evidence::check(!`HD_IO_AXI_0_AR_VALID, "PMP-denied burst issued to DDR");
    end while (!`HD_IO_CLIENTS_0_READDATA_VALID);
    evidence::check(
        `HD_IO_CLIENTS_0_READDATA_BITS_TID===2 && `HD_IO_CLIENTS_0_READDATA_BITS_TLAST &&
                    `HD_IO_CLIENTS_0_READDATA_BITS_TUSER===2 && `HD_IO_CLIENTS_0_READDATA_BITS_TDATA==='0,
        "Denied read returned incorrect completion");
    @(negedge clock);
    cov.behavior.sample(3);
    control.done = 1;
  end
endmodule
