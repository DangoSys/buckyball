interface permission_control_if (
    input logic clock
);
  logic reset, context_valid, request_valid, request_ready, response_valid, response_ready;
  logic write, is_pte, allow;
  logic [1:0] privilege;
  logic [7:0] id, context_id, response_id;
  logic [12:0] bytes;
  logic [43:0] address;
  logic [31:0] pmp_config;
  logic [63:0] pmp_addr[4];
  clocking cb @(posedge clock);
    default input #1step output #0;
    output reset, context_valid, request_valid, response_ready, write, is_pte;
    output privilege, id, context_id, bytes, address, pmp_config, pmp_addr;
    input request_ready, response_valid, response_id, allow;
  endclocking
endinterface

package permission_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual permission_control_if control;
    int checked = 0;
    localparam longint unsigned BASE = 'h80000000;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 100us;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual permission_control_if)::get(this, "", "control", control))
        `uvm_fatal("VIF", "Permission interface missing")
    endfunction
    task initialize();
      control.cb.reset <= 1;
      control.cb.context_valid <= 0;
      control.cb.request_valid <= 0;
      control.cb.response_ready <= 0;
      control.cb.pmp_config <= 0;
      control.cb.context_id <= 0;
      control.cb.id <= 0;
      control.cb.address <= 0;
      control.cb.bytes <= 0;
      control.cb.write <= 0;
      control.cb.is_pte <= 0;
      control.cb.privilege <= 0;
      foreach (control.pmp_addr[i]) control.cb.pmp_addr[i] <= 0;
      repeat (3) @(control.cb);
      control.cb.reset <= 0;
      repeat (2) @(control.cb);
    endtask
    task check_range(longint unsigned address, int bytes, bit write, int privilege, bit expected,
                     bit is_pte = 0, bit context_valid = 1, bit wrong_tag = 0, bit mutate = 0,
                     int hold = 0);
      int tag = checked + 1;
      control.cb.id <= tag;
      control.cb.context_id <= wrong_tag ? tag + 1 : tag;
      control.cb.context_valid <= context_valid;
      control.cb.address <= address;
      control.cb.bytes <= bytes;
      control.cb.write <= write;
      control.cb.is_pte <= is_pte;
      control.cb.privilege <= privilege;
      control.cb.request_valid <= 1;
      do @(control.cb); while (!control.cb.request_ready);
      control.cb.request_valid <= 0;
      if (mutate) begin
        control.cb.pmp_config <= 0;
        foreach (control.pmp_addr[i]) control.cb.pmp_addr[i] <= 0;
        control.cb.context_valid <= 0;
        control.cb.context_id <= tag + 1;
      end
      do @(control.cb); while (!control.cb.response_valid);
      for (int cycle = 0; cycle <= hold; ++cycle) begin
        if (!control.cb.response_valid || control.cb.response_id !== tag ||
            control.cb.allow !== expected || control.cb.request_ready)
          `uvm_fatal("PERMISSION", $sformatf(
                     "case%0d PA=%h bytes=%0d write=%0d prv=%0d expected=%b got=%b",
                     tag,
                     address,
                     bytes,
                     write,
                     privilege,
                     expected,
                     control.cb.allow
                     ))
        if (cycle < hold) @(control.cb);
      end
      control.cb.response_ready <= 1;
      @(control.cb);
      control.cb.response_ready <= 0;
      @(control.cb);
      ++checked;
    endtask
    task execute();
      initialize();
      check_range(BASE, 64, 0, 3, 1);
      check_range(BASE, 64, 0, 1, 0);
      check_range('h90000000, 64, 0, 3, 1);
      check_range('h90000000, 64, 1, 3, 0);
      check_range(BASE + 'h2000000, 64, 0, 3, 0);
      check_range(BASE, 0, 0, 3, 0);
      check_range(BASE, 4097, 0, 3, 0);
      check_range(BASE, 64, 0, 2, 0);
      check_range(BASE, 64, 0, 3, 0, 0, 0);
      check_range(BASE, 64, 0, 3, 0, 0, 1, 1);
      check_range('h1000000000, 64, 0, 3, 0);

      control.cb.pmp_config  <= 'h1b;  // NAPOT R/W, 8 KiB at BASE.
      control.cb.pmp_addr[0] <= BASE + 'hffc;
      check_range(BASE, 4096, 0, 1, 1);
      check_range(BASE + 1, 127, 0, 1, 1);
      check_range(BASE + 8192 - 64, 64, 1, 1, 1);
      check_range(BASE + 8192, 64, 0, 1, 0);
      check_range(BASE, 8, 0, 1, 1, 1);
      check_range(BASE, 8, 1, 1, 0, 1);
      check_range(BASE, 8, 0, 0, 0, 1);
      check_range(BASE, 64, 0, 1, 0, 1);

      control.cb.pmp_config <= 'h99;  // Locked NAPOT read-only, applies in M-mode too.
      check_range(BASE, 64, 0, 3, 1);
      check_range(BASE, 64, 1, 3, 0);
      check_range(BASE, 4096, 0, 1, 1, 0, 1, 0, 1, 9);  // Captured PMP survives changes.

      // Higher-priority 64-byte read-only entry in the middle of a writable region.
      control.cb.pmp_config  <= 'h1b19;
      control.cb.pmp_addr[0] <= BASE + 64 + 28;
      control.cb.pmp_addr[1] <= BASE + 'hffc;
      check_range(BASE, 192, 0, 1, 1);
      check_range(BASE, 192, 1, 1, 0);
      check_range(BASE, 64, 1, 1, 1);
      check_range(BASE + 64, 64, 1, 1, 0);
      check_range(BASE + 128, 64, 1, 1, 1);

      // Two adjacent TOR entries: check each actual line rather than rejecting a page envelope.
      control.cb.pmp_config  <= 'h0b0900;
      control.cb.pmp_addr[0] <= BASE;
      control.cb.pmp_addr[1] <= BASE + 64;
      control.cb.pmp_addr[2] <= BASE + 128;
      check_range(BASE, 128, 0, 1, 1);
      check_range(BASE, 64, 1, 1, 0);
      check_range(BASE + 64, 64, 1, 1, 1);
      check_range(BASE, 128, 1, 1, 0);

      // A denied hole blocks a proposed union while both original pixels remain permitted.
      control.cb.pmp_config <= 'h1b18;
      control.cb.pmp_addr[0] <= BASE + 20;  // Higher-priority 16-byte denied NAPOT hole at BASE+16.
      control.cb.pmp_addr[1] <= BASE + 'hffc;
      check_range(BASE, 96, 0, 1, 0);
      check_range(BASE, 16, 0, 1, 1);
      check_range(BASE + 64, 32, 0, 1, 1);

      // Cancel an accepted long check by joint reset. No stale response may escape.
      control.cb.pmp_config <= 'h1b;
      control.cb.pmp_addr[0] <= BASE + 'hffc;
      control.cb.context_valid <= 1;
      control.cb.context_id <= 99;
      control.cb.id <= 99;
      control.cb.address <= BASE;
      control.cb.bytes <= 4096;
      control.cb.write <= 0;
      control.cb.is_pte <= 0;
      control.cb.privilege <= 1;
      control.cb.request_valid <= 1;
      do @(control.cb); while (!control.cb.request_ready);
      control.cb.request_valid <= 0;
      repeat (4) @(control.cb);
      initialize();
      repeat (8) begin
        @(control.cb);
        if (control.cb.response_valid) `uvm_fatal("RESET", "Old permission response survived reset")
      end
      check_range(BASE, 64, 0, 3, 1);
      `uvm_info("PASS", $sformatf("%0d frozen-context PMP/PMA checks passed", checked), UVM_LOW)
    endtask
  endclass
endpackage

module permission_tb;
  import uvm_pkg::*;
  import permission_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  permission_control_if control (clock);
  PermissionVerification dut (
      .clock(clock),
      .reset(control.reset),
      .io_request_valid(control.request_valid),
      .io_request_ready(control.request_ready),
      .io_request_bits_id(control.id),
      .io_request_bits_pa(control.address),
      .io_request_bits_bytes(control.bytes),
      .io_request_bits_write(control.write),
      .io_request_bits_isPte(control.is_pte),
      .io_request_bits_privilege(control.privilege),
      .io_response_valid(control.response_valid),
      .io_response_ready(control.response_ready),
      .io_response_bits_id(control.response_id),
      .io_response_bits_allow(control.allow),
      .io_contextValid(control.context_valid),
      .io_contextId(control.context_id),
      .io_pmpConfig(control.pmp_config),
      .io_pmpAddresses_0(control.pmp_addr[0]),
      .io_pmpAddresses_1(control.pmp_addr[1]),
      .io_pmpAddresses_2(control.pmp_addr[2]),
      .io_pmpAddresses_3(control.pmp_addr[3])
  );
  initial begin
    uvm_config_db#(virtual permission_control_if)::set(null, "uvm_test_top*", "control", control);
    run_test("protocol_test");
  end
endmodule
