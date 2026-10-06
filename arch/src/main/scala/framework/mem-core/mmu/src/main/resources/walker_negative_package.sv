`include "walker_config.svh"
package walker_negative_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual walker_control_if control;
    virtual stream_if #(`WALK_REQ_WIDTH) req;
    virtual stream_if #(`WALK_RESP_WIDTH) resp;
    virtual stream_if #(`WALK_ACCESS_WIDTH) access;
    virtual stream_if #(`WALK_RESULT_WIDTH) result;
    int invalid_case;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual walker_control_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual stream_if #(`WALK_REQ_WIDTH))::get(
              this, "", "req", req
          ) || !uvm_config_db#(virtual stream_if #(`WALK_RESP_WIDTH))::get(
              this, "", "resp", resp
          ) || !uvm_config_db#(virtual stream_if #(`WALK_ACCESS_WIDTH))::get(
              this, "", "access", access
          ) || !uvm_config_db#(virtual stream_if #(`WALK_RESULT_WIDTH))::get(
              this, "", "result", result
          ) || !uvm_config_db#(int)::get(
              this, "", "invalid_case", invalid_case
          ))
        `uvm_fatal("VIF", "walker negative interface missing")
    endfunction
    task execute();
      control.reset = 1;
      control.mode = 8;
      control.root_ppn = 16;
      req.valid = 0;
      req.bits = '0;
      resp.ready = 1;
      access.ready = 1;
      result.valid = 0;
      result.bits = '0;
      repeat (4) @(negedge control.clock);
      control.reset = 0;
      req.bits[`WALK_REQ_PRIVILEGE_OFFSET+:`WALK_REQ_PRIVILEGE_WIDTH] = 1;
      case (invalid_case)
        0: control.mode = 9;
        1: req.bits[`WALK_REQ_PRIVILEGE_OFFSET+:`WALK_REQ_PRIVILEGE_WIDTH] = 2;
        2: begin
          req.bits[`WALK_REQ_WRITE_OFFSET]   = 1;
          req.bits[`WALK_REQ_EXECUTE_OFFSET] = 1;
        end
        default: `uvm_fatal("CASE", "Unknown invalid walker request")
      endcase
      req.valid = 1;
      `uvm_info("INVALID_INPUT", $sformatf("Driving walker contract violation %0d", invalid_case),
                UVM_LOW)
      repeat (10) @(negedge control.clock);
      `uvm_fatal("MISSING_ASSERTION",
                 "Illegal walker input did not trigger the specified contract assertion")
    endtask
  endclass
endpackage
