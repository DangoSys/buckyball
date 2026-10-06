package chi_bad_requester_req_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual mesh_fabric_if vif;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      timeout = 10us;
      if (!uvm_config_db#(virtual mesh_fabric_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "requester mesh interface missing")
    endfunction
    task execute();
      // Issue-H base-profile REQ is 137 bits, carried as five 32-bit mesh chunks.
      bit [159:0] request;
      request         = '0;
      request[4+:7]   = 1;  // TgtID: populated RN-F, deliberately the wrong role.
      request[11+:7]  = 2;  // SrcID: another populated requester.
      request[18+:12] = 7;  // TxnID is opaque to transport.
      request[50+:7]  = 7'h01;  // ReadShared.
      request[58+:6]  = 6;
      request[64+:44] = 44'h100;
      request[112]    = 1;
      request[119+:4] = 4'b1100;
      request[123]    = 1;
      request[133]    = 1;
      vif.reset       = 1;
      vif.active      = 0;
      for (int vc = 0; vc < 4; vc++) begin
        vif.valid[0][vc] = 0;
        vif.data[0][vc]  = 0;
        vif.head[0][vc]  = 0;
        vif.tail[0][vc]  = 0;
      end
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      for (int chunk = 0; chunk < 5; chunk++) begin
        vif.valid[0][0] = 1;
        vif.data[0][0]  = request[chunk*32+:32];
        vif.head[0][0]  = chunk == 0;
        vif.tail[0][0]  = chunk == 4;
        do @(posedge vif.clock); while (!vif.ready[0][0]);
        @(negedge vif.clock);
      end
      vif.valid[0][0] = 0;
      `uvm_info("ROLE_INPUT",
                "Delivered all five REQ chunks: SrcID2, TgtID1, REQ VC0, correct HEAD/TAIL",
                UVM_LOW)
      repeat (8) @(negedge vif.clock);
      `uvm_fatal("MISSING_ASSERTION", "requester accepted or discarded a wrong-role REQ")
    endtask
  endclass
endpackage
