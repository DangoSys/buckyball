class fp2int_resp_monitor extends uvm_monitor;
  `uvm_component_utils(fp2int_resp_monitor)

  virtual fp2int_if vif;
  uvm_analysis_port #(fp2int_resp_item) resp_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    resp_ap = new("resp_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
  endfunction

  task run_phase(uvm_phase phase);
    fp2int_resp_item item;

    wait (vif.reset === 1'b0);
    forever begin
      @(posedge vif.clock);
      if (vif.cmd_resp_valid && vif.cmd_resp_ready) begin
        item = fp2int_resp_item::type_id::create("item");
        item.rob_id = vif.cmd_resp_bits_rob_id;
        item.is_sub = vif.cmd_resp_bits_is_sub;
        item.sub_rob_id = vif.cmd_resp_bits_sub_rob_id;
        resp_ap.write(item);
      end
    end
  endtask
endclass
