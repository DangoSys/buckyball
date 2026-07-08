class fp2int_read_monitor extends uvm_monitor;
  `uvm_component_utils(fp2int_read_monitor)

  virtual fp2int_if vif;
  uvm_analysis_port #(fp2int_read_item) read_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    read_ap = new("read_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
  endfunction

  task run_phase(uvm_phase phase);
    fp2int_read_item item;

    wait (vif.reset === 1'b0);
    forever begin
      @(posedge vif.clock);
      if (vif.bank_read_0_req_valid && vif.bank_read_0_req_ready) begin
        item = fp2int_read_item::type_id::create("item");
        item.bank_id = vif.bank_read_0_bank_id;
        item.rob_id = vif.bank_read_0_rob_id;
        item.group_id = vif.bank_read_0_group_id;
        item.addr = vif.bank_read_0_req_bits_addr;
        read_ap.write(item);
      end
    end
  endtask
endclass

class fp2int_write_monitor extends uvm_monitor;
  `uvm_component_utils(fp2int_write_monitor)

  virtual fp2int_if vif;
  uvm_analysis_port #(fp2int_write_item) write_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    write_ap = new("write_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
  endfunction

  task run_phase(uvm_phase phase);
    fp2int_write_item item;

    wait (vif.reset === 1'b0);
    forever begin
      @(posedge vif.clock);
      if (vif.bank_write_0_req_valid && vif.bank_write_0_req_ready) begin
        item = fp2int_write_item::type_id::create("item");
        item.bank_id = vif.bank_write_0_bank_id;
        item.rob_id = vif.bank_write_0_rob_id;
        item.addr = vif.bank_write_0_req_bits_addr;
        item.mask = vif.bank_write_0_req_bits_mask;
        item.data = vif.bank_write_0_req_bits_data;
        write_ap.write(item);
      end
    end
  endtask
endclass
