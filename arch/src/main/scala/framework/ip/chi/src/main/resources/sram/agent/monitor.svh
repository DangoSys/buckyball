class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual chi_sram_if vif;
  uvm_analysis_port #(sample_item) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_sram_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI monitor interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    sample_item t;
    forever begin
      @(posedge vif.clock);
      t = sample_item::type_id::create("observed");
      t.sampled.reset = vif.reset;
      t.sampled.req_pend = vif.req_pend;
      t.sampled.req_valid = vif.req_valid;
      t.sampled.req_credit = vif.req_credit;
      t.sampled.req_flit = vif.req_flit;
      t.sampled.dat_pend = vif.dat_pend;
      t.sampled.dat_valid = vif.dat_valid;
      t.sampled.dat_credit = vif.dat_credit;
      t.sampled.dat_flit = vif.dat_flit;
      t.sampled.rsp_pend = vif.rsp_pend;
      t.sampled.rsp_valid = vif.rsp_valid;
      t.sampled.rsp_credit = vif.rsp_credit;
      t.sampled.rsp_flit = vif.rsp_flit;
      t.sampled.out_dat_pend = vif.out_dat_pend;
      t.sampled.out_dat_valid = vif.out_dat_valid;
      t.sampled.out_dat_credit = vif.out_dat_credit;
      t.sampled.out_dat_flit = vif.out_dat_flit;
      t.sampled.rx_active_req = vif.rx_active_req;
      t.sampled.rx_active_ack = vif.rx_active_ack;
      t.sampled.tx_active_req = vif.tx_active_req;
      t.sampled.tx_active_ack = vif.tx_active_ack;
      t.sampled.outstanding = vif.outstanding;
      ap.write(t);
    end
  endtask
endclass
