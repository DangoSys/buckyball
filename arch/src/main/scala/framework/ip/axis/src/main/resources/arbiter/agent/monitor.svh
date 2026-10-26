class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual arbiter_if vif;
  uvm_analysis_port #(cycle_item) input_ap;
  uvm_analysis_port #(result_item) output_ap;
  int unsigned accepted = 0;
  int unsigned transferred = 0;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    input_ap  = new("input_ap", this);
    output_ap = new("output_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Arbiter monitor requires vif")
  endfunction

  task run_phase(uvm_phase phase);
    cycle_item  observed;
    result_item actual;
    forever begin
      @(posedge vif.clock);
      observed = cycle_item::type_id::create("observed");
      observed.reset = vif.reset;
      observed.valid = vif.in_valid;
      observed.data = vif.in_data;
      observed.keep = vif.in_keep;
      observed.last = vif.in_last;
      observed.ready = vif.out_ready;
      observed.accepted = vif.in_valid & vif.in_ready;
      observed.output_valid = vif.out_valid;
      observed.output_last = vif.out_last;
      input_ap.write(observed);
      if (!vif.reset) begin
        accepted += $countones(observed.accepted);
        if (vif.out_valid && vif.out_ready) transferred++;
        actual = result_item::type_id::create("actual");
        actual.valid = vif.out_valid;
        actual.ready = vif.in_ready;
        actual.data = vif.out_valid ? vif.out_data : 0;
        actual.keep = vif.out_valid ? vif.out_keep : 0;
        actual.last = vif.out_valid ? vif.out_last : 0;
        output_ap.write(actual);
      end
    end
  endtask
endclass
