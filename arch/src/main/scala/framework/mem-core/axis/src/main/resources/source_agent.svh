class source_driver extends uvm_driver #(item);
  `uvm_component_utils(source_driver)

  virtual axis_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual axis_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("AXIS_VIF", "source driver requires vif")
    end
  endfunction

  task run_phase(uvm_phase phase);
    item req;
    vif.tvalid <= 1'b0;
    forever begin
      seq_item_port.get_next_item(req);
      do @(negedge vif.clock); while (vif.reset);
      vif.tvalid <= 1'b1;
      vif.tdata  <= req.data;
      vif.tkeep  <= req.keep;
      vif.tlast  <= req.last;
      do @(posedge vif.clock); while (!vif.tready);
      @(negedge vif.clock);
      vif.tvalid <= 1'b0;
      seq_item_port.item_done();
    end
  endtask
endclass

class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)

  virtual axis_if vif;
  uvm_analysis_port #(item) ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    ap = new("ap", this);
    if (!uvm_config_db#(virtual axis_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("AXIS_VIF", "monitor requires vif")
    end
  endfunction

  task run_phase(uvm_phase phase);
    item observed;
    forever begin
      @(posedge vif.clock);
      if (!vif.reset && vif.tvalid && vif.tready) begin
        observed = item::type_id::create("observed");
        observed.data = vif.tdata;
        observed.keep = vif.tkeep;
        observed.last = vif.tlast;
        ap.write(observed);
      end
    end
  endtask
endclass

class source_agent extends uvm_agent;
  `uvm_component_utils(source_agent)

  uvm_sequencer #(item) seqr;
  source_driver driver;
  monitor mon;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    seqr   = uvm_sequencer#(item)::type_id::create("seqr", this);
    driver = source_driver::type_id::create("driver", this);
    mon    = monitor::type_id::create("mon", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    driver.seq_item_port.connect(seqr.seq_item_export);
  endfunction
endclass
