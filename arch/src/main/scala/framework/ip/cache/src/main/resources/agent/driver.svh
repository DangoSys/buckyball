class driver extends uvm_driver #(request_item);
  `uvm_component_utils(driver)
  virtual cache_request_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cache_request_if)::get(this, "", "request", vif))
      `uvm_fatal("VIF", "Cache request interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    request_item req;
    int cycles;
    vif.reset = 1;
    vif.valid = 0;
    vif.id = 0;
    vif.op = 0;
    vif.addr = 0;
    vif.way = 0;
    vif.data = 0;
    vif.mask = 0;
    vif.metadata = 0;
    vif.eligible = 0;
    forever begin
      seq_item_port.get_next_item(req);
      if (req.reset_command) begin
        vif.cb.valid <= 0;
        vif.cb.reset <= 1;
        repeat (req.reset_cycles) @(vif.cb);
        vif.cb.reset <= 0;
        @(vif.cb);
      end else begin
        vif.cb.valid <= 1;
        vif.cb.id <= req.id;
        vif.cb.op <= req.op;
        vif.cb.addr <= req.addr;
        vif.cb.way <= req.way;
        vif.cb.data <= req.data;
        vif.cb.mask <= req.mask;
        vif.cb.metadata <= req.metadata;
        vif.cb.eligible <= req.eligible;
        cycles = 0;
        do begin
          @(vif.cb);
          cycles++;
        end while (!vif.cb.ready);
        if (req.check_rate)
          evidence::check(cycles == 1,
                          "Cache inserted a request bubble without response backpressure");
        vif.cb.valid <= 0;
        if (req.idle_between) @(vif.cb);
      end
      seq_item_port.item_done();
    end
  endtask
endclass
