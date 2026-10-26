class command_monitor extends uvm_monitor;
  `uvm_component_utils(command_monitor)
  virtual port_if control;
  virtual axis_if write_vif;
  virtual axis_if read_vif;
  uvm_analysis_port #(operation) ap;
  uvm_analysis_port #(command) commands;
  int unsigned completed = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
    commands = new("commands", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual axis_if)::get(
            this, "", "write_vif", write_vif
        ) || !uvm_config_db#(virtual axis_if)::get(
            this, "", "read_vif", read_vif
        ))
      `uvm_fatal("VIF", "BankSet monitor interfaces are required")
  endfunction
  task run_phase(uvm_phase phase);
    command active;
    operation op;
    int unsigned transferred = 0;
    forever begin
      @(posedge control.clock);
      if (control.reset) begin
        active = null;
        transferred = 0;
      end else begin
        if (control.valid && control.ready) begin
          evidence::check(active == null, "BankSet accepted overlapping commands");
          active = command::type_id::create("observed");
          active.write = control.write;
          active.word_addr = control.addr / 4;
          active.beats = control.beats;
          transferred = 0;
          if (!active.write) begin
            for (int i = 0; i < active.beats; i++) begin
              op = operation::type_id::create("read");
              op.word_addr = active.word_addr + i;
              op.last = i == active.beats - 1;
              ap.write(op);
            end
          end
        end
        if (write_vif.tvalid && write_vif.tready) begin
          if (active == null) `uvm_fatal("COMMAND", "Write beat without a command")
          evidence::check(active.write && transferred < active.beats,
                          "Unexpected BankSet write beat");
          op = operation::type_id::create("write");
          op.write = 1;
          op.word_addr = active.word_addr + transferred;
          op.data = write_vif.tdata;
          op.mask = write_vif.tkeep;
          ap.write(op);
          transferred++;
        end
        if (read_vif.tvalid && read_vif.tready) begin
          if (active == null) `uvm_fatal("COMMAND", "Read beat without a command")
          evidence::check(!active.write && transferred < active.beats,
                          "Unexpected BankSet read beat");
          transferred++;
        end
        if (control.done_valid) begin
          if (active == null) `uvm_fatal("COMMAND", "Completion without a command")
          evidence::check(
              transferred == active.beats && control.error === 1'b0 && control.ready === 1'b0,
              "Invalid BankSet completion");
          if (control.done_ready) begin
            commands.write(active);
            active = null;
            completed++;
          end else active.completion_stall++;
        end
      end
    end
  endtask
endclass
