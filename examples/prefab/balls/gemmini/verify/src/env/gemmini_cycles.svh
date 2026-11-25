// Gemmini's cycles: its BEMU cycle model (emu/src/latency.rs). PRELOAD and the
// COMPUTEs take their cycles from the configuration the last GEMMINI_CONFIG
// set, which a reset clears to OS without transposes.
class gemmini_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(gemmini_cycles)

  bit ws;
  bit a_transpose;
  bit b_transpose;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    longint unsigned cycles = gemmini_bemu_cycles(cmd.iter, cmd.rs2, ws, a_transpose, b_transpose);
    if (cmd.rs2[3:0] == 0) begin
      ws          = cmd.rs2[4];
      a_transpose = cmd.rs2[7];
      b_transpose = cmd.rs2[8];
    end
    return cycles;
  endfunction

  task run_phase(uvm_phase phase);
    fork
      super.run_phase(phase);
      forever begin
        @(posedge vif.clock);
        if (vif.reset) {ws, a_transpose, b_transpose} = '0;
      end
    join
  endtask
endclass

// The SubROB expansion port against the same model: from the command
// handshake to the first cycle its completion (a loop configuration) or its
// last row (a loop) is valid. A loop takes its bounds from the metadata.
class gemmini_expansion_cycles extends uvm_component;
  `uvm_component_utils(gemmini_expansion_cycles)

  virtual gemmini_expansion_if xif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual gemmini_expansion_if)::get(this, "", "expansion", xif))
      `uvm_fatal("NOVIF", "gemmini_expansion_if not found")
  endfunction

  function bit [63:0] word(int unsigned address);
    bit [63:0] value;
    for (int i = 0; i < 8; i++) value[i*8+:8] = xif.metadata[address+i];
    return value;
  endfunction

  function longint unsigned expected(bit [6:0] funct);
    bit [63:0] w0, w1, w2;
    if (funct == `GEMMINI_LOOP_WS_FUNCT7) begin
      w0 = word(0);
      return gemmini_bemu_loop_ws_cycles(w0[47:32], w0[31:16], w0[15:0]);
    end
    if (funct == `GEMMINI_LOOP_CONV_WS_FUNCT7) begin
      w0 = word(64);
      w1 = word(72);
      w2 = word(80);
      return gemmini_bemu_loop_conv_ws_cycles(w0[15:0], w1[31:16], w1[15:0], w0[47:32], w2[7:0]);
    end
    return gemmini_bemu_loop_config_cycles();
  endfunction

  task run_phase(uvm_phase phase);
    longint unsigned cycle;
    longint unsigned accepted;
    longint unsigned want;
    bit [6:0] funct;
    bit pending;
    forever begin
      @(posedge xif.clock);
      if (xif.reset) begin
        cycle   = 0;
        pending = 1'b0;
        continue;
      end
      cycle++;
      if (xif.command_valid && xif.command_ready) begin
        funct    = xif.command_bits_funct;
        want     = expected(funct);
        accepted = cycle;
        pending  = 1'b1;
      end
      if (pending && (xif.complete_valid || (xif.rows_valid && xif.rows_bits_last))) begin
        if (cycle - accepted != want)
          `uvm_fatal(
              "CYCLES", $sformatf(
              "expansion funct %0d: RTL %0d cycles, BEMU models %0d", funct, cycle - accepted, want
              ))
        pending = 1'b0;
      end
    end
  endtask
endclass
