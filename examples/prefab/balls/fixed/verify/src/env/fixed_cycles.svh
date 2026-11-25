// FIXED's cycles: its BEMU cycle model (emu/src/latency.rs). HSWISH and
// HSIGMOID lanes depend on the A and B rows the command reads, taken from the
// bank model, which holds them unchanged until the response (C differs).
class fixed_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(fixed_cycles)

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void connect_phase(uvm_phase phase);
    bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;
    super.connect_phase(phase);
    $cast(env, get_parent());
    mem_model = env.mem_model;
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    int a[4096];
    int b[4096];
    int unsigned b_rows;
    if (cmd.rs2[2:0] == 1 || cmd.rs2[2:0] == 2) begin
      b_rows = cmd.rs2[9] ? 2 * cmd.iter : 8;
      for (int unsigned row = 0; row < cmd.iter; row++)
      for (int lane = 0; lane < 4; lane++)
      a[row*4+lane] = mem_model.mem[cmd.op1_bank][cmd.op1_group][row][lane*32+:32];
      for (int unsigned row = 0; row < b_rows; row++)
      for (int lane = 0; lane < 4; lane++)
      b[row*4+lane] = mem_model.mem[cmd.op2_bank][cmd.op2_group][row][lane*32+:32];
    end
    return fixed_bemu_cycles(cmd.iter, cmd.rs2, a, b);
  endfunction
endclass
