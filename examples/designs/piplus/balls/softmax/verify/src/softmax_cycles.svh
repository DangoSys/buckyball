// SOFTMAX's cycles: its BEMU cycle model (emu/src/latency.rs), on the input
// rows the command reads from the bank model.
class softmax_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(softmax_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;
    int unsigned words[4096];
    bit [127:0] row;
    $cast(env, get_parent());
    for (int unsigned r = 0; r < 1024; r++) begin
      row = env.mem_model.mem[cmd.op1_bank][cmd.op1_group][r];
      for (int l = 0; l < 4; l++) words[4*r+l] = row[32*l+:32];
    end
    return softmax_bemu_cycles(cmd.iter, cmd.rs2, words);
  endfunction
endclass
