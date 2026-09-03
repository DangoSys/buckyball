class smatmul_chain_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(smatmul_chain_seq)

  int unsigned case_index;
  int unsigned seed = MATRIX_SEED;
  int unsigned bid;

  function new(string name = "smatmul_chain_seq");
    super.new(name);
  endfunction

  task body();
    int unsigned count;
    smatmul_case_load(seed, case_index, bid);
    count = smatmul_case_num_commands();
    if (count < 2 || count > 3) `uvm_fatal("CASE", "command count must be 2 or 3")
    for (int unsigned index = 0; index < count; index++) begin
      smatmul_cmd_item item;
      item = smatmul_cmd_item::type_id::create($sformatf("item_%0d", index));
      start_item(item);
      item.load_rust_command(index);
      finish_item(item);
    end
  endtask
endclass
