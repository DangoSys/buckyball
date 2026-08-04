class fp2int_basic_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(fp2int_basic_seq)

  int unsigned case_index;
  int unsigned seed;
  int unsigned bid;

  function new(string name = "fp2int_basic_seq");
    super.new(name);
    seed = FP2INT_SEED;
  endfunction

  task body();
    fp2int_cmd_item item;
    item = fp2int_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(seed, case_index, bid);
    finish_item(item);
  endtask
endclass
