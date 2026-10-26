class im2col_basic_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(im2col_basic_seq)

  int unsigned case_index;
  int unsigned seed;
  int unsigned bid;

  function new(string name = "im2col_basic_seq");
    super.new(name);
    seed = IM2COL_SEED;
  endfunction

  task body();
    im2col_cmd_item item;
    item = im2col_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(seed, case_index, bid);
    finish_item(item);
  endtask
endclass
