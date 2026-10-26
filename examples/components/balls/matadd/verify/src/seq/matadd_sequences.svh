class matadd_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(matadd_seq)

  int unsigned case_index;
  int unsigned bid;

  function new(string name = "matadd_seq");
    super.new(name);
  endfunction

  task body();
    matadd_cmd_item item;
    item = matadd_cmd_item::type_id::create("item");
    start_item(item);
    item.load_case(case_index, bid);
    finish_item(item);
  endtask
endclass
