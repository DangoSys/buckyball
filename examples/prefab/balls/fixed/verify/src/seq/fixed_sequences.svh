class fixed_basic_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(fixed_basic_seq)

  int unsigned case_index;

  function new(string name = "fixed_basic_seq");
    super.new(name);
  endfunction

  task body();
    fixed_cmd_item item;
    item = fixed_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(case_index);
    finish_item(item);
  endtask
endclass
