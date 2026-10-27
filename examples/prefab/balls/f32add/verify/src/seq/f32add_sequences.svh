class f32add_basic_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(f32add_basic_seq)

  int unsigned case_index;

  function new(string name = "f32add_basic_seq");
    super.new(name);
  endfunction

  task body();
    f32add_cmd_item item;
    item = f32add_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(case_index);
    finish_item(item);
  endtask
endclass
