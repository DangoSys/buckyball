class mxquant_step_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxquant_step_seq)

  int unsigned step;

  function new(string name = "mxquant_step_seq");
    super.new(name);
  endfunction

  task body();
    mxquant_cmd_item item;
    item = mxquant_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_step(step);
    finish_item(item);
  endtask
endclass
