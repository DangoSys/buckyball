class mxfp2int_step_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxfp2int_step_seq)

  int unsigned step;

  function new(string name = "mxfp2int_step_seq");
    super.new(name);
  endfunction

  task body();
    mxfp2int_cmd_item item;
    item = mxfp2int_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_step(step);
    finish_item(item);
  endtask
endclass
