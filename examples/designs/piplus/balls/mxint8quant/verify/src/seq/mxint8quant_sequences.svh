class mxint8quant_step_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxint8quant_step_seq)

  int unsigned step;

  function new(string name = "mxint8quant_step_seq");
    super.new(name);
  endfunction

  task body();
    mxint8quant_cmd_item item;
    item = mxint8quant_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_step(step);
    finish_item(item);
  endtask
endclass
