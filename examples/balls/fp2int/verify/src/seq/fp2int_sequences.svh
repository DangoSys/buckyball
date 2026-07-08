class fp2int_basic_seq extends uvm_sequence #(fp2int_cmd_item);
  `uvm_object_utils(fp2int_basic_seq)

  function new(string name = "fp2int_basic_seq");
    super.new(name);
  endfunction

  task body();
    fp2int_cmd_item item;

    item = fp2int_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(32'hBEEF_0001, 0);
    finish_item(item);
  endtask
endclass
