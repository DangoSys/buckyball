class fp2int_basic_seq extends uvm_sequence #(fp2int_cmd_item);
  `uvm_object_utils(fp2int_basic_seq)

  int unsigned case_index;

  function new(string name = "fp2int_basic_seq");
    super.new(name);
    case_index = 0;
  endfunction

  task body();
    fp2int_cmd_item item;

    item = fp2int_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(32'hBEEF_0001, case_index);
    finish_item(item);
  endtask
endclass
