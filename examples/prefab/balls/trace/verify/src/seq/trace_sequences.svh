class trace_basic_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(trace_basic_seq)

  int unsigned num_cmds;

  function new(string name = "trace_basic_seq");
    super.new(name);
  endfunction

  task body();
    trace_cmd_item item;
    for (int unsigned i = 0; i < num_cmds; i++) begin
      item = trace_cmd_item::type_id::create("item");
      start_item(item);
      item.load_rust_case(i);
      finish_item(item);
    end
  endtask
endclass
