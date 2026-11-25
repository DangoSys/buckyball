class home_sequence extends access_sequence;
  `uvm_object_utils(home_sequence)
  function new(string name = "scenario");
    super.new(name);
  endfunction
  virtual task send(int node, int opcode, longint unsigned address, bit error = 0);
    int key = (node << 12) | (serial & 4095);
    // Requester-owned transaction IDs are reusable after the final response arrives.
    while (test_env.model.requests.exists(key)) @(control.cb);
    super.send(node, opcode, address, error);
  endtask
endclass
