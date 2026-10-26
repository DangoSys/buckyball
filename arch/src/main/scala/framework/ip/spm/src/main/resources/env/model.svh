class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)
  byte unsigned memory[`SPM_BYTES];
  bit initialized[`SPM_BYTES];
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    longint unsigned base = `SPM_BASE;
    longint unsigned count = 64'd1 << req.size;
    logic [64:0] finish = {1'b0, req.address} + count;
    logic [64:0] limit = {1'b0, base} + `SPM_BYTES;
    logic [`SPM_BITS/8-1:0] allowed = (`SPM_BITS / 8)'((64'd1 << count) - 1);
    bit legal = count <= `SPM_BITS/8 && (req.address & (count-1)) == 0 &&
                req.address >= base && finish <= limit &&
                !(req.write && (req.read_only || (req.mask & ~allowed) != 0));
    int offset = req.address - base;
    rsp.value = 0;
    rsp.error = !legal;
    if (legal) begin
      for (int i = 0; i < count; i++) begin
        if (req.write) begin
          if (req.mask[i]) begin
            memory[offset+i] = req.data[8*i+:8];
            initialized[offset+i] = 1;
          end
        end else begin
          if (!initialized[offset+i])
            `uvm_fatal("UNINITIALIZED", "SPM read before byte initialization")
          rsp.value[8*i+:8] = memory[offset+i];
        end
      end
    end
    expected_ap.write(rsp);
  endfunction
endclass
