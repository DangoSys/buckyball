class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task access (longint unsigned address, int size, bit write, logic [`SPM_BITS-1:0] data,
               logic [`SPM_BITS/8-1:0] mask, bit read_only, int stall, bit cancel);
    request_item req = request_item::type_id::create("request");
    start_item(req);
    req.address = address;
    req.size = size;
    req.write = write;
    req.data = data;
    req.mask = mask;
    req.read_only = read_only;
    req.stall = stall;
    req.cancel = cancel;
    finish_item(req);
    if (!cancel) sent++;
  endtask
  task body();
    int beat_bytes = `SPM_BITS / 8;
    int maximum = $clog2(beat_bytes);
    logic [`SPM_BITS-1:0] pattern;
    for (int offset = 0; offset < `SPM_BYTES; offset += beat_bytes) begin
      for (int byte_ = 0; byte_ < beat_bytes; byte_++) pattern[8*byte_+:8] = offset + byte_;
      access (`SPM_BASE + offset, maximum, 1, pattern, '1, 0, 0, 0);
    end
    for (int size = 0; size <= maximum; size++) begin
      for (int offset = 0; offset < `SPM_BYTES; offset += (1 << size))
      access (`SPM_BASE + offset, size, 0, 0, 0, 0, 3, 0);
      for (int bit_ = 0; bit_ < (1 << size); bit_++) begin
        access (`SPM_BASE + beat_bytes - (1 << size), size, 1, '1, 1 << bit_, 0, 3, 0);
        access (`SPM_BASE + beat_bytes - (1 << size), size, 0, 0, 0, 0, 0, 0);
      end
    end
    access (`SPM_BASE, maximum, 1, 0, 0, 0, 0, 0);
    access (`SPM_BASE, maximum, 0, 0, 0, 0, 0, 0);
    access (`SPM_BASE - 1, 0, 0, 0, 0, 0, 0, 0);
    access (`SPM_BASE + `SPM_BYTES, 0, 0, 0, 0, 0, 3, 0);
    access (64'hffffffffffffffff, 1, 0, 0, 0, 0, 0, 0);
    access (`SPM_BASE + 1, 1, 0, 0, 0, 0, 3, 0);
    access (`SPM_BASE, maximum + 1, 0, 0, 0, 0, 0, 0);
    access (`SPM_BASE, 0, 1, 0, 2, 0, 0, 0);
    access (`SPM_BASE, maximum, 1, 0, '1, 1, 3, 0);
    access (`SPM_BASE, maximum, 0, 0, 0, 1, 0, 0);
    access (`SPM_BASE, maximum, 1, '1, '1, 0, 0, 1);
    access (`SPM_BASE, maximum, 0, 0, 0, 0, 0, 0);
    access (`SPM_BASE, maximum, 0, 0, 0, 0, 0, 1);
    access (`SPM_BASE, maximum, 0, 0, 0, 0, 3, 0);
  endtask
endclass
