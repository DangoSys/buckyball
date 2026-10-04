virtual stream_if #(`RAM_A_WIDTH) aw, ar;
virtual stream_if #(`RAM_W_WIDTH) w;
virtual stream_if #(`RAM_B_WIDTH) b;
virtual stream_if #(`RAM_R_WIDTH) r;
typedef struct {
  u64 addr;
  int beat;
  bit [511:0] data;
} axi_read_t;
typedef struct {
  u64 addr;
  int due;
} axi_write_t;
typedef struct packed {
  bit [511:0] data;
  bit [63:0]  mask;
} axi_line_t;
axi_read_t axi_reads[int];
axi_write_t axi_writes[int];
int aw_ids[$];
u64 aw_addresses[$];
axi_line_t w_lines[$], assembly;
int
    w_beat = 0,
    r_id = 0,
    b_id = 0,
    r_cursor = 0,
    final_b_waits = 0,
    axi_reads_count = 0,
    axi_writes_count = 0;
bit r_active = 0, b_active = 0;
bit [`RAM_R_WIDTH-1:0] r_packet;
bit [`RAM_B_WIDTH-1:0] b_packet;
u64 held_write_line = 0;
function void axi_build();
  ck(uvm_config_db#(virtual stream_if #(`RAM_A_WIDTH))::get(this, "", "aw", aw), "Missing AW");
  ck(uvm_config_db#(virtual stream_if #(`RAM_A_WIDTH))::get(this, "", "ar", ar), "Missing AR");
  ck(uvm_config_db#(virtual stream_if #(`RAM_W_WIDTH))::get(this, "", "w", w), "Missing W");
  ck(uvm_config_db#(virtual stream_if #(`RAM_B_WIDTH))::get(this, "", "b", b), "Missing B");
  ck(uvm_config_db#(virtual stream_if #(`RAM_R_WIDTH))::get(this, "", "r", r), "Missing R");
endfunction
function void axi_init();
  aw.ready = 0;
  ar.ready = 0;
  w.ready  = 0;
  r.valid  = 0;
  r.bits   = 0;
  b.valid  = 0;
  b.bits   = 0;
endfunction
function void address_check(bit [`RAM_A_WIDTH-1:0] packet);
  ck(packet[`F(A, LEN)] == 3 && packet[`F(A, SIZE)] == 4 && packet[`F(A, BURST)] == 1 && packet[
     `F(A, LOCK)] == 0, "Wrong nonexclusive AXI line burst");
endfunction
task axi_service();
  forever begin
    @(control.sample);
    if (control.sample.reset) begin
      axi_reads.delete();
      axi_writes.delete();
      aw_ids.delete();
      aw_addresses.delete();
      w_lines.delete();
      assembly = '0;
      w_beat   = 0;
      r_active = 0;
      b_active = 0;
    end else begin
      if (aw.sample.valid && aw.sample.ready) begin
        address_check(aw.sample.bits);
        aw_ids.push_back(aw.sample.bits[`F(A, ID)]);
        aw_addresses.push_back(aw.sample.bits[`F(A, ADDR)]);
      end
      if (w.sample.valid && w.sample.ready) begin
        ck(w.sample.bits[`F(W, LAST)] == (w_beat == 3), "Incorrect AXI WLAST");
        assembly.data[w_beat*128+:128] = w.sample.bits[`F(W, DATA)];
        assembly.mask[w_beat*16+:16]   = w.sample.bits[`F(W, STRB)];
        if (w_beat == 3) begin
          w_lines.push_back(assembly);
          w_beat   = 0;
          assembly = '0;
        end else w_beat++;
      end
      if (aw_ids.size() > 0 && w_lines.size() > 0) begin
        int id = aw_ids.pop_front();
        u64 addr = aw_addresses.pop_front();
        axi_line_t value = w_lines.pop_front();
        bit [511:0] expected;
        ck(!axi_reads.exists(id) && !axi_writes.exists(id), "Active AXI ID reused");
        if (!faults.exists(addr) || !(faults[addr] & 2)) begin
          ram_ref_read(oracle, addr, expected);
          for (int byte_index = 0; byte_index < 64; byte_index++)
          if (value.mask[byte_index])
            ck(value.data[byte_index*8+:8] == expected[byte_index*8+:8],
               "Actual AXI write differs from byte/atomic oracle");
        end
        ram_ref_line_write(ddr, addr, value.data, value.mask, int'(faults.exists(addr
                           ) && (faults[addr] & 2) != 0));
        axi_writes[id] = '{addr, cycle + 12};
        axi_writes_count++;
      end
      if (ar.sample.valid && ar.sample.ready) begin
        int id = ar.sample.bits[`F(A, ID)];
        u64 addr = ar.sample.bits[`F(A, ADDR)];
        bit [511:0] value;
        address_check(ar.sample.bits);
        ck(!axi_reads.exists(id) && !axi_writes.exists(id), "Active AXI read ID reused");
        ram_ref_read(ddr, addr, value);
        axi_reads[id] = '{addr, 0, value};
        axi_reads_count++;
      end
      if (r.sample.valid && r.sample.ready) begin
        if (axi_reads[r_id].beat == 3) axi_reads.delete(r_id);
        else axi_reads[r_id].beat++;
        r_active = 0;
        r_cursor = (r_id + 1) % 16;
      end
      if (b.sample.valid && b.sample.ready) begin
        axi_writes.delete(b_id);
        b_active = 0;
      end
      foreach (axi_writes[id])
      if (held_write_line != 0 && axi_writes[id].addr == held_write_line) final_b_waits++;
    end
    @(negedge control.clock);
    aw.ready = !control.reset && cycle % 5 != 0;
    w.ready  = !control.reset && cycle % 7 >= 2;
    ar.ready = !control.reset && cycle % 4 != 0;
    if (!r_active && !hold_memory)
      for (int off = 0; off < 16; off++) begin
        int id = (r_cursor + off) % 16;
        if (!r_active && axi_reads.exists(
                id
            ) && (held_line == 0 || axi_reads[id].addr != held_line)) begin
          r_active = 1;
          r_id = id;
          r_packet = '0;
          r_packet[`F(R, ID)] = id;
          r_packet[`F(R, DATA)] = axi_reads[id].data[axi_reads[id].beat*128+:128];
          r_packet[`F(R, LAST)] = axi_reads[id].beat == 3;
          if (faults.exists(
                  axi_reads[id].addr
              ) && (faults[axi_reads[id].addr] & 1) && axi_reads[id].beat == 1)
            r_packet[`F(R, RESP)] = 2;
        end
      end
    if (!b_active && !hold_memory)
      for (int id = 15; id >= 0; id--) begin
        if (!b_active && axi_writes.exists(
                id
            ) && axi_writes[id].due <= cycle && (held_line == 0 || axi_writes[id].addr != held_line)
                && (held_write_line == 0 || axi_writes[id].addr != held_write_line)) begin
          b_active = 1;
          b_id = id;
          b_packet = '0;
          b_packet[`F(B, ID)] = id;
          if (faults.exists(axi_writes[id].addr) && (faults[axi_writes[id].addr] & 2))
            b_packet[`F(B, RESP)] = 3;
        end
      end
    r.valid = !control.reset && r_active;
    r.bits  = r_packet;
    b.valid = !control.reset && b_active;
    b.bits  = b_packet;
  end
endtask
function void axi_drained();
  ck(
      !axi_reads.num()&&!axi_writes.num()&&!aw_ids.size()&&!aw_addresses.size()&&!w_lines.size()&&!w_beat&&!r_active&&!b_active&&final_b_waits>=15,
      "AXI descriptors did not drain or final B hold missing");
  `uvm_info("UNCACHED_RAM_DDR_PASS", $sformatf(
            "actual AXI read/write bursts=%0d/%0d finalBheldCycles=%0d; AMO conflict held until real B and all queues drained",
            axi_reads_count,
            axi_writes_count,
            final_b_waits
            ), UVM_LOW)
endfunction
