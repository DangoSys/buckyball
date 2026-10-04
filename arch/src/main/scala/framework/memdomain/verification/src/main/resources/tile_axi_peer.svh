typedef struct {
  bit [63:0] addr;
  int beat;
} axi_read_t;
typedef struct {
  bit [63:0] addr;
  int due;
} axi_write_t;
axi_read_t reads[int];
axi_write_t writes[int];
int aw_ids[$];
bit [63:0] aw_addresses[$];
bit [511:0] w_lines[$];
bit [63:0] w_masks[$];
bit [511:0] assembly;
bit [63:0] assembly_mask;
int w_beat = 0, r_id = 0, b_id = 0, r_cursor = 0;
bit r_active = 0, b_active = 0;
int axi_reads = 0, axi_writes = 0, final_b_waits = 0;
task automatic axi_service();
  forever begin
    @(sample);
    if (!sample.reset) begin
      if (sample.io_axi_aw_valid && sample.io_axi_aw_ready) begin
        ck(
            sample.io_axi_aw_bits_len==3&&sample.io_axi_aw_bits_size==4&&sample.io_axi_aw_bits_burst==1&&!sample.io_axi_aw_bits_lock,
            "Actual DDR AW line burst");
        aw_ids.push_back(sample.io_axi_aw_bits_id);
        aw_addresses.push_back(sample.io_axi_aw_bits_addr);
      end
      if (sample.io_axi_w_valid && sample.io_axi_w_ready) begin
        ck(sample.io_axi_w_bits_last == (w_beat == 3), "Actual DDR WLAST");
        assembly[w_beat*128+:128] = sample.io_axi_w_bits_data;
        assembly_mask[w_beat*16+:16] = sample.io_axi_w_bits_strb;
        if (w_beat == 3) begin
          w_lines.push_back(assembly);
          w_masks.push_back(assembly_mask);
          w_beat = 0;
        end else w_beat++;
      end
      if (aw_ids.size() && w_lines.size()) begin
        int id = aw_ids.pop_front();
        bit [63:0] addr = aw_addresses.pop_front();
        bit [511:0] data = w_lines.pop_front();
        bit [63:0] mask = w_masks.pop_front();
        ck(!reads.exists(id) && !writes.exists(id), "Actual DDR active AXI ID reused for write");
        for (int beat = 0; beat < 4; beat++)
        tile_peer_write128(addr + 16 * beat, data[128*beat+:64], data[128*beat+64+:64],
                           int'(mask[16*beat+:16]));
        writes[id] = '{addr, cycles + 12};
        axi_writes++;
      end
      if (sample.io_axi_ar_valid && sample.io_axi_ar_ready) begin
        int id = sample.io_axi_ar_bits_id;
        bit [63:0] addr = sample.io_axi_ar_bits_addr;
        ck(
            sample.io_axi_ar_bits_len==3&&sample.io_axi_ar_bits_size==4&&sample.io_axi_ar_bits_burst==1&&!sample.io_axi_ar_bits_lock,
            "Actual DDR AR line burst");
        ck(!reads.exists(id) && !writes.exists(id), "Actual DDR active AXI ID reused for read");
        reads[id] = '{addr, 0};
        axi_reads++;
      end
      if (sample.io_axi_r_valid && sample.io_axi_r_ready) begin
        if (reads[r_id].beat == 3) reads.delete(r_id);
        else reads[r_id].beat++;
        r_active = 0;
        r_cursor = (r_id + 1) % 16;
      end
      if (sample.io_axi_b_valid && sample.io_axi_b_ready) begin
        writes.delete(b_id);
        b_active = 0;
      end
      foreach (writes[id]) if (writes[id].due > cycles) final_b_waits++;
    end
    @(negedge clock);
    io_axi_aw_ready = !reset && cycles % 5 != 0;
    io_axi_w_ready  = !reset && cycles % 7 >= 2;
    io_axi_ar_ready = !reset && cycles % 4 != 0;
    if (!r_active)
      for (int off = 0; off < 16; off++) begin
        int id = (r_cursor + off) % 16;
        if (!r_active && reads.exists(id)) begin
          bit [63:0] addr = reads[id].addr + 16 * reads[id].beat;
          r_active = 1;
          r_id = id;
          io_axi_r_bits_id = id;
          io_axi_r_bits_data = {tile_peer_read64(addr + 8), tile_peer_read64(addr)};
          io_axi_r_bits_last = reads[id].beat == 3;
          io_axi_r_bits_resp = 0;
        end
      end
    if (!b_active)
      for (int id = 15; id >= 0; id--)
      if (!b_active && writes.exists(id) && writes[id].due <= cycles) begin
        b_active = 1;
        b_id = id;
        io_axi_b_bits_id = id;
        io_axi_b_bits_resp = 0;
      end
    io_axi_r_valid = !reset && r_active;
    io_axi_b_valid = !reset && b_active;
  end
endtask
