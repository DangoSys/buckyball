class driver extends uvm_driver #(batch);
  `uvm_component_utils(driver)
  virtual bus_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual bus_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "AXI driver requires an interface")
  endfunction
  task run_phase(uvm_phase phase);
    batch req;
    address posted_reads[$], posted_writes[$];
    int read_beat[$];
    int w_sent[M];
    int w_completed, r_completed, b_completed, cycles;
    int read_selected, write_selected;
    int ar_sent[M], aw_sent[M];
    bit write_responded[$];
    address a;
    for (int m = 0; m < M; m++) begin
      vif.ports[m].ar_valid = 0;
      vif.ports[m].aw_valid = 0;
      vif.ports[m].w_valid = 0;
      vif.ports[m].r_ready = 0;
      vif.ports[m].b_ready = 0;
      vif.ports[m].ar = '0;
      vif.ports[m].aw = '0;
      vif.ports[m].w = '0;
    end
    vif.ports[M].ar_ready = 0;
    vif.ports[M].aw_ready = 0;
    vif.ports[M].w_ready = 0;
    vif.ports[M].r_valid = 0;
    vif.ports[M].b_valid = 0;
    vif.ports[M].r = '0;
    vif.ports[M].b = '0;
    forever begin
      seq_item_port.get_next_item(req);
      posted_reads.delete();
      posted_writes.delete();
      read_beat.delete();
      write_responded.delete();
      w_completed = 0;
      r_completed = 0;
      b_completed = 0;
      cycles = 0;
      read_selected = -1;
      write_selected = -1;
      foreach (w_sent[m]) begin
        w_sent[m]  = 0;
        ar_sent[m] = 0;
        aw_sent[m] = 0;
      end
      while (r_completed < 2 * M || b_completed < 2 * M) begin
        do @(negedge vif.clock); while (vif.reset);
        for (int m = 0; m < M; m++) begin
          a = '0;
          a.id = req.serial;
          a.addr = 'h1000 + req.serial * 'h100000 + m * 'h10000 + ar_sent[m] * 'h1000;
          a.len = req.beats - 1;
          a.size = $clog2(D / 8);
          a.burst = (req.beats == 256 || req.beats == 1) ? 1 : req.serial % 3;
          a.cache = req.serial;
          a.prot = req.serial;
          a.qos = req.serial ^ m;
          a.region = ~req.serial;
          vif.ports[m].ar = a;
          a.addr = 'h1800 + req.serial * 'h100000 + m * 'h10000 + aw_sent[m] * 'h1000;
          vif.ports[m].aw = a;
          vif.ports[m].ar_valid = ar_sent[m] < 2;
          vif.ports[m].aw_valid = aw_sent[m] < 2;
          vif.ports[m].w_valid = w_sent[m] < 2 * req.beats;
          a.addr = 'h1800 + req.serial * 'h100000 + m * 'h10000 + (w_sent[m] / req.beats) * 'h1000;
          vif.ports[m].w.data = read_pattern(a, w_sent[m] % req.beats) ^ 'h55aa;
          vif.ports[m].w.strb = (req.serial % 2) ? (w_sent[m] ^ m) : '1;
          vif.ports[m].w.last = w_sent[m] % req.beats == req.beats - 1;
          vif.ports[m].r_ready = !req.stalls || (cycles + m) % 5 == 0;
          vif.ports[m].b_ready = !req.stalls || (cycles + m) % 4 == 0;
        end
        vif.ports[M].ar_ready = !req.stalls || cycles % 4 == 0;
        vif.ports[M].aw_ready = !req.stalls || cycles % 9 == 0;
        vif.ports[M].w_ready  = !req.stalls || cycles % 3 == 0;
        if (read_selected < 0 && cycles > 12) begin
          for (int n = posted_reads.size() - 1; n >= 0; n--) begin
            if (read_beat[n] <= posted_reads[n].len) begin
              read_selected = n;
              break;
            end
          end
        end
        vif.ports[M].r_valid = read_selected >= 0;
        if (read_selected >= 0) begin
          a = posted_reads[read_selected];
          vif.ports[M].r.id = a.id;
          vif.ports[M].r.data = read_pattern(a, read_beat[read_selected]);
          vif.ports[M].r.resp = (req.serial % 3 == 0) ? 0 : (req.serial % 3) + 1;
          vif.ports[M].r.last = read_beat[read_selected] == a.len;
        end
        if (write_selected < 0 && cycles > 15) begin
          for (int n = posted_writes.size() - 1; n >= 0; n--) begin
            if (n < w_completed && !write_responded[n]) begin
              write_selected = n;
              break;
            end
          end
        end
        vif.ports[M].b_valid = write_selected >= 0;
        if (write_selected >= 0) begin
          vif.ports[M].b.id   = posted_writes[write_selected].id;
          vif.ports[M].b.resp = (req.serial % 3 == 0) ? 0 : (req.serial % 3) + 1;
        end
        @(posedge vif.clock);
        for (int m = 0; m < M; m++) begin
          if (vif.ports[m].aw_valid && vif.ports[m].aw_ready) aw_sent[m]++;
          if (vif.ports[m].ar_valid && vif.ports[m].ar_ready) ar_sent[m]++;
          if (vif.ports[m].w_valid && vif.ports[m].w_ready) w_sent[m]++;
        end
        if (vif.ports[M].ar_valid && vif.ports[M].ar_ready) begin
          posted_reads.push_back(vif.ports[M].ar);
          read_beat.push_back(0);
        end
        if (vif.ports[M].aw_valid && vif.ports[M].aw_ready) begin
          posted_writes.push_back(vif.ports[M].aw);
          write_responded.push_back(0);
        end
        if (vif.ports[M].w_valid && vif.ports[M].w_ready) begin
          if (vif.ports[M].w.last) begin
            w_completed++;
          end
        end
        if (vif.ports[M].r_valid && vif.ports[M].r_ready) begin
          read_beat[read_selected]++;
          if (vif.ports[M].r.last) r_completed++;
          read_selected = -1;
        end
        if (vif.ports[M].b_valid && vif.ports[M].b_ready) begin
          write_responded[write_selected] = 1;
          b_completed++;
          write_selected = -1;
        end
        cycles++;
      end
      @(negedge vif.clock);
      for (int m = 0; m < M; m++) begin
        vif.ports[m].ar_valid = 0;
        vif.ports[m].aw_valid = 0;
        vif.ports[m].w_valid  = 0;
      end
      vif.ports[M].r_valid = 0;
      vif.ports[M].b_valid = 0;
      seq_item_port.item_done();
    end
  endtask
endclass
