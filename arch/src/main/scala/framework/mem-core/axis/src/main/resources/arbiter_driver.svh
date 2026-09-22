class driver extends uvm_component;
  `uvm_component_utils(driver)

  virtual arbiter_if vif;
  bit done;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("VIF", "driver requires vif")
    end
  endfunction

  task run_phase(uvm_phase phase);
    bit [31:0] data0[3] = '{32'h00000000, 32'hffffffff, 32'haaaaaaaa};
    bit [31:0] data1[3] = '{32'hffffffff, 32'h00000000, 32'h55555555};
    bit [3:0] keep0[3] = '{4'h1, 4'hf, 4'h5};
    bit [3:0] keep1[3] = '{4'hf, 4'h1, 4'ha};
    int unsigned index0 = 0;
    int unsigned index1 = 0;
    int unsigned cycle = 0;

    vif.in_valid <= 2'b00;
    vif.in_data[0] <= 32'h00000000;
    vif.in_data[1] <= 32'h00000000;
    vif.in_keep[0] <= 4'h0;
    vif.in_keep[1] <= 4'h0;
    vif.in_last <= 2'b00;
    vif.out_ready <= 1'b0;
    do @(negedge vif.clock); while (vif.reset);

    while (index0 < 3 || index1 < 3) begin
      vif.in_valid[0] <= index0 < 3;
      vif.in_valid[1] <= index1 < 3;
      vif.in_data[0]  <= data0[index0<3?index0 : 2];
      vif.in_data[1]  <= data1[index1<3?index1 : 2];
      vif.in_keep[0]  <= keep0[index0<3?index0 : 2];
      vif.in_keep[1]  <= keep1[index1<3?index1 : 2];
      vif.in_last[0]  <= index0 == 2;
      vif.in_last[1]  <= index1 == 2;
      vif.out_ready   <= cycle % 3 != 0;
      @(posedge vif.clock);
      if (vif.in_valid[0] && vif.in_ready[0]) index0++;
      if (vif.in_valid[1] && vif.in_ready[1]) index1++;
      cycle++;
      @(negedge vif.clock);
    end

    vif.in_valid <= 2'b10;
    vif.in_data[1] <= 32'hffffffff;
    vif.in_keep[1] <= 4'hf;
    vif.in_last <= 2'b10;
    vif.out_ready <= 1'b1;
    do @(posedge vif.clock); while (!vif.in_ready[1]);
    @(negedge vif.clock);

    index0 = 0;
    while (index0 < 2) begin
      vif.in_valid <= 2'b01;
      vif.in_data[0] <= index0 == 0 ? 32'hffffffff : 32'h00000000;
      vif.in_keep[0] <= index0 == 0 ? 4'hf : 4'h1;
      vif.in_last <= index0 == 1;
      @(posedge vif.clock);
      if (vif.in_ready[0]) index0++;
      @(negedge vif.clock);
    end

    vif.in_valid <= 2'b00;
    vif.in_data[0] <= 32'h00000000;
    vif.in_data[1] <= 32'h00000000;
    vif.in_keep[0] <= 4'h0;
    vif.in_keep[1] <= 4'h0;
    vif.in_last <= 2'b00;
    done = 1'b1;
  endtask
endclass
