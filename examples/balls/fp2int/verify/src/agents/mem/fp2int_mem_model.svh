class fp2int_mem_model extends uvm_component;
  `uvm_component_utils(fp2int_mem_model)

  virtual fp2int_if vif;
  uvm_analysis_imp_stim #(fp2int_cmd_item, fp2int_mem_model) stim_imp;
  bit [127:0] input_words[FP2INT_NUM_WORDS];
  bit have_stim;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
  endfunction

  function void write_stim(fp2int_cmd_item item);
    for (int i = 0; i < FP2INT_NUM_WORDS; i++) begin
      input_words[i] = item.input_words[i];
    end
    have_stim = 1'b1;
  endfunction

  task run_phase(uvm_phase phase);
    init_signals();
    forever begin
      @(posedge vif.clock);
      if (vif.reset) begin
        init_signals();
      end else begin
        handle_read();
        handle_write();
      end
    end
  endtask

  task init_signals();
    vif.cmd_resp_ready <= 1'b1;
    vif.bank_read_0_req_ready <= 1'b1;
    vif.bank_read_0_resp_valid <= 1'b0;
    vif.bank_read_0_resp_bits_data <= '0;
    vif.bank_write_0_req_ready <= 1'b1;
    vif.bank_write_0_resp_valid <= 1'b0;
    vif.bank_write_0_resp_bits_ok <= 1'b1;
    vif.sub_rob_req_ready <= 1'b1;
    vif.mmio_read_req_ready <= 1'b1;
    vif.mmio_read_resp_valid <= 1'b0;
    vif.mmio_read_resp_bits_data <= '0;
    have_stim = 1'b0;
  endtask

  task handle_read();
    if (vif.bank_read_0_resp_valid && vif.bank_read_0_resp_ready) begin
      vif.bank_read_0_resp_valid <= 1'b0;
    end

    if (!vif.bank_read_0_resp_valid && vif.bank_read_0_req_valid && vif.bank_read_0_req_ready) begin
      if (!have_stim) begin
        `uvm_fatal("MEM", "read request received before stimulus")
      end
      if (vif.bank_read_0_req_bits_addr >= FP2INT_NUM_WORDS) begin
        `uvm_fatal("MEM", $sformatf("read addr out of range: %0d", vif.bank_read_0_req_bits_addr))
      end

      vif.bank_read_0_resp_bits_data <= input_words[vif.bank_read_0_req_bits_addr];
      vif.bank_read_0_resp_valid <= 1'b1;
    end
  endtask

  task handle_write();
    if (vif.bank_write_0_resp_valid && vif.bank_write_0_resp_ready) begin
      vif.bank_write_0_resp_valid <= 1'b0;
    end

    if (!vif.bank_write_0_resp_valid &&
        vif.bank_write_0_req_valid &&
        vif.bank_write_0_req_ready) begin
      vif.bank_write_0_resp_valid   <= 1'b1;
      vif.bank_write_0_resp_bits_ok <= 1'b1;
    end
  endtask
endclass
