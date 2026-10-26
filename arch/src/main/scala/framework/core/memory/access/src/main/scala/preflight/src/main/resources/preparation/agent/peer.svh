class peer extends uvm_component;
  `uvm_component_utils(peer)
  virtual preparation_if vif;
  fixture image;
  bit pending;
  int cycle, due;
  bit error;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual preparation_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Preparation interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(posedge vif.clock);
      cycle++;
      if (vif.reset) pending = 0;
      else begin
        if (`PO(PTERESPONSE_READY) && `PI(PTERESPONSE_VALID)) pending = 0;
        if (`PO(PTEREQUEST_VALID) && `PI(PTEREQUEST_READY)) begin
          pending = 1;
          due = cycle + 3;
          error = image.pte_error;
        end
      end
      @(negedge vif.clock);
      vif.backend = '0;
      begin
        int id = `PO(CONTEXTTAG);
        int policy = image.policy[id];
        vif.backend[`PREP_IN_CONTEXTVALID_OFFSET+:`PREP_IN_CONTEXTVALID_WIDTH] = policy != 4;
        vif.backend[`PREP_IN_CONTEXTID_OFFSET+:`PREP_IN_CONTEXTID_WIDTH] = id;
        vif.backend[`PREP_IN_CONTEXTPMP_0_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_0_ADDR_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_0_MASK_OFFSET+:`PREP_IN_CONTEXTPMP_0_MASK_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_A_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_L_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_R_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_R_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_W_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_W_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_1_ADDR_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_MASK_OFFSET+:`PREP_IN_CONTEXTPMP_1_MASK_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_A_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_L_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_R_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_R_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_W_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_W_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_2_ADDR_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_MASK_OFFSET+:`PREP_IN_CONTEXTPMP_2_MASK_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_2_CFG_A_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_2_CFG_L_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_CFG_R_OFFSET+:`PREP_IN_CONTEXTPMP_2_CFG_R_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_2_CFG_W_OFFSET+:`PREP_IN_CONTEXTPMP_2_CFG_W_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_3_ADDR_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_MASK_OFFSET+:`PREP_IN_CONTEXTPMP_3_MASK_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_3_CFG_A_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_3_CFG_L_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_CFG_R_OFFSET+:`PREP_IN_CONTEXTPMP_3_CFG_R_WIDTH] = 0;
        vif.backend[`PREP_IN_CONTEXTPMP_3_CFG_W_OFFSET+:`PREP_IN_CONTEXTPMP_3_CFG_W_WIDTH] = 0;
        if (policy == 1 || policy == 2 || policy == 6) begin
          vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_A_WIDTH] = 3;
          vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_L_WIDTH] = 1;
          vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_R_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_R_WIDTH]=policy!=1;
          vif.backend[`PREP_IN_CONTEXTPMP_0_CFG_W_OFFSET+:`PREP_IN_CONTEXTPMP_0_CFG_W_WIDTH]=policy==6;
          vif.backend[`PREP_IN_CONTEXTPMP_0_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_0_ADDR_WIDTH] = '1;
          vif.backend[`PREP_IN_CONTEXTPMP_0_MASK_OFFSET+:`PREP_IN_CONTEXTPMP_0_MASK_WIDTH] = '1;
        end
        if (policy == 3) begin
          vif.backend[`PREP_IN_CONTEXTPMP_0_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_0_ADDR_WIDTH]='h2008>>2;
          vif.backend[`PREP_IN_CONTEXTPMP_1_ADDR_OFFSET+:`PREP_IN_CONTEXTPMP_1_ADDR_WIDTH]='h2004>>2;
          vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_A_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_A_WIDTH] = 1;
          vif.backend[`PREP_IN_CONTEXTPMP_1_CFG_L_OFFSET+:`PREP_IN_CONTEXTPMP_1_CFG_L_WIDTH] = 1;
        end
      end
      vif.backend[`PREP_IN_PTEREQUEST_READY_OFFSET+:`PREP_IN_PTEREQUEST_READY_WIDTH]=cycle%3!=0;
      vif.backend[`PREP_IN_PTERESPONSE_VALID_OFFSET+:`PREP_IN_PTERESPONSE_VALID_WIDTH]=pending&&cycle>=due;
      vif.backend[`PREP_IN_PTERESPONSE_BITS_DATA_OFFSET+:`PREP_IN_PTERESPONSE_BITS_DATA_WIDTH]=64'hcf;
      vif.backend[`PREP_IN_PTERESPONSE_BITS_ERROR_OFFSET+:`PREP_IN_PTERESPONSE_BITS_ERROR_WIDTH]=error;
    end
  endtask
endclass
