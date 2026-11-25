class peer extends uvm_component;
  `uvm_component_utils(peer)
  virtual virtual_if vif;
  fixture image;
  bit auth_pending, cache_pending, uncached_pending;
  bit auth_allow, cache_error;
  logic [63:0] cache_data, uncached_data;
  logic [`VA_LINE_BITS-1:0] cache_line;
  int tag, cycle, auth_due, cache_due, uncached_due;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual virtual_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Virtual access interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(posedge vif.clock);
      cycle++;
      if (vif.reset) begin
        auth_pending = 0;
        cache_pending = 0;
        uncached_pending = 0;
      end else begin
        if (`VO(AUTHORIZATIONRESPONSE_READY) && `VI(AUTHORIZATIONRESPONSE_VALID)) auth_pending = 0;
        if (`VO(CACHERESPONSE_READY) && `VI(CACHERESPONSE_VALID)) cache_pending = 0;
        if (`VO(UNCACHEDRESPONSE_READY) && `VI(UNCACHEDRESPONSE_VALID)) uncached_pending = 0;
        if (`VO(AUTHORIZATIONREQUEST_VALID) && `VI(AUTHORIZATIONREQUEST_READY)) begin
          bit pte = `VO(AUTHORIZATIONREQUEST_BITS_ISPTE);
          auth_pending = 1;
          auth_due = cycle + 3;
          auth_allow = pte ? !image.deny_pte : !image.deny_data;
        end
        if (`VO(CACHEREQUEST_VALID) && `VI(CACHEREQUEST_READY)) begin
          longint unsigned address = `VO(CACHEREQUEST_BITS_ADDR);
          cache_pending = 1;
          cache_due = cycle + 4;
          cache_data = image.ptes.exists(address) ? image.ptes[address] : image.word(address);
          cache_error = image.ptes.exists(address) ? image.errors[address] : image.backend_error;
          cache_line = image.line(address);
        end
        if (`VO(UNCACHEDREQUEST_VALID) && `VI(UNCACHEDREQUEST_READY)) begin
          uncached_pending = 1;
          uncached_due = cycle + 4;
          uncached_data = image.word(`VO(UNCACHEDREQUEST_BITS_ADDR));
          tag = `VO(UNCACHEDREQUEST_BITS_TAG);
        end
      end
      @(negedge vif.clock);
      vif.backend = '0;
      vif.backend[`VA_IN_AUTHORIZATIONREQUEST_READY_OFFSET+:`VA_IN_AUTHORIZATIONREQUEST_READY_WIDTH]=cycle%4!=0;
      vif.backend[`VA_IN_AUTHORIZATIONRESPONSE_VALID_OFFSET+:`VA_IN_AUTHORIZATIONRESPONSE_VALID_WIDTH]=auth_pending&&cycle>=auth_due;
      vif.backend[`VA_IN_AUTHORIZATIONRESPONSE_BITS_OFFSET+:`VA_IN_AUTHORIZATIONRESPONSE_BITS_WIDTH]=auth_allow;
      vif.backend[`VA_IN_CACHEREQUEST_READY_OFFSET+:`VA_IN_CACHEREQUEST_READY_WIDTH]=cycle%5!=0;
      vif.backend[`VA_IN_UNCACHEDREQUEST_READY_OFFSET+:`VA_IN_UNCACHEDREQUEST_READY_WIDTH]=cycle%3!=0;
      vif.backend[`VA_IN_CACHERESPONSE_VALID_OFFSET+:`VA_IN_CACHERESPONSE_VALID_WIDTH]=cache_pending&&cycle>=cache_due;
      vif.backend[`VA_IN_CACHERESPONSE_BITS_DATA_OFFSET+:`VA_IN_CACHERESPONSE_BITS_DATA_WIDTH]=cache_data;
      vif.backend[`VA_IN_CACHERESPONSE_BITS_ERROR_OFFSET+:`VA_IN_CACHERESPONSE_BITS_ERROR_WIDTH]=cache_error;
      vif.backend[`VA_IN_CACHERESPONSE_BITS_LINE_OFFSET+:`VA_IN_CACHERESPONSE_BITS_LINE_WIDTH]=cache_line;
      vif.backend[`VA_IN_UNCACHEDRESPONSE_VALID_OFFSET+:`VA_IN_UNCACHEDRESPONSE_VALID_WIDTH]=uncached_pending&&cycle>=uncached_due;
      vif.backend[`VA_IN_UNCACHEDRESPONSE_BITS_DATA_OFFSET+:`VA_IN_UNCACHEDRESPONSE_BITS_DATA_WIDTH]=uncached_data;
      vif.backend[`VA_IN_UNCACHEDRESPONSE_BITS_TAG_OFFSET+:`VA_IN_UNCACHEDRESPONSE_BITS_TAG_WIDTH]=tag;
      vif.backend[`VA_IN_UNCACHEDRESPONSE_BITS_ERROR_OFFSET+:`VA_IN_UNCACHEDRESPONSE_BITS_ERROR_WIDTH]=image.backend_error;
    end
  endtask
endclass
