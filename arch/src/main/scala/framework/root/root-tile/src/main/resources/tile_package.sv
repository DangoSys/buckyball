`include "tile_config.svh"
`include "cache_system_config.svh"
`include "multicore_fixture.svh"
`define TF(K, F) `TILE_``K``_``F``_OFFSET +: `TILE_``K``_``F``_WIDTH
`define TH(K, F) `COH_``K``_``F``_OFFSET +: `COH_``K``_``F``_WIDTH
package tile_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle core_ref_create(input string path);
  import "DPI-C" function void core_ref_destroy(input chandle model);
  import "DPI-C" function void core_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function void core_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  typedef logic [`COH_REQ_WIDTH-1:0] req_t;
  typedef logic [`COH_RSP_WIDTH-1:0] rsp_t;
  typedef logic [`COH_DAT_WIDTH-1:0] dat_t;
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual tile_if control;
    virtual stream_if #(`TILE_MEMREQ_WIDTH) mem_req;
    virtual stream_if #(`TILE_MEMRESP_WIDTH) mem_resp;
    virtual stream_if #(`TILE_UNCACHED_WIDTH) uncached_req[2];
    virtual stream_if #(`TILE_URESP_WIDTH) uncached_resp[2];
    chandle model;
    typedef struct {
      bit [`TILE_MEMRESP_WIDTH-1:0] packet;
      int due;
    } memory_entry;
    memory_entry memory_queue[$];
    bit memory_active = 0, mmio_pending[2] = '{0, 0};
    bit [`TILE_MEMRESP_WIDTH-1:0] memory_packet;
    bit [`TILE_URESP_WIDTH-1:0] mmio_packet[2];
    int mmio_due[2], mmio_wait[2] = '{0, 0}, memory_wait = 0;
    int stages[2] = '{0, 0}, retired[2] = '{0, 0}, mmio_stalls[2] = '{0, 0};
    bit exited[2] = '{0, 0}, exit_retired[2] = '{0, 0};
    int cycles = 0, reads = 0, writes = 0, memory_stalls = 0, node_requests[2] = '{0, 0};
    req_t requests[int];
    int fill_seen[int], fill_dbid[int], ack_owner[int];
    bit [511:0] fill_line[int], return_line[int];
    longint unsigned return_address[int];
    int return_owner[int], return_seen[int];
    bit return_is_snoop[int];
    int grants = 0, snoops = 0, writebacks = 0, acks = 0, dirty_snoops = 0;
    bit final_line_seen = 0;
    localparam longint unsigned SHARED = 64'h800040c0;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 5ms;
    endfunction
    function bit [511:0] final_line();
      return {
        64'hffeeddccbbaa9988,
        64'h8000000000000001,
        64'h1122334455667788,
        64'h0102030405060708,
        64'd1,
        64'd1,
        64'd1,
        64'h8877665544332211
      };
    endfunction
    function void build_phase(uvm_phase phase);
      bit [511:0] initial_line;
      super.build_phase(phase);
      if (!uvm_config_db#(virtual tile_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual stream_if #(`TILE_MEMREQ_WIDTH))::get(
              this, "", "mem_req", mem_req
          ) || !uvm_config_db#(virtual stream_if #(`TILE_MEMRESP_WIDTH))::get(
              this, "", "mem_resp", mem_resp
          ))
        `uvm_fatal("VIF", "Tile interfaces missing")
      for (int h = 0; h < 2; h++) begin
        if (!uvm_config_db#(virtual stream_if #(`TILE_UNCACHED_WIDTH))::get(
                this, "", $sformatf("uncached_req_%0d", h), uncached_req[h]
            ) || !uvm_config_db#(virtual stream_if #(`TILE_URESP_WIDTH))::get(
                this, "", $sformatf("uncached_resp_%0d", h), uncached_resp[h]
            ))
          `uvm_fatal("VIF", "Per-hart MMIO interface missing")
      end
      model = core_ref_create(`TILE_IMAGE);
      core_ref_read(model, SHARED, initial_line);
      if (initial_line !== 512'b0)
        `uvm_fatal("FIXTURE", "Shared 64-byte fixture must be explicitly initialized to zero")
    endfunction
    function void observe_protocol();
      if (control.sample.req_valid) begin
        req_t r = control.sample.req_bits;
        int node, key;
        if ($isunknown(r)) `uvm_fatal("CHI_X", "Unknown accepted request")
        node = r[`TH(REQ, SRCID)];
        key  = (node << 1) | int'(r[`TH(REQ, TXNID)]);
        if (node < 1 || node > 2 || r[
            `TH(REQ, TGTID)
            ] != 64 || r[
            `TH(REQ, TXNID)
            ] >= 2 || requests.exists(
                key
            ))
          `uvm_fatal("REQ_ID", "Requester node/bank identity or duplicate transaction")
        requests[key] = r;
        node_requests[node-1]++;
      end
      if (control.sample.rsp_valid) begin
        rsp_t r = control.sample.rsp_bits;
        int key, id;
        key = (int'(r[`TH(RSP, TGTID)]) << 1) | int'(r[`TH(RSP, TXNID)]);
        id  = r[`TH(RSP, DBID)];
        if ($isunknown(
                r
            ) || !requests.exists(
                key
            ) || r[
            `TH(RSP, SRCID)
            ] != 64 || r[
            `TH(RSP, RESPERR)
            ] != 0)
          `uvm_fatal("RSP_ID", "Unmatched Home response or error")
        if (r[`TH(RSP, OPCODE)] == `COH_COMP_DBID) begin
          if (id >= 4 || return_owner.exists(
                  id
              ) || requests[key][
              `TH(REQ, OPCODE)
              ] != `COH_WRITEBACK)
            `uvm_fatal("WB_ID", "Invalid writeback DBID/descriptor")
          return_owner[id] = r[`TH(RSP, TGTID)];
          return_address[id] = requests[key][`TH(REQ, ADDR)];
          return_seen[id] = 0;
          return_line[id] = '0;
          return_is_snoop[id] = 0;
        end else if (r[`TH(RSP, OPCODE)] != `COH_COMP)
          `uvm_fatal("RSP_OPCODE", "Unexpected retry/response")
        requests.delete(key);
      end
      if (control.sample.dat_valid) begin
        dat_t d = control.sample.dat_bits;
        int key, id, beat;
        key  = (int'(d[`TH(DAT, TGTID)]) << 1) | int'(d[`TH(DAT, TXNID)]);
        id   = d[`TH(DAT, DBID)];
        beat = int'(d[`TH(DAT, DATAID)]) >> 1;
        if ($isunknown(
                d
            ) || !requests.exists(
                key
            ) || id >= 4 || d[
            `TH(DAT, SRCID)
            ] != 64 || d[
            `TH(DAT, HOMENID)
            ] != 64 || d[
            `TH(DAT, OPCODE)
            ] != `COH_COMP_DATA || d[
            `TH(DAT, RESPERR)
            ] != 0 || d[
            `TH(DAT, BE)
            ] != 32'hffffffff || (d[
            `TH(DAT, DATAID)
            ] != 0 && d[
            `TH(DAT, DATAID)
            ] != 2))
          `uvm_fatal("COMP_DATA", "Invalid coherent grant")
        if (!fill_seen.exists(key)) begin
          fill_seen[key] = 0;
          fill_line[key] = '0;
          fill_dbid[key] = id;
        end
        if ((fill_seen[key] & (1 << beat)) || fill_dbid[key] != id)
          `uvm_fatal("COMP_BEAT", "Duplicate beat or changed DBID")
        fill_line[key][beat*256+:256] = d[`TH(DAT, DATA)];
        fill_seen[key] |= 1 << beat;
        if (fill_seen[key] == 3) begin
          if (requests[key][`TH(REQ, ADDR)] == SHARED && fill_line[key] === final_line())
            final_line_seen = 1;
          if (ack_owner.exists(id)) `uvm_fatal("ACK_SLOT", "Home DBID reused before CompAck")
          ack_owner[id] = d[`TH(DAT, TGTID)];
          grants++;
          requests.delete(key);
          fill_seen.delete(key);
          fill_dbid.delete(key);
          fill_line.delete(key);
        end
      end
      if (control.sample.snp_valid) begin
        int id = control.sample.snp_bits[`TH(SNP, TXNID)];
        int target = control.sample.snp_bits[`TH(SNP, TARGET)];
        if ($isunknown(
                control.sample.snp_bits
            ) || id >= 4 || target < 1 || target > 2 || return_owner.exists(
                id
            ) || control.sample.snp_bits[
            `TH(SNP, SRCID)
            ] != 64)
          `uvm_fatal("SNOOP_ID", "Invalid directed snoop descriptor")
        return_owner[id] = target;
        return_address[id] = 64'(control.sample.snp_bits[`TH(SNP, ADDR)]) << 3;
        return_seen[id] = 0;
        return_line[id] = '0;
        return_is_snoop[id] = 1;
      end
      if (control.sample.rx_rsp_valid) begin
        rsp_t r = control.sample.rx_rsp_bits;
        int   id = r[`TH(RSP, TXNID)];
        if ($isunknown(r) || r[`TH(RSP, TGTID)] != 64)
          `uvm_fatal("RX_RSP", "Invalid requester response")
        if (r[`TH(RSP, OPCODE)] == `COH_COMP_ACK) begin
          if (!ack_owner.exists(id) || ack_owner[id] != r[`TH(RSP, SRCID)])
            `uvm_fatal("ACK", "CompAck did not return Home DBID/owner")
          ack_owner.delete(id);
          acks++;
        end else begin
          if (r[
              `TH(RSP, OPCODE)
              ] != `COH_SNP_RESP || !return_owner.exists(
                  id
              ) || !return_is_snoop[id] || return_owner[id] != r[
              `TH(RSP, SRCID)
              ] || return_seen[id] != 0)
            `uvm_fatal("SNOOP_RSP", "Unmatched snoop response")
          return_owner.delete(id);
          return_seen.delete(id);
          return_address.delete(id);
          return_line.delete(id);
          return_is_snoop.delete(id);
        end
      end
      if (control.sample.rx_dat_valid) begin
        dat_t d = control.sample.rx_dat_bits;
        int   id = d[`TH(DAT, TXNID)];
        int   beat = int'(d[`TH(DAT, DATAID)]) >> 1;
        if ($isunknown(
                d
            ) || !return_owner.exists(
                id
            ) || return_owner[id] != d[
            `TH(DAT, SRCID)
            ] || d[
            `TH(DAT, TGTID)
            ] != 64 || d[
            `TH(DAT, RESPERR)
            ] != 0 || (d[
            `TH(DAT, DATAID)
            ] != 0 && d[
            `TH(DAT, DATAID)
            ] != 2) || (return_seen[id] & (1 << beat)) || d[
            `TH(DAT, BE)
            ] != 32'hffffffff)
          `uvm_fatal("RETURN_DATA", "Unmatched/duplicate/partial return data")
        if (d[`TH(DAT, OPCODE)] != (return_is_snoop[id] ? `COH_SNP_DATA : `COH_COPYBACK_DATA))
          `uvm_fatal("RETURN_OPCODE", "Wrong return role")
        return_line[id][beat*256+:256] = d[`TH(DAT, DATA)];
        return_seen[id] |= 1 << beat;
        if (return_seen[id] == 3) begin
          if(return_is_snoop[id]&&return_address[id]==SHARED&&return_line[id]===final_line())
            final_line_seen = 1;
          if (return_is_snoop[id]) begin
            snoops++;
            if (d[`TH(DAT, RESP)] & 4) dirty_snoops++;
          end else writebacks++;
          return_owner.delete(id);
          return_seen.delete(id);
          return_address.delete(id);
          return_line.delete(id);
          return_is_snoop.delete(id);
        end
      end
    endfunction
    task service();
      forever begin
        @(control.sample);
        if (!control.sample.reset) begin
          cycles++;
          observe_protocol();
          if (mem_resp.sample.valid && mem_resp.sample.ready) memory_active = 0;
          for (int h = 0; h < 2; h++) begin
            if (uncached_resp[h].sample.valid && uncached_resp[h].sample.ready) mmio_pending[h] = 0;
            if (control.sample.trapped[h])
              `uvm_fatal("TRAP", $sformatf(
                         "hart%0d cause=%h value=%h pc=%h",
                         h,
                         control.sample.trapCause[h],
                         control.sample.trapValue[h],
                         control.sample.trapPc[h]
                         ))
            if (control.sample.retired[h]) begin
              retired[h]++;
              if (control.sample.retiredPc[h] == `CORE_EXIT_PC) begin
                if (!exited[h] || exit_retired[h])
                  `uvm_fatal("EXIT_RETIRE", "Exit retired before MMIO success or twice")
                exit_retired[h] = 1;
              end
            end
            if (uncached_req[h].sample.valid && !uncached_req[h].sample.ready) begin
              mmio_stalls[h]++;
              mmio_wait[h]++;
            end
            if (uncached_req[h].sample.valid && uncached_req[h].sample.ready) begin
              longint unsigned address = uncached_req[h].sample.bits[`TF(UNCACHED, ADDR)];
              longint unsigned value = uncached_req[h].sample.bits[`TF(UNCACHED, DATA)];
              if (mmio_pending[h] || exited[h] || $isunknown(
                      uncached_req[h].sample.bits
                  ) || !uncached_req[h].sample.bits[
                  `TF(UNCACHED, WRITE)
                  ] || uncached_req[h].sample.bits[
                  `TF(UNCACHED, SIZE)
                  ] != 3 || (address != 64'h10000000 && address != 64'h10000008))
                `uvm_fatal("MMIO", "Unexpected per-hart MMIO operation")
              if (address == 64'h10000008) begin
                if (value != stages[h] + 1 || value > 2)
                  `uvm_fatal("STAGE", "Per-hart stage out of order")
                stages[h] = value;
              end else begin
                if (value != 0 || stages[h] != 2)
                  `uvm_fatal("FIRMWARE", $sformatf(
                             "hart%0d failed stage%0d code%h", h, stages[h], value))
                exited[h] = 1;
              end
              mmio_packet[h] = '0;
              mmio_packet[h][`TF(URESP, TAG)] = uncached_req[h].sample.bits[`TF(UNCACHED, TAG)];
              mmio_pending[h] = 1;
              mmio_due[h] = cycles + 5 + h;
              mmio_wait[h] = 0;
            end
          end
          if (mem_req.sample.valid && !mem_req.sample.ready) begin
            memory_stalls++;
            memory_wait++;
          end
          if (mem_req.sample.valid && mem_req.sample.ready) begin
            memory_entry e;
            bit [511:0] line;
            longint unsigned address = mem_req.sample.bits[`TF(MEMREQ, ADDR)];
            if ($isunknown(mem_req.sample.bits) || address[5:0] != 0)
              `uvm_fatal("MEMORY", "Unknown/unaligned backend request")
            e.packet = '0;
            e.packet[`TF(MEMRESP, ID)] = mem_req.sample.bits[`TF(MEMREQ, ID)];
            if (mem_req.sample.bits[`TF(MEMREQ, WRITE)]) begin
              core_ref_write(model, address, mem_req.sample.bits[`TF(MEMREQ, DATA)],
                             mem_req.sample.bits[`TF(MEMREQ, MASK)]);
              writes++;
            end else begin
              core_ref_read(model, address, line);
              e.packet[`TF(MEMRESP, DATA)] = line;
              reads++;
            end
            e.due = cycles + 5 + (reads + writes) % 7;
            memory_queue.push_back(e);
            memory_wait = 0;
          end
        end
        @(negedge control.clock);
        mem_req.ready = !control.reset && memory_wait >= 2;
        for (int h = 0; h < 2; h++) begin
          uncached_req[h].ready  = !control.reset && !mmio_pending[h] && mmio_wait[h] >= 2;
          uncached_resp[h].valid = !control.reset && mmio_pending[h] && cycles >= mmio_due[h];
          uncached_resp[h].bits  = mmio_packet[h];
        end
        if (!memory_active)
          for (int q = memory_queue.size() - 1; q >= 0; q--)
          if (memory_queue[q].due <= cycles) begin
            memory_packet = memory_queue[q].packet;
            memory_queue.delete(q);
            memory_active = 1;
            break;
          end
        mem_resp.valid = !control.reset && memory_active;
        mem_resp.bits  = memory_packet;
      end
    endtask
    task execute();
      control.reset  = 1;
      mem_req.ready  = 0;
      mem_resp.valid = 0;
      mem_resp.bits  = '0;
      for (int h = 0; h < 2; h++) begin
        uncached_req[h].ready  = 0;
        uncached_resp[h].valid = 0;
        uncached_resp[h].bits  = '0;
      end
      repeat (5) @(negedge control.clock);
      control.reset = 0;
      fork
        service();
      join_none
      wait (exit_retired[0] && exit_retired[1]);
      do
        @(control.sample);
      while (control.sample.outstanding || memory_active || memory_queue.size() ||
             mmio_pending[0] || mmio_pending[1] || requests.num() || fill_seen.num() ||
             ack_owner.num() || return_owner.num());
      if(!final_line_seen||!dirty_snoops||!node_requests[0]||!node_requests[1]||!memory_stalls||!mmio_stalls[0]||!mmio_stalls[1])
        `uvm_fatal(
            "SCENARIOS",
            "Missing final full64B coherent line, dirty snoop, distinct nodes or backpressure")
      `uvm_info(
          "TILE",
          $sformatf(
              "2 real harts stages=2/2 exits=0/0 retired=%0d/%0d; CHI req=%0d/%0d grants=%0d snoopLines=%0d dirty=%0d copybacks=%0d acks=%0d; DDR R/W=%0d/%0d stalls=%0d MMIO stalls=%0d/%0d; final64B checked and all descriptors drained",
              retired[0], retired[1], node_requests[0], node_requests[1], grants, snoops,
              dirty_snoops, writebacks, acks, reads, writes, memory_stalls, mmio_stalls[0],
              mmio_stalls[1]), UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      core_ref_destroy(model);
      super.final_phase(phase);
    endfunction
  endclass
endpackage
