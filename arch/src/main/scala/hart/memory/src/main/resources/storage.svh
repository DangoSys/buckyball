`define CHANNEL(
    c) '{`DF_OUT_STORAGE_``c``_REQUEST_VALID_OFFSET,`DF_IN_STORAGE_``c``_REQUEST_READY_OFFSET,`DF_OUT_STORAGE_``c``_REQUEST_BITS_TDEST_OFFSET,`DF_OUT_STORAGE_``c``_REQUEST_BITS_TUSER_OFFSET,`DF_OUT_STORAGE_``c``_REQUEST_BITS_TID_OFFSET,`DF_OUT_STORAGE_``c``_REQUEST_BITS_TDATA_OFFSET,`DF_OUT_STORAGE_``c``_REQUEST_BITS_TKEEP_OFFSET,`DF_IN_STORAGE_``c``_RESPONSE_VALID_OFFSET,`DF_OUT_STORAGE_``c``_RESPONSE_READY_OFFSET,`DF_IN_STORAGE_``c``_RESPONSE_BITS_TID_OFFSET,`DF_IN_STORAGE_``c``_RESPONSE_BITS_TDATA_OFFSET,`DF_IN_STORAGE_``c``_RESPONSE_BITS_TUSER_OFFSET}
const int channel[4][12] = '{`CHANNEL(0), `CHANNEL(1), `CHANNEL(2), `CHANNEL(3)};
bit pending[4];
bit [127:0] response_data[4];
bit [7:0] response_tag[4];
bit response_write[4];
logic [127:0] memory[`DF_BANKS][1024];
int local_reads, local_writes, returned_beats;
bit [127:0] observed_data[8];
task tick(bit resetting = 0);
  for (int c = 0; c < 4; c++) begin
    put(channel[c][1], 1, !pending[c] && cycle % 3 != 1);
    put(channel[c][7], 1, pending[c]);
    put(channel[c][9], `DF_IN_STORAGE_0_RESPONSE_BITS_TID_WIDTH, response_tag[c]);
    put(channel[c][10], 128, response_data[c]);
    put(channel[c][11], 2, {response_write[c], 1'b0});
  end
  @(negedge vif.clock);
  vif.inputs = pins;
  vif.reset  = resetting;
  @(posedge vif.clock);
  sampled = vif.outputs;
  cycle++;
  if (resetting) begin
    foreach (pending[c]) pending[c] = 0;
  end else begin
    for (int c = 0; c < 4; c++) begin
      if (pending[c] && get(channel[c][8], 1)) pending[c] = 0;
      if (get(channel[c][0], 1) && pins[channel[c][1]]) begin
        int bank = int'(get(channel[c][2], `DF_OUT_STORAGE_0_REQUEST_BITS_TDEST_WIDTH));
        int row = int'(get(channel[c][3], 17)) >> 1;
        bit write_bit = get(channel[c][3], 1);
        evidence::check(bank < `DF_BANKS && row < 1024, "Storage address out of range");
        pending[c] = 1;
        response_tag[c] = get(channel[c][4], `DF_OUT_STORAGE_0_REQUEST_BITS_TID_WIDTH);
        response_write[c] = write_bit;
        if (write_bit) begin
          evidence::check(get(channel[c][6], 16) == 16'hffff, "Partial local row");
          memory[bank][row] = get(channel[c][5], 128);
          local_writes++;
        end else begin
          response_data[c] = memory[bank][row];
          local_reads++;
        end
      end
    end
    if (`GET(THREADS_0_READDATA_VALID)) begin
      observed_data[returned_beats++] = `GET(THREADS_0_READDATA_BITS_TDATA);
      evidence::check(`GET(THREADS_0_READDATA_BITS_TUSER) == 0, "DMA read error");
    end
    if (`GET(THREADS_1_READDATA_VALID)) begin
      observed_data[returned_beats++] = `GET(THREADS_1_READDATA_BITS_TDATA);
      evidence::check(`GET(THREADS_1_READDATA_BITS_TUSER) == 0, "DMA read error");
    end
  end
  `PUT(THREADS_0_READDATA_CREDIT, `GET(THREADS_0_READDATA_VALID) && !resetting);
  `PUT(THREADS_1_READDATA_CREDIT, `GET(THREADS_1_READDATA_VALID) && !resetting);
  #1;
endtask
