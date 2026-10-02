module AccessWriteDPI (
    input clock,
    reset,
    fire,
    idle,
    input [63:0] stream_hart,
    hart,
    inst,
    input [31:0] shared,
    physical,
    bank,
    group,
    addr,
    mask,
    input [127:0] data
);
  reg  [63:0] produced = 0;
  reg  [63:0] owner = 0;
  wire [63:0] stream_owner = produced == 0 ? stream_hart : owner;
  export "DPI-C" function access_snapshot;
  function void access_snapshot(output int unsigned owner_lo, owner_hi, is_shared, producer,
                                count_lo, count_hi, is_idle);
    owner_lo  = owner[31:0];
    owner_hi  = owner[63:32];
    is_shared = shared;
    producer  = physical;
    count_lo  = produced[31:0];
    count_hi  = produced[63:32];
    is_idle   = {31'b0, idle};
  endfunction
  import "DPI-C" context function void dpi_access_write(
    input int unsigned owner_lo,
    owner_hi,
    hart_lo,
    hart_hi,
    inst_lo,
    inst_hi,
    is_shared,
    pbank,
    vbank,
    bank_group,
    seq_lo,
    seq_hi,
    row_addr,
    write_mask,
    data0,
    data1,
    data2,
    data3
  );
  always @(posedge clock) begin
    if (reset) begin
      produced <= 0;
      owner <= 0;
    end else if (fire) begin
      owner <= stream_owner;
      dpi_access_write(stream_owner[31:0], stream_owner[63:32], hart[31:0], hart[63:32], inst[31:0],
                       inst[63:32], shared, physical, bank, group, produced[31:0], produced[63:32],
                       addr, mask, data[31:0], data[63:32], data[95:64], data[127:96]);
      produced <= produced + 1;
    end
  end
endmodule
