interface chi_codec_if (
    input logic clock
);
  logic reset, pause, valid, ready, out_valid, out_ready;
  logic [6:0] target;
  logic [388:0] data, out_data;
  logic chunk_fire, chunk_head, chunk_tail, chunk_x, chunk_y;
endinterface
