// Decoder classification of arbitrary encodings, including every illegal form and other
// CSR accesses. The instruction is fetched, then the task is cancelled in the cycle it
// would execute, so the decoded class is compared with the reference without trapping.
// The executor state and decoder output are observed hierarchically for this race only.
task automatic decode_race(int count);
  int unsigned opcodes[14] = '{
      'h37,
      'h17,
      'h6f,
      'h67,
      'h63,
      'h03,
      'h23,
      'h13,
      'h1b,
      'h33,
      'h3b,
      'h7b,
      'h0f,
      'h73
  };
  for (int n = 0; n < count; n++) begin
    int c = n % 2;
    logic [31:0] instruction = $urandom;
    logic [3:0] kind;
    case ($urandom_range(
        0, 7
    ))
      0: ;
      1: instruction[6:0] = opcodes[$urandom_range(0, 13)];
      2, 3: begin
        // Funct7 values around the defined ones.
        int f7s[4] = '{0, 1, 'h20, 'h21};
        instruction[6:0]   = opcodes[$urandom_range(7, 10)];
        instruction[31:25] = f7s[$urandom_range(0, 3)];
      end
      4, 5: begin
        // SYSTEM: ECALL, EBREAK, csrr mcycle and neighbouring CSR encodings.
        int unsigned forms[6] = '{
            'h00000073,
            'h00100073,
            'hb0002073,
            'hb0202073,
            'hc0002073,
            'h30002073
        };
        instruction = forms[$urandom_range(0, 5)];
        if ($urandom_range(0, 1)) instruction[11:7] = $urandom;
        if ($urandom_range(0, 3) == 0) instruction[14:12] = $urandom;
        if ($urandom_range(0, 3) == 0) instruction[19:15] = $urandom;
      end
      default: instruction = {$urandom_range(0, 1) ? 7'h00 : 7'h20, instruction[24:7], 7'h1b};
    endcase
    put(c * 2, 0, {64'b0, 32'h00000073, instruction}, 1);
    entry[c] = 0;
    code_end[c] = 8;
    argument[c] = `TLS_BASE;
    stack[c] = `TLS_END;
    in_use = 1;
    launch(c);
    do begin
      @(negedge clock);
      kind = c ? dut.ants_1.executor.decoder.io_decoded_kind : dut.ants_0.executor.decoder.io_decoded_kind;
    end while ((c ? dut.ants_1.executor.state : dut.ants_0.executor.state) != 4'h3);
    cancel[c] = 1;
    evidence::check(kind == ant_decode_kind(instruction), $sformatf(
                    "instruction %h decoded as %0d, reference %0d",
                    instruction,
                    kind,
                    ant_decode_kind(
                        instruction
                    )
                    ));
    @(negedge clock);
    cancel[c] = 0;
    finish(c, 0, 1);
    in_use = 0;
  end
endtask
