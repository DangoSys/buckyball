// RV64IM program generator shared by the Ant and service benches. Forward control flow only:
// every branch or jump lands on a later instruction group, and the final group is
// `li a7, 0; ecall`. Loads and stores stay inside the given TLS and TSS windows.
function automatic int unsigned itype(int imm, int rs1, int f3, int rd, int opcode);
  return ((imm & 'hfff) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode;
endfunction

function automatic int unsigned rtype(int f7, int rs2, int rs1, int f3, int rd, int opcode);
  return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode;
endfunction

function automatic int unsigned stype(int imm, int rs2, int rs1, int f3);
  return (((imm >> 5) & 'h7f) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 31) << 7) |
      'h23;
endfunction

function automatic int unsigned btype(int imm);
  return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 'h3f) << 25) | (((imm >> 1) & 'hf) << 8) |
      (((imm >> 11) & 1) << 7);
endfunction

function automatic int unsigned jtype(int imm);
  return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 'h3ff) << 21) | (((imm >> 11) & 1) << 20) |
      (((imm >> 12) & 'hff) << 12);
endfunction

function automatic void generate_program(int groups, longint unsigned tls_window,
                                         longint unsigned tss_window, ref int unsigned code[$]);
  int start[$], at[$], target[$];
  code = {};
  for (int g = 0; g < groups; g++) begin
    int choice = $urandom_range(0, 99);
    int rd = $urandom_range(0, 31), rs1 = $urandom_range(0, 31), rs2 = $urandom_range(0, 31);
    start.push_back(code.size());
    if (choice < 24) begin
      // Load or store through a fresh base register; the address stays in a window.
      bit store = $urandom_range(0, 2) == 0;
      int f3 = store ? $urandom_range(0, 3) : $urandom_range(0, 6);
      int bytes = 1 << (f3 & 3);
      longint unsigned address = ($urandom_range(
          0, 1
      ) ? tls_window : tss_window) + ($urandom_range(
          0, 1023
      ) & ~(bytes - 1));
      int imm = ($urandom_range(0, 511) - 256) & ~(bytes - 1);
      longint unsigned base = address - longint'(imm);
      int high = (base + 'h800) >> 12;
      int low = base - (high << 12);
      int rb = $urandom_range(1, 31);
      code.push_back((high << 12) | (rb << 7) | 'h37);
      code.push_back(itype(low, rb, 0, rb, 'h13));
      code.push_back(store ? stype(imm, rs2, rb, f3) : itype(imm, rb, f3, rd, 'h03));
    end else if (choice < 38) begin
      int forms[13][3] = '{
          '{0, 0, 'h33},
          '{32, 0, 'h33},
          '{0, 1, 'h33},
          '{0, 2, 'h33},
          '{0, 3, 'h33},
          '{0, 4, 'h33},
          '{0, 5, 'h33},
          '{32, 5, 'h33},
          '{0, 6, 'h33},
          '{0, 7, 'h33},
          '{0, 0, 'h3b},
          '{32, 0, 'h3b},
          '{0, 1, 'h3b}
      };
      int form = $urandom_range(0, 14);
      if (form == 13) code.push_back(rtype(0, rs2, rs1, 5, rd, 'h3b));
      else if (form == 14) code.push_back(rtype(32, rs2, rs1, 5, rd, 'h3b));
      else code.push_back(rtype(forms[form][0], rs2, rs1, forms[form][1], rd, forms[form][2]));
    end else if (choice < 52) begin
      int f3 = $urandom_range(0, 7);
      int imm = $urandom;
      if ($urandom_range(0, 3) == 0) begin
        // Word immediates: ADDIW, SLLIW, SRLIW, SRAIW.
        int forms[4] = '{0, 1, 5, 5};
        int form = $urandom_range(0, 3);
        imm = form == 0 ? imm : (form == 3 ? 'h400 : 0) | $urandom_range(0, 31);
        code.push_back(itype(imm, rs1, forms[form], rd, 'h1b));
      end else begin
        if (f3 == 1) imm = $urandom_range(0, 63);
        if (f3 == 5) imm = ($urandom_range(0, 1) ? 'h400 : 0) | $urandom_range(0, 63);
        code.push_back(itype(imm, rs1, f3, rd, 'h13));
      end
    end else if (choice < 62) begin
      // M extension, including the word forms.
      int word = $urandom_range(0, 3) == 0;
      int f3 = word ? (($urandom_range(
          0, 4
      ) == 0) ? 0 : $urandom_range(
          4, 7
      )) : $urandom_range(
          0, 7
      );
      code.push_back(rtype(1, rs2, rs1, f3, rd, word ? 'h3b : 'h33));
    end else if (choice < 67) begin
      // Register corner values for the multiplier and divider.
      case ($urandom_range(
          0, 4
      ))
        0: code.push_back(itype(0, 0, 0, rd, 'h13));
        1: code.push_back(itype(1, 0, 0, rd, 'h13));
        2: code.push_back(itype(-1, 0, 0, rd, 'h13));
        3: code.push_back((32'h80000 << 12) | (rd << 7) | 'h37);
        4: begin
          code.push_back(itype(1, 0, 0, rd, 'h13));
          code.push_back(itype(63, rd, 1, rd, 'h13));
        end
      endcase
    end else if (choice < 71) begin
      code.push_back(($urandom & 'hfffff000) | (rd << 7) | ($urandom_range(0, 1) ? 'h37 : 'h17));
    end else if (choice < 80) begin
      int f3s[6] = '{0, 1, 4, 5, 6, 7};
      at.push_back(code.size());
      target.push_back(g + $urandom_range(1, 6));
      code.push_back(
          rtype(0, $urandom_range(0, 3) == 0 ? rs1 : rs2, rs1, f3s[$urandom_range(0, 5)], 0, 'h63));
    end else if (choice < 83) begin
      at.push_back(code.size());
      target.push_back(g + $urandom_range(1, 6));
      code.push_back((rd << 7) | 'h6f);
    end else if (choice < 86) begin
      // AUIPC then JALR; bit 0 of the offset is cleared by JALR.
      int rb = $urandom_range(1, 31);
      code.push_back((rb << 7) | 'h17);
      at.push_back(code.size());
      target.push_back(g + $urandom_range(1, 6));
      code.push_back(itype($urandom_range(0, 1), rb, 0, rd, 'h67));
    end else if (choice < 91) begin
      code.push_back(rtype($urandom_range(0, 127), rs2, rs1, $urandom_range(0, 3), rd, 'h7b));
    end else if (choice < 94) begin
      code.push_back(itype($urandom, $urandom_range(0, 31), 0, $urandom_range(0, 31), 'h0f));
    end else begin
      code.push_back(itype('hb00, 0, 2, rd, 'h73));
    end
  end
  start.push_back(code.size());
  code.push_back(itype(0, 0, 0, 17, 'h13));
  code.push_back('h73);
  foreach (at[i]) begin
    int t = target[i] > groups ? groups : target[i];
    int offset = 4 * (start[t] - at[i]);
    case (code[at[i]] & 'h7f)
      'h63: code[at[i]] |= btype(offset);
      'h6f: code[at[i]] |= jtype(offset);
      'h67: code[at[i]] += (offset + 4) << 20;
    endcase
  end
endfunction
