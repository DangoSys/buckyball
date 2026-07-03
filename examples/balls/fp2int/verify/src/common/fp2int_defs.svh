import "DPI-C" function int fp2int_ref_i32(
  input int unsigned fp_bits,
  input int unsigned scale_bits
);
import "DPI-C" function int unsigned fp2int_case_bid(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_funct7(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_iter(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_scale_bits(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_op1_bank(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_op2_bank(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_wr_bank(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_op1_col(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_op2_col(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_wr_col(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_meta_bank(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_rob_id(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_is_sub(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function int unsigned fp2int_case_sub_rob_id(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function longint unsigned fp2int_case_word_lo(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned word_index
);
import "DPI-C" function longint unsigned fp2int_case_word_hi(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned word_index
);

localparam int FP2INT_NUM_WORDS = 4;
localparam int FP2INT_TIMEOUT_CYCLES = 200;
