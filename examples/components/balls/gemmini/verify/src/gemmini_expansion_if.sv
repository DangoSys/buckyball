// SubROB expansion port of GemminiBall: loop configuration and loop triggers.
interface gemmini_expansion_if;
  logic                    command_ready;
  logic                    command_valid;
  logic [`BB_ROB_ID_W-1:0] command_bits_rob_id;
  logic [             6:0] command_bits_funct;
  logic [            63:0] command_bits_rs2;
  logic                    accepts;
  logic                    starts;
  logic                    complete_ready;
  logic                    complete_valid;
  logic [`BB_ROB_ID_W-1:0] complete_bits_rob_id;
  logic                    write_ready;
  logic                    write_valid;
  logic [             9:0] write_bits_addr;
  logic [             7:0] write_bits_data;
endinterface
