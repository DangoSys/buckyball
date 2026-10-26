module RedmuleNative (
    input logic clk_i,
    rst_ni,
    output logic busy_o,
    evt_o,
    input logic ctrl_req_i,
    ctrl_wen_i,
    input logic [31:0] ctrl_addr_i,
    ctrl_data_i,
    output logic ctrl_gnt_o,
    ctrl_valid_o,
    output logic [31:0] ctrl_data_o,
    output logic mem_req_o,
    mem_wen_o,
    output logic [31:0] mem_addr_o,
    mem_be_o,
    output logic [255:0] mem_data_o,
    output logic [7:0] mem_id_o,
    input logic mem_gnt_i,
    mem_valid_i,
    input logic [255:0] mem_data_i,
    input logic [7:0] mem_id_i,
    output logic mem_ready_o
);
  localparam hci_package::hci_size_parameter_t HCI_SIZE = '{
      DW: 256,
      AW: 32,
      BW: 8,
      UW: 1,
      IW: 8,
      EW: 1,
      EHW: 1
  };
  hci_core_intf #(
      .DW (256),
      .AW (32),
      .BW (8),
      .UW (1),
      .IW (8),
      .EW (1),
      .EHW(1)
  ) memory (
      clk_i
  );
  hwpe_ctrl_intf_periph #(.ID_WIDTH(4)) control (clk_i);
  hwpe_stream_intf_stream #(.DATA_WIDTH(256))
      x_in (clk_i), w_in (clk_i), x_out (clk_i), w_out (clk_i);
  assign x_in.valid = 0;
  assign x_in.data = 0;
  assign x_in.strb = 0;
  assign w_in.valid = 0;
  assign w_in.data = 0;
  assign w_in.strb = 0;
  assign x_out.ready = 1;
  assign w_out.ready = 1;
  assign control.req = ctrl_req_i;
  assign control.wen = ctrl_wen_i;
  assign control.add = ctrl_addr_i;
  assign control.data = ctrl_data_i;
  assign control.be = 4'hf;
  assign control.id = 0;
  assign ctrl_gnt_o = control.gnt;
  assign ctrl_valid_o = control.r_valid;
  assign ctrl_data_o = control.r_data;
  assign mem_req_o = memory.req;
  assign mem_wen_o = memory.wen;
  assign mem_addr_o = memory.add;
  assign mem_be_o = memory.be;
  assign mem_data_o = memory.data;
  assign mem_id_o = memory.id;
  assign mem_ready_o = memory.r_ready;
  assign memory.gnt = mem_gnt_i;
  assign memory.r_valid = mem_valid_i;
  assign memory.r_data = mem_data_i;
  assign memory.r_id = mem_id_i;
  assign memory.r_user = 0;
  assign memory.r_opc = 0;
  assign memory.r_ecc = 0;
  assign memory.egnt = {1{mem_gnt_i}};
  assign memory.r_evalid = {1{mem_valid_i}};
  redmule_top #(
      .DataW(256),
      .Height(8),
      .Width(8),
      .NumPipeRegs(1),
      .FpFormat(fpnew_pkg::FP16),
      .PipeConfig(fpnew_pkg::DISTRIBUTED),
      .CtrlIntfConfig(redmule_pkg::HWPE_TARGET),
      .HCI_SIZE_tcdm(HCI_SIZE)
  ) native_i (
      .clk_i,
      .rst_ni,
      .test_mode_i(1'b0),
      .busy_o,
      .evt_o,
      .w_stream_i(w_in),
      .x_stream_i(x_in),
      .w_stream_o(w_out),
      .x_stream_o(x_out),
      .x_issue_req_i(1'b0),
      .x_issue_resp_o(),
      .x_issue_valid_i(1'b0),
      .x_issue_ready_o(),
      .x_register_i(1'b0),
      .x_register_valid_i(1'b0),
      .x_register_ready_o(),
      .x_commit_i(1'b0),
      .x_commit_valid_i(1'b0),
      .x_result_o(),
      .x_result_valid_o(),
      .x_result_ready_i(1'b0),
      .sync_o(),
      .sync_i(1'b0),
      .tcdm(memory),
      .target(control)
  );
endmodule
