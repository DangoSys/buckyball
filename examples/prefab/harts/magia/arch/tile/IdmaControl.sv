/*
 * Copyright (C) 2023-2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Solderpad Hardware License, Version 0.51
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: SHL-0.51
 *
 * Authors: Luca Balboni <luca.balboni10@studio.unibo.it>
  *          Based on idma_ctrl by Victor Isachi
 *
 * iDMA Memory-Mapped Controller
 *
 * This module provides memory-mapped control interface for iDMA transfers
 * with interrupt support. It wraps both AXI2OBI and OBI2AXI transfer channels
 * along with the memory-mapped bridge, providing equivalent functionality
 * to idma_ctrl but using memory-mapped register access instead of ISA extensions.
 */

module idma_ctrl_mm
  import magia_tile_pkg::*;
  import idma_pkg::*;
#(
    parameter int unsigned ERROR_CAP = 3,
    parameter type obi_req_t = magia_tile_pkg::core_obi_data_req_t,
    parameter type obi_rsp_t = magia_tile_pkg::core_obi_data_rsp_t,
    parameter type idma_fe_reg_req_t = magia_tile_pkg::idma_fe_reg_req_t,
    parameter type idma_fe_reg_rsp_t = magia_tile_pkg::idma_fe_reg_rsp_t,
    parameter type axi_req_t = magia_tile_pkg::idma_axi_req_t,
    parameter type axi_rsp_t = magia_tile_pkg::idma_axi_rsp_t,
    parameter type idma_obi_req_t = magia_tile_pkg::idma_obi_req_t,
    parameter type idma_obi_rsp_t = magia_tile_pkg::idma_obi_rsp_t
) (
    input logic clk_i,
    input logic rst_ni,
    input logic test_en_i,
    input logic clear_i,

    // OBI Slave Interface (CPU memory-mapped access)
    input  obi_req_t obi_req_i,
    output obi_rsp_t obi_rsp_o,

    // AXI Master Interfaces (to L2 memory)
    output axi_req_t axi_read_req_o,   // AXI2OBI: L2 read
    input  axi_rsp_t axi_read_rsp_i,
    output axi_req_t axi_write_req_o,  // OBI2AXI: L2 write
    input  axi_rsp_t axi_write_rsp_i,

    // OBI Master Interfaces (to L1 memory)
    output idma_obi_req_t obi_read_req_o,   // OBI2AXI: L1 read
    input  idma_obi_rsp_t obi_read_rsp_i,
    output idma_obi_req_t obi_write_req_o,  // AXI2OBI: L1 write
    input  idma_obi_rsp_t obi_write_rsp_i,

    // Serialized IRQ outputs
    output logic irq_a2o_busy_o,
    output logic irq_a2o_start_o,
    output logic irq_a2o_done_o,
    output logic irq_a2o_error_o,
    output logic irq_o2a_busy_o,
    output logic irq_o2a_start_o,
    output logic irq_o2a_done_o,
    output logic irq_o2a_error_o
);

  /*******************************************************/
  /**       Internal Signal Definitions Beginning       **/
  /*******************************************************/

  // Internal signals for transfer channel connections
  idma_fe_reg_req_t idma_fe_reg_axi2obi_req;
  idma_fe_reg_rsp_t idma_fe_reg_axi2obi_rsp;
  idma_fe_reg_req_t idma_fe_reg_obi2axi_req;
  idma_fe_reg_rsp_t idma_fe_reg_obi2axi_rsp;

  // Direct transfer channel IRQ signals (used for IRQ logic)
  logic a2o_transfer_busy;
  logic a2o_transfer_start;
  logic a2o_transfer_done;
  logic a2o_transfer_error;
  logic o2a_transfer_busy;
  logic o2a_transfer_start;
  logic o2a_transfer_done;
  logic o2a_transfer_error;

  /*******************************************************/
  /**        Transfer Channels Instantiation            **/
  /*******************************************************/

  // AXI2OBI Transfer Channel (L2 to L1)
  idma_axi_obi_transfer_ch #(
      .CHANNEL_T        (magia_tile_pkg::AXI2OBI),
      .ERROR_CAP        (idma_pkg::error_cap_e'(ERROR_CAP)),
      .idma_fe_reg_req_t(magia_tile_pkg::idma_fe_reg_req_t),
      .idma_fe_reg_rsp_t(magia_tile_pkg::idma_fe_reg_rsp_t),
      .axi_req_t        (magia_tile_pkg::idma_axi_req_t),
      .axi_rsp_t        (magia_tile_pkg::idma_axi_rsp_t),
      .obi_req_t        (magia_tile_pkg::idma_obi_req_t),
      .obi_rsp_t        (magia_tile_pkg::idma_obi_rsp_t)
  ) i_l2_to_l1_ch (
      .clk_i           (clk_i),
      .rst_ni          (rst_ni),
      .testmode_i      (test_en_i),
      .clear_i         (clear_i),
      .cfg_req_i       (idma_fe_reg_axi2obi_req),
      .cfg_rsp_o       (idma_fe_reg_axi2obi_rsp),
      .axi_req_o       (axi_read_req_o),
      .axi_rsp_i       (axi_read_rsp_i),
      .obi_req_o       (obi_write_req_o),
      .obi_rsp_i       (obi_write_rsp_i),
      .transfer_busy_o (a2o_transfer_busy),
      .transfer_start_o(a2o_transfer_start),
      .transfer_done_o (a2o_transfer_done),
      .transfer_error_o(a2o_transfer_error)
  );

  idma_obi_req_t obi_read_req_converter;
  idma_obi_rsp_t obi_read_rsp_converter;

  // OBI2AXI Transfer Channel (L1 to L2)
  idma_axi_obi_transfer_ch #(
      .CHANNEL_T        (magia_tile_pkg::OBI2AXI),
      .ERROR_CAP        (idma_pkg::error_cap_e'(ERROR_CAP)),
      .idma_fe_reg_req_t(magia_tile_pkg::idma_fe_reg_req_t),
      .idma_fe_reg_rsp_t(magia_tile_pkg::idma_fe_reg_rsp_t),
      .axi_req_t        (magia_tile_pkg::idma_axi_req_t),
      .axi_rsp_t        (magia_tile_pkg::idma_axi_rsp_t),
      .obi_req_t        (magia_tile_pkg::idma_obi_req_t),
      .obi_rsp_t        (magia_tile_pkg::idma_obi_rsp_t)
  ) i_l1_to_l2_ch (
      .clk_i           (clk_i),
      .rst_ni          (rst_ni),
      .testmode_i      (test_en_i),
      .clear_i         (clear_i),
      .cfg_req_i       (idma_fe_reg_obi2axi_req),
      .cfg_rsp_o       (idma_fe_reg_obi2axi_rsp),
      .axi_req_o       (axi_write_req_o),
      .axi_rsp_i       (axi_write_rsp_i),
      .obi_req_o       (obi_read_req_converter),
      .obi_rsp_i       (obi_read_rsp_converter),
      .transfer_busy_o (o2a_transfer_busy),
      .transfer_start_o(o2a_transfer_start),
      .transfer_done_o (o2a_transfer_done),
      .transfer_error_o(o2a_transfer_error)
  );


  obi_rready_converter #(
      .obi_a_chan_t(idma_obi_a_chan_t),
      .obi_r_chan_t(idma_obi_r_chan_t),
      .Depth(1)
  ) obi_rready_converter_read_i (
      .clk_i       (clk_i),
      .rst_ni      (rst_ni),
      .test_mode_i (test_en_i),
      .sbr_a_chan_i(obi_read_req_converter.a),
      .req_i       (obi_read_req_converter.req),
      .gnt_o       (obi_read_rsp_converter.gnt),
      .rready_i    (obi_read_req_converter.rready),
      .sbr_r_chan_o(obi_read_rsp_converter.r),
      .rvalid_o    (obi_read_rsp_converter.rvalid),
      .mgr_a_chan_o(obi_read_req_o.a),
      .req_o       (obi_read_req_o.req),
      .mgr_r_chan_i(obi_read_rsp_i.r),
      .gnt_i       (obi_read_rsp_i.gnt),
      .rvalid_i    (obi_read_rsp_i.rvalid)
  );
  // We are always ready for responses, because we don't
  // send more requests than we can absorb in the fifo
  assign obi_read_req_o.rready = 1'b1;


  /*******************************************************/
  /**     Memory-Mapped Bridge with IRQ Serialization   **/
  /*******************************************************/

  idma_obi_ctrl_decoder i_idma_obi_ctrl_decoder (
      .obi_req_i(obi_req_i),
      .obi_rsp_o(obi_rsp_o),

      .idma_axi2obi_req_o(idma_fe_reg_axi2obi_req),
      .idma_axi2obi_rsp_i(idma_fe_reg_axi2obi_rsp),
      .idma_obi2axi_req_o(idma_fe_reg_obi2axi_req),
      .idma_obi2axi_rsp_i(idma_fe_reg_obi2axi_rsp)
  );


  // Clean IRQ pass-through logic - equivalent to idma_ctrl behavior
  assign irq_a2o_start_o = a2o_transfer_start;
  assign irq_a2o_busy_o  = a2o_transfer_busy;
  assign irq_a2o_done_o  = a2o_transfer_done;
  assign irq_a2o_error_o = a2o_transfer_error;

  assign irq_o2a_start_o = o2a_transfer_start;
  assign irq_o2a_busy_o  = o2a_transfer_busy;
  assign irq_o2a_done_o  = o2a_transfer_done;
  assign irq_o2a_error_o = o2a_transfer_error;

  /*******************************************************/
  /**              Simple IRQ Logic End                **/
  /*******************************************************/

endmodule : idma_ctrl_mm
