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
 * Authors: Alessandro Nadalini <alessandro.nadalini3@unibo.it>
 *          Victor Isachi <victor.isachi@unibo.it>
 *
 * MAGIA NoC Package
 */

`include "axi/typedef.svh"
`include "floo_noc/typedef.svh"

package magia_noc_pkg;
  import floo_pkg::*;
  import magia_tile_pkg::*;
  import magia_pkg::*;

  typedef logic [0:0] rob_idx_t;
  typedef logic [0:0] port_id_t;
  typedef logic [$clog2(N_TILES_X+1)-1:0] x_bits_t;
  typedef logic [$clog2(N_TILES_Y)-1:0] y_bits_t;
  typedef struct packed {
    x_bits_t  x;
    y_bits_t  y;
    port_id_t port_id;
  } id_t;

  typedef logic route_t;

  localparam int unsigned SamNumRules = N_TILES + N_TILES_Y;

  typedef struct packed {
    id_t idx;
    logic [ADDR_W-1:0] start_addr;
    logic [ADDR_W-1:0] end_addr;
  } sam_rule_t;

  typedef struct packed {
    int unsigned offset;
    int unsigned len;
    int unsigned base_id;
  } collective_mask_sel_t;

  typedef struct packed {
    id_t id;
    collective_mask_sel_t mask_x;
    collective_mask_sel_t mask_y;
  } collective_idx_t;

  typedef struct packed {
    collective_idx_t idx;
    logic [31:0] start_addr;
    logic [31:0] end_addr;
  } collective_sam_rule_t;


  /**********************************
    **          ROUTING TABLE        **
    **********************************/
  // Helper function to get the correct routing table
  function automatic sam_rule_t [SamNumRules-1:0] gen_sam_table(int unsigned num_tiles);
    sam_rule_t [SamNumRules-1:0] Sam;

    case (num_tiles)
      32'd4: begin
        Sam = floo_axi_nw_mesh_2x2_noc_pkg::Sam;
      end
      32'd16: begin
        Sam = floo_axi_nw_mesh_4x4_noc_pkg::Sam;
      end
      32'd64: begin
        Sam = floo_axi_nw_mesh_8x8_noc_pkg::Sam;
      end
      32'd256: begin
        Sam = floo_axi_nw_mesh_16x16_noc_pkg::Sam;
      end
      32'd1024: begin
        Sam = floo_axi_nw_mesh_32x32_noc_pkg::Sam;
      end
    endcase

    return Sam;
  endfunction

  // Helper function to get the correct routing table
  function automatic collective_sam_rule_t [SamNumRules-1:0] gen_collective_sam_table(
      int unsigned num_tiles);
    collective_sam_rule_t [SamNumRules-1:0] CollectiveSam;

    case (num_tiles)
      32'd4: begin
        CollectiveSam = floo_axi_nw_mesh_2x2_noc_pkg::CollectiveSam;
      end
      32'd16: begin
        CollectiveSam = floo_axi_nw_mesh_4x4_noc_pkg::CollectiveSam;
      end
      32'd64: begin
        CollectiveSam = floo_axi_nw_mesh_8x8_noc_pkg::CollectiveSam;
      end
      32'd256: begin
        CollectiveSam = floo_axi_nw_mesh_16x16_noc_pkg::CollectiveSam;
      end
      32'd1024: begin
        CollectiveSam = floo_axi_nw_mesh_32x32_noc_pkg::Sam;
      end
    endcase

    return CollectiveSam;
  endfunction

  // Actual declaration of the Sam rules
  localparam sam_rule_t [SamNumRules-1:0] Sam = gen_sam_table(N_TILES);
  localparam collective_sam_rule_t [SamNumRules-1:0] CollectiveSam = gen_collective_sam_table(
      N_TILES
  );

  // Helper function to get the correct route configuration
  function automatic route_cfg_t gen_route_config(int unsigned num_tiles);
    route_cfg_t RouteCfg;

    case (num_tiles)
      32'd4: begin
        RouteCfg = floo_axi_nw_mesh_2x2_noc_pkg::RouteCfg;
      end
      32'd16: begin
        RouteCfg = floo_axi_nw_mesh_4x4_noc_pkg::RouteCfg;
      end
      32'd64: begin
        RouteCfg = floo_axi_nw_mesh_8x8_noc_pkg::RouteCfg;
      end
      32'd256: begin
        RouteCfg = floo_axi_nw_mesh_16x16_noc_pkg::RouteCfg;
      end
      32'd1024: begin
        RouteCfg = floo_axi_nw_mesh_32x32_noc_pkg::RouteCfg;
      end
    endcase

    return RouteCfg;
  endfunction

  // Actual declaration of RouteCfg
  localparam route_cfg_t RouteCfg = gen_route_config(N_TILES);

  typedef logic [ADDR_W-1:0] axi_narrow_data_mst_addr_t;
  typedef logic [DATA_W-1:0] axi_narrow_data_mst_data_t;
  typedef logic [STRB_W-1:0] axi_narrow_data_mst_strb_t;
  typedef logic [L2_ID_W-1:0] axi_narrow_data_mst_id_t;
  typedef struct packed {
    logic [ADDR_W-1:0] collective_mask;
    floo_pkg::collect_op_e collective_op;
  } axi_narrow_data_mst_user_t;
  `AXI_TYPEDEF_ALL_CT(axi_narrow_data_mst, axi_narrow_data_mst_req_t, axi_narrow_data_mst_rsp_t,
                      axi_narrow_data_mst_addr_t, axi_narrow_data_mst_id_t,
                      axi_narrow_data_mst_data_t, axi_narrow_data_mst_strb_t,
                      axi_narrow_data_mst_user_t)

  typedef logic [ADDR_W-1:0] axi_narrow_data_slv_addr_t;
  typedef logic [DATA_W-1:0] axi_narrow_data_slv_data_t;
  typedef logic [STRB_W-1:0] axi_narrow_data_slv_strb_t;
  typedef logic [AXI_NOC_ID_W-1:0] axi_narrow_data_slv_id_t;
  typedef struct packed {
    logic [ADDR_W-1:0] collective_mask;
    floo_pkg::collect_op_e collective_op;
  } axi_narrow_data_slv_user_t;
  `AXI_TYPEDEF_ALL_CT(axi_narrow_data_slv, axi_narrow_data_slv_req_t, axi_narrow_data_slv_rsp_t,
                      axi_narrow_data_slv_addr_t, axi_narrow_data_slv_id_t,
                      axi_narrow_data_slv_data_t, axi_narrow_data_slv_strb_t,
                      axi_narrow_data_slv_user_t)

  typedef logic [ADDR_W-1:0] axi_wide_data_mst_addr_t;
  typedef logic [WIDE_DATA_W-1:0] axi_wide_data_mst_data_t;
  typedef logic [WIDE_STRB_W-1:0] axi_wide_data_mst_strb_t;
  typedef logic [iDMA_AxiIdWidth-1:0] axi_wide_data_mst_id_t;
  typedef struct packed {
    logic [ADDR_W-1:0] collective_mask;
    floo_pkg::collect_op_e collective_op;
  } axi_wide_data_mst_user_t;
  `AXI_TYPEDEF_ALL_CT(axi_wide_data_mst, axi_wide_data_mst_req_t, axi_wide_data_mst_rsp_t,
                      axi_wide_data_mst_addr_t, axi_wide_data_mst_id_t, axi_wide_data_mst_data_t,
                      axi_wide_data_mst_strb_t, axi_wide_data_mst_user_t)

  typedef logic [ADDR_W-1:0] axi_wide_data_slv_addr_t;
  typedef logic [WIDE_DATA_W-1:0] axi_wide_data_slv_data_t;
  typedef logic [WIDE_STRB_W-1:0] axi_wide_data_slv_strb_t;
  typedef logic [iDMA_AxiIdWidth-1:0] axi_wide_data_slv_id_t;
  typedef struct packed {
    logic [ADDR_W-1:0] collective_mask;
    floo_pkg::collect_op_e collective_op;
  } axi_wide_data_slv_user_t;
  `AXI_TYPEDEF_ALL_CT(axi_wide_data_slv, axi_wide_data_slv_req_t, axi_wide_data_slv_rsp_t,
                      axi_wide_data_slv_addr_t, axi_wide_data_slv_id_t, axi_wide_data_slv_data_t,
                      axi_wide_data_slv_strb_t, axi_wide_data_slv_user_t)



  `FLOO_TYPEDEF_HDR_T(hdr_t, id_t, id_t, nw_ch_e, rob_idx_t, id_t, floo_pkg::collect_op_e)
  localparam axi_cfg_t AxiCfgN = '{
      AddrWidth: ADDR_W,
      DataWidth: DATA_W,
      UserWidth: ADDR_W + COLLECTIVE_OP_W,
      InIdWidth: AXI_NOC_ID_W,
      OutIdWidth: L2_ID_W
  };
  localparam axi_cfg_t AxiCfgW = '{
      AddrWidth: ADDR_W,
      DataWidth: WIDE_DATA_W,
      UserWidth: ADDR_W + COLLECTIVE_OP_W,
      InIdWidth: iDMA_AxiIdWidth,
      OutIdWidth: iDMA_AxiIdWidth
  };

  `FLOO_TYPEDEF_NW_CHAN_ALL(axi, req, rsp, wide, axi_narrow_data_slv, axi_wide_data_slv, AxiCfgN,
                            AxiCfgW, hdr_t)
  `FLOO_TYPEDEF_NW_LINK_ALL(req, rsp, wide, req, rsp, wide)

endpackage
