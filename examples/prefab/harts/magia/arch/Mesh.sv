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
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 *
 * MAGIA
 */

`ifndef VERILATOR
`include "fractal_sync/assign.svh"
`endif

module MagiaMesh
  import magia_pkg::*;
  import magia_tile_pkg::*;
  import magia_noc_pkg::*;
#(
  parameter int unsigned N_TILES_Y         = magia_pkg::N_TILES_Y,          // Number of Tile rowns
  parameter int unsigned N_TILES_X         = magia_pkg::N_TILES_X,          // Number of Tile columns
  parameter int unsigned N_TILES           = magia_pkg::N_TILES,            // Number of Tiles in the Mesh
  parameter int unsigned N_MEM_BANKS       = magia_pkg::N_MEM_BANKS,        // Number of TCDM banks (1 extra bank for missaligned accesses) per Tile
  parameter int unsigned N_WORDS_BANK      = magia_pkg::N_WORDS_BANK,       // Number of words per TCDM bank

  parameter int unsigned TILE_FSYNC_AGGR_W = magia_pkg::TILE_FSYNC_AGGR_W,  // Width of the FractalSync aggr of the Tile - FS network link
  parameter int unsigned TILE_FSYNC_LVL_W  = magia_pkg::TILE_FSYNC_LVL_W,   // Width of the FractalSync lvl of the Tile - FS network link
  parameter int unsigned TILE_FSYNC_ID_W   = magia_pkg::TILE_FSYNC_ID_W     // Width of the FractalSync id of the Tile - FS network link
)(
  input  logic                                  clk_i,
  input  logic                                  rst_ni,
  input  logic                                  test_mode_i,
  input  logic                                  tile_enable_i,

  input  logic                                  scan_cg_en_i,

  input  logic[31:0]                            boot_addr_i,
  input  logic[31:0]                            mtvec_addr_i,
  input  logic[31:0]                            dm_halt_addr_i,
  input  logic[31:0]                            dm_exception_addr_i,
  input  logic[ 3:0]                            mimpid_patch_i,

  output logic[63:0]                            mcycle_o[N_TILES],

  input  logic[magia_pkg::N_IRQ-1:0]            irq_i[N_TILES],

  input  logic                                  debug_req_i,
  output logic                                  debug_havereset_o[N_TILES],
  output logic                                  debug_running_o[N_TILES],
  output logic                                  debug_halted_o[N_TILES],
  output logic                                  debug_pc_valid_o[N_TILES],
  output logic[31:0]                            debug_pc_o[N_TILES],

  input  logic                                  fetch_enable_i,
  output logic                                  core_sleep_o[N_TILES],

  input  logic                                  wu_wfe_i,

  // Only west-side L2
  output floo_req_t  [N_TILES_Y-1:0]            l2_noc_req_o,
  input  floo_rsp_t  [N_TILES_Y-1:0]            l2_noc_rsp_i,
  input  floo_req_t  [N_TILES_Y-1:0]            l2_noc_req_i,
  output floo_rsp_t  [N_TILES_Y-1:0]            l2_noc_rsp_o,

  input  floo_wide_t [N_TILES_Y-1:0]            l2_noc_wide_i,
  output floo_wide_t [N_TILES_Y-1:0]            l2_noc_wide_o
`ifdef VERILATOR
  , output magia_tile_observe_t                  tile_observe_o[N_TILES]
`endif
);

/*******************************************************/
/**       Internal Signal Definitions Beginning       **/
/*******************************************************/

  logic[31:0] mhartid[N_TILES];

  // FlooNoC buses
  floo_req_t  [N_TILES-1:0] tile_south_req_in,  tile_south_req_out;
  floo_rsp_t  [N_TILES-1:0] tile_south_rsp_in,  tile_south_rsp_out;
  floo_wide_t [N_TILES-1:0] tile_south_wide_in, tile_south_wide_out;
  floo_req_t  [N_TILES-1:0] tile_east_req_in,   tile_east_req_out;
  floo_rsp_t  [N_TILES-1:0] tile_east_rsp_in,   tile_east_rsp_out;
  floo_wide_t [N_TILES-1:0] tile_east_wide_in,  tile_east_wide_out;
  floo_req_t  [N_TILES-1:0] tile_north_req_in,  tile_north_req_out;
  floo_rsp_t  [N_TILES-1:0] tile_north_rsp_in,  tile_north_rsp_out;
  floo_wide_t [N_TILES-1:0] tile_north_wide_in, tile_north_wide_out;
  floo_req_t  [N_TILES-1:0] tile_west_req_in,   tile_west_req_out;
  floo_rsp_t  [N_TILES-1:0] tile_west_rsp_in,   tile_west_rsp_out;
  floo_wide_t [N_TILES-1:0] tile_west_wide_in,  tile_west_wide_out;

  magia_tile_pkg::ht_tile_fsync_req_t ht_tile_fsync_req[N_TILES][1]; // Single link CU-FSync interface
  magia_tile_pkg::ht_tile_fsync_rsp_t ht_tile_fsync_rsp[N_TILES][1]; // Single link CU-FSync interface
  magia_tile_pkg::hn_tile_fsync_req_t hn_tile_fsync_req[N_TILES];
  magia_tile_pkg::hn_tile_fsync_rsp_t hn_tile_fsync_rsp[N_TILES];
  magia_tile_pkg::vt_tile_fsync_req_t vt_tile_fsync_req[N_TILES][1]; // Single link CU-FSync interface
  magia_tile_pkg::vt_tile_fsync_rsp_t vt_tile_fsync_rsp[N_TILES][1]; // Single link CU-FSync interface
  magia_tile_pkg::vn_tile_fsync_req_t vn_tile_fsync_req[N_TILES];
  magia_tile_pkg::vn_tile_fsync_rsp_t vn_tile_fsync_rsp[N_TILES];

  magia_pkg::h_root_fsync_req_t h_root_fsync_req[1][1]; // Single node, single link root node out interface
  magia_pkg::h_root_fsync_rsp_t h_root_fsync_rsp[1][1]; // Single node, single link root node out interface
  magia_pkg::v_root_fsync_req_t v_root_fsync_req[1][1]; // Single node, single link root node out interface
  magia_pkg::v_root_fsync_rsp_t v_root_fsync_rsp[1][1]; // Single node, single link root node out interface

  // What each tile drives on its links. Verilator evaluates the tile as one opaque block,
  // every output a function of every input, so tiles wired output to input are one
  // combinational loop that it converges serially. There the links are sampled on the
  // falling edge: a link output depends only on the state of its tile, so the neighbor
  // sees the same value before the next rising edge, and the tiles evaluate in parallel.
  floo_req_t  [N_TILES-1:0] tile_south_req_drive, tile_east_req_drive, tile_north_req_drive, tile_west_req_drive;
  floo_rsp_t  [N_TILES-1:0] tile_south_rsp_drive, tile_east_rsp_drive, tile_north_rsp_drive, tile_west_rsp_drive;
  floo_wide_t  [N_TILES-1:0] tile_south_wide_drive, tile_east_wide_drive, tile_north_wide_drive, tile_west_wide_drive;
`ifdef VERILATOR
  magia_tile_pkg::ht_tile_fsync_req_t ht_tile_fsync_drive[N_TILES][1];
  magia_tile_pkg::hn_tile_fsync_req_t hn_tile_fsync_drive[N_TILES];
  magia_tile_pkg::vt_tile_fsync_req_t vt_tile_fsync_drive[N_TILES][1];
  magia_tile_pkg::vn_tile_fsync_req_t vn_tile_fsync_drive[N_TILES];
  always_ff @(negedge clk_i) begin
    tile_south_req_out <= tile_south_req_drive;
    tile_south_rsp_out <= tile_south_rsp_drive;
    tile_south_wide_out <= tile_south_wide_drive;
    tile_east_req_out <= tile_east_req_drive;
    tile_east_rsp_out <= tile_east_rsp_drive;
    tile_east_wide_out <= tile_east_wide_drive;
    tile_north_req_out <= tile_north_req_drive;
    tile_north_rsp_out <= tile_north_rsp_drive;
    tile_north_wide_out <= tile_north_wide_drive;
    tile_west_req_out <= tile_west_req_drive;
    tile_west_rsp_out <= tile_west_rsp_drive;
    tile_west_wide_out <= tile_west_wide_drive;
    ht_tile_fsync_req <= ht_tile_fsync_drive;
    hn_tile_fsync_req <= hn_tile_fsync_drive;
    vt_tile_fsync_req <= vt_tile_fsync_drive;
    vn_tile_fsync_req <= vn_tile_fsync_drive;
  end
`else
  assign tile_south_req_out = tile_south_req_drive;
  assign tile_south_rsp_out = tile_south_rsp_drive;
  assign tile_south_wide_out = tile_south_wide_drive;
  assign tile_east_req_out = tile_east_req_drive;
  assign tile_east_rsp_out = tile_east_rsp_drive;
  assign tile_east_wide_out = tile_east_wide_drive;
  assign tile_north_req_out = tile_north_req_drive;
  assign tile_north_rsp_out = tile_north_rsp_drive;
  assign tile_north_wide_out = tile_north_wide_drive;
  assign tile_west_req_out = tile_west_req_drive;
  assign tile_west_rsp_out = tile_west_rsp_drive;
  assign tile_west_wide_out = tile_west_wide_drive;
`endif

/*******************************************************/
/**          Internal Signal Definitions End          **/
`ifndef VERILATOR
/*******************************************************/
/**           Interface Definitions Beginning         **/
/*******************************************************/

  fractal_sync_if #(.AGGR_WIDTH(TILE_FSYNC_AGGR_W),                .LVL_WIDTH(TILE_FSYNC_LVL_W),                .ID_WIDTH(TILE_FSYNC_ID_W))                ht_fsync_if[N_TILES]();
  fractal_sync_if #(.AGGR_WIDTH(magia_tile_pkg::FSYNC_NBR_AGGR_W), .LVL_WIDTH(magia_tile_pkg::FSYNC_NBR_LVL_W), .ID_WIDTH(magia_tile_pkg::FSYNC_NBR_ID_W)) hn_fsync_if[N_TILES]();
  fractal_sync_if #(.AGGR_WIDTH(TILE_FSYNC_AGGR_W),                .LVL_WIDTH(TILE_FSYNC_LVL_W),                .ID_WIDTH(TILE_FSYNC_ID_W))                vt_fsync_if[N_TILES]();
  fractal_sync_if #(.AGGR_WIDTH(magia_tile_pkg::FSYNC_NBR_AGGR_W), .LVL_WIDTH(magia_tile_pkg::FSYNC_NBR_LVL_W), .ID_WIDTH(magia_tile_pkg::FSYNC_NBR_ID_W)) vn_fsync_if[N_TILES]();

/*******************************************************/
/**             Interface Definitions End             **/
/*******************************************************/
/**          Interface Assignments Beginning          **/
/*******************************************************/

  for (genvar i = 0; i < N_TILES; i++) begin: gen_fsync_if_assign
    `FSYNC_ASSIGN_I2S_REQ(ht_fsync_if[i],          ht_tile_fsync_req[i][0])
    `FSYNC_ASSIGN_S2I_RSP(ht_tile_fsync_rsp[i][0], ht_fsync_if[i])
    `FSYNC_ASSIGN_I2S_REQ(hn_fsync_if[i],          hn_tile_fsync_req[i])
    `FSYNC_ASSIGN_S2I_RSP(hn_tile_fsync_rsp[i],    hn_fsync_if[i])
    `FSYNC_ASSIGN_I2S_REQ(vt_fsync_if[i],          vt_tile_fsync_req[i][0])
    `FSYNC_ASSIGN_S2I_RSP(vt_tile_fsync_rsp[i][0], vt_fsync_if[i])
    `FSYNC_ASSIGN_I2S_REQ(vn_fsync_if[i],          vn_tile_fsync_req[i])
    `FSYNC_ASSIGN_S2I_RSP(vn_tile_fsync_rsp[i],    vn_fsync_if[i])
  end

/*******************************************************/
/**             Interface Assignments End             **/
`endif
/*******************************************************/
/**            Hardwired Signals Beginning            **/
/*******************************************************/

  for (genvar i = 0; i < N_TILES; i++) begin: gen_mhartid
    assign mhartid[i] = i;
  end

  assign h_root_fsync_rsp[0][0].wake    = 1'b0;
  assign h_root_fsync_rsp[0][0].sig.lvl = '0;
  assign h_root_fsync_rsp[0][0].sig.id  = '0;
  assign h_root_fsync_rsp[0][0].error   = 1'b0;
  assign v_root_fsync_rsp[0][0].wake    = 1'b0;
  assign v_root_fsync_rsp[0][0].sig.lvl = '0;
  assign v_root_fsync_rsp[0][0].sig.id  = '0;
  assign v_root_fsync_rsp[0][0].error   = 1'b0;

/*******************************************************/
/**               Hardwired Signals End               **/
/*******************************************************/
/**               MAGIA Tiles Beginning               **/
/*******************************************************/

  for (genvar i = 0; i < N_TILES_Y; i++) begin: gen_y_tile
    for (genvar j = 0; j < N_TILES_X; j++) begin: gen_x_tile
`ifdef VERILATOR
      MagiaTile #(
        .N_MEM_BANKS  ( N_MEM_BANKS   ),
        .N_WORDS_BANK ( N_WORDS_BANK  )
      ) i_magia_tile (
`else
      magia_tile #(
        .N_MEM_BANKS  ( N_MEM_BANKS   ),
        .N_WORDS_BANK ( N_WORDS_BANK  ),

        .CORE_ISA     (               ),
        .CORE_A       (               ),
        .CORE_B       (               ),
        .CORE_M       (               ),
        .ERROR_CAP    (               )
      ) i_magia_tile (
`endif
        .clk_i                                                     ,
        .rst_ni                                                    ,
        .test_mode_i                                               ,
        .tile_enable_i                                             ,

        .noc_south_req_o     ( tile_south_req_drive[i*N_TILES_X+j]     ),
        .noc_south_rsp_i     ( tile_south_rsp_in[i*N_TILES_X+j]      ),
        .noc_south_wide_o    ( tile_south_wide_drive[i*N_TILES_X+j]    ),
        .noc_east_req_o      ( tile_east_req_drive[i*N_TILES_X+j]      ),
        .noc_east_rsp_i      ( tile_east_rsp_in[i*N_TILES_X+j]       ),
        .noc_east_wide_o     ( tile_east_wide_drive[i*N_TILES_X+j]     ),
        .noc_north_req_o     ( tile_north_req_drive[i*N_TILES_X+j]     ),
        .noc_north_rsp_i     ( tile_north_rsp_in[i*N_TILES_X+j]      ),
        .noc_north_wide_o    ( tile_north_wide_drive[i*N_TILES_X+j]    ),
        .noc_west_req_o      ( tile_west_req_drive[i*N_TILES_X+j]      ),
        .noc_west_rsp_i      ( tile_west_rsp_in[i*N_TILES_X+j]       ),
        .noc_west_wide_o     ( tile_west_wide_drive[i*N_TILES_X+j]     ),

        .noc_south_req_i     ( tile_south_req_in[i*N_TILES_X+j]   ),
        .noc_south_rsp_o     ( tile_south_rsp_drive[i*N_TILES_X+j]  ),
        .noc_south_wide_i    ( tile_south_wide_in[i*N_TILES_X+j]  ),
        .noc_east_req_i      ( tile_east_req_in[i*N_TILES_X+j]    ),
        .noc_east_rsp_o      ( tile_east_rsp_drive[i*N_TILES_X+j]   ),
        .noc_east_wide_i     ( tile_east_wide_in[i*N_TILES_X+j]   ),
        .noc_north_req_i     ( tile_north_req_in[i*N_TILES_X+j]   ),
        .noc_north_rsp_o     ( tile_north_rsp_drive[i*N_TILES_X+j]  ),
        .noc_north_wide_i    ( tile_north_wide_in[i*N_TILES_X+j]  ),
        .noc_west_req_i      ( tile_west_req_in[i*N_TILES_X+j]    ),
        .noc_west_rsp_o      ( tile_west_rsp_drive[i*N_TILES_X+j]   ),
        .noc_west_wide_i     ( tile_west_wide_in[i*N_TILES_X+j]   ),

        .x_id_i              ( {{(32-$clog2(N_TILES_X+1)){1'b0}}, CollectiveSam[i*N_TILES_X+j].idx.id.x}  ),
        .y_id_i              ( {{(32-$clog2(N_TILES_Y)){1'b0}}, CollectiveSam[i*N_TILES_X+j].idx.id.y}    ),

`ifdef VERILATOR
        .ht_fsync_req_o      ( ht_tile_fsync_drive[i*N_TILES_X+j][0] ),
        .ht_fsync_rsp_i      ( ht_tile_fsync_rsp[i*N_TILES_X+j][0] ),
        .hn_fsync_req_o      ( hn_tile_fsync_drive[i*N_TILES_X+j]    ),
        .hn_fsync_rsp_i      ( hn_tile_fsync_rsp[i*N_TILES_X+j]    ),
        .vt_fsync_req_o      ( vt_tile_fsync_drive[i*N_TILES_X+j][0] ),
        .vt_fsync_rsp_i      ( vt_tile_fsync_rsp[i*N_TILES_X+j][0] ),
        .vn_fsync_req_o      ( vn_tile_fsync_drive[i*N_TILES_X+j]    ),
        .vn_fsync_rsp_i      ( vn_tile_fsync_rsp[i*N_TILES_X+j]    ),
`else
        .ht_fsync_if_o       ( ht_fsync_if[i*N_TILES_X+j]         ),
        .hn_fsync_if_o       ( hn_fsync_if[i*N_TILES_X+j]         ),
        .vt_fsync_if_o       ( vt_fsync_if[i*N_TILES_X+j]         ),
        .vn_fsync_if_o       ( vn_fsync_if[i*N_TILES_X+j]         ),
`endif

        .scan_cg_en_i                                                 ,

        .boot_addr_i                                                  ,
        .mtvec_addr_i                                                 ,
        .dm_halt_addr_i                                               ,
        .dm_exception_addr_i                                          ,
        .mhartid_i           ( mhartid[i*N_TILES_X+j]                ),
        .mimpid_patch_i                                               ,

        .mcycle_o            ( mcycle_o[i*N_TILES_X+j]               ),

        .irq_i               ( irq_i[i*N_TILES_X+j]                  ),

        // Tile expects [N_CLUSTER_CORES:0] (1 main + 8 cluster cores).
        // Replicate the single top-level debug_req_i bit across all cores;
        // implicit name-based connection would leave bits [N:1] unconnected (X)
        // and X-propagate into cv32e40p_controller.debug_req_pending, corrupting
        // ctrl_fsm_cs (observed: ctrl_fsm_cs=0x1xx, debug_req_pending=x).
        .debug_req_i         ( '0                               ),
        .debug_havereset_o   ( debug_havereset_o[i*N_TILES_X+j] ),
        .debug_running_o     ( debug_running_o[i*N_TILES_X+j]   ),
        .debug_halted_o      ( debug_halted_o[i*N_TILES_X+j]    ),
        .debug_pc_valid_o    ( debug_pc_valid_o[i*N_TILES_X+j]  ),
        .debug_pc_o          ( debug_pc_o[i*N_TILES_X+j]        ),

        .fetch_enable_i                                               ,
        .core_sleep_o        ( core_sleep_o[i*N_TILES_X+j]           ),
        .wu_wfe_i
`ifdef VERILATOR
        , .observe_o         ( tile_observe_o[i*N_TILES_X+j]      )
`endif
      );
  `ifdef CORE_TRACES
  `ifdef CV32E40X
`ifndef VERILATOR
      localparam string core_trace_file_name = $sformatf("%s%0d", "log_file_", i*N_TILES_X+j);
      defparam i_magia_tile.i_cv32e40x_ctrl_core.rvfi_i.tracer_i.LOGFILE_PATH_PLUSARG = core_trace_file_name;
`endif
  `endif
  // Note: cv32e40p tracer generates its own filename: trace_core_{cluster_id}_{core_id}.log
  `endif

      if (i == 0) begin
        if (j == 0) begin // T-L corner
          assign tile_north_req_in[i*N_TILES_X+j]  = '0;
          assign tile_north_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_north_wide_in[i*N_TILES_X+j] = '0;
          assign tile_west_req_in[i*N_TILES_X+j]   = l2_noc_req_i[i];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = l2_noc_rsp_i[i];
          assign tile_west_wide_in[i*N_TILES_X+j]  = l2_noc_wide_i[i];
          assign l2_noc_rsp_o[i]                   = tile_west_rsp_out[i*N_TILES_X+j];
          assign l2_noc_req_o[i]                   = tile_west_req_out[i*N_TILES_X+j];
          assign l2_noc_wide_o[i]                  = tile_west_wide_out[i*N_TILES_X+j];
          assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
          assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
          assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
          assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
          assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
          assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
        end else if (j == N_TILES_X-1) begin // T-R corner
          assign tile_north_req_in[i*N_TILES_X+j]  = '0;
          assign tile_north_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_north_wide_in[i*N_TILES_X+j] = '0;
          assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
          assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
          assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
          assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
          assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
          assign tile_east_req_in[i*N_TILES_X+j]   = '0;
          assign tile_east_rsp_in[i*N_TILES_X+j]   = '0;
          assign tile_east_wide_in[i*N_TILES_X+j]  = '0;
        end else if ((j > 0) && (j < (N_TILES_X-1))) begin // First row without corners
          assign tile_north_req_in[i*N_TILES_X+j]  = '0;
          assign tile_north_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_north_wide_in[i*N_TILES_X+j] = '0;
          assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
          assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
          assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
          assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
          assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
          assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
          assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
          assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
        end
      end else if (i == N_TILES_Y-1) begin
        if (j == 0) begin // B-L corner
          assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
          assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
          assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
          assign tile_west_req_in[i*N_TILES_X+j]   = l2_noc_req_i[i];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = l2_noc_rsp_i[i];
          assign tile_west_wide_in[i*N_TILES_X+j]  = l2_noc_wide_i[i];
          assign l2_noc_rsp_o[i]                   = tile_west_rsp_out[i*N_TILES_X+j];
          assign l2_noc_req_o[i]                   = tile_west_req_out[i*N_TILES_X+j];
          assign l2_noc_wide_o[i]                  = tile_west_wide_out[i*N_TILES_X+j];
          assign tile_south_req_in[i*N_TILES_X+j]  = '0;
          assign tile_south_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_south_wide_in[i*N_TILES_X+j] = '0;
          assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
          assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
          assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
        end else if (j == N_TILES_X-1) begin // B-R corner
          assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
          assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
          assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
          assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
          assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
          assign tile_south_req_in[i*N_TILES_X+j]  = '0;
          assign tile_south_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_south_wide_in[i*N_TILES_X+j] = '0;
          assign tile_east_req_in[i*N_TILES_X+j]   = '0;
          assign tile_east_rsp_in[i*N_TILES_X+j]   = '0;
          assign tile_east_wide_in[i*N_TILES_X+j]  = '0;
        end else if ((j > 0) && (j < (N_TILES_X-1))) begin // Last row without corners
          assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
          assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
          assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
          assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
          assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
          assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
          assign tile_south_req_in[i*N_TILES_X+j]  = '0;
          assign tile_south_rsp_in[i*N_TILES_X+j]  = '0;
          assign tile_south_wide_in[i*N_TILES_X+j] = '0;
          assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
          assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
          assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
        end
      end else if (j == 0 && i != 0 && i != N_TILES_Y-1) begin // First column
        assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
        assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
        assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
        assign tile_west_req_in[i*N_TILES_X+j]   = l2_noc_req_i[i];
        assign tile_west_rsp_in[i*N_TILES_X+j]   = l2_noc_rsp_i[i];
        assign tile_west_wide_in[i*N_TILES_X+j]  = l2_noc_wide_i[i];
        assign l2_noc_rsp_o[i]                   = tile_west_rsp_out[i*N_TILES_X+j];
        assign l2_noc_req_o[i]                   = tile_west_req_out[i*N_TILES_X+j];
        assign l2_noc_wide_o[i]                  = tile_west_wide_out[i*N_TILES_X+j];
        assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
        assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
        assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
        assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
        assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
        assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
      end else if (j == N_TILES_X-1 && i != 0 && i != N_TILES_Y-1) begin // Last column
        assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
        assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
        assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
        assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
        assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
        assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
        assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
        assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
        assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
        assign tile_east_req_in[i*N_TILES_X+j]   = '0;
        assign tile_east_rsp_in[i*N_TILES_X+j]   = '0;
        assign tile_east_wide_in[i*N_TILES_X+j]  = '0;
      end else begin // Central tiles
        assign tile_north_req_in[i*N_TILES_X+j]  = tile_south_req_out[(i-1)*N_TILES_X+j];
        assign tile_north_rsp_in[i*N_TILES_X+j]  = tile_south_rsp_out[(i-1)*N_TILES_X+j];
        assign tile_north_wide_in[i*N_TILES_X+j] = tile_south_wide_out[(i-1)*N_TILES_X+j];
        assign tile_west_req_in[i*N_TILES_X+j]   = tile_east_req_out[i*N_TILES_X+j-1];
        assign tile_west_rsp_in[i*N_TILES_X+j]   = tile_east_rsp_out[i*N_TILES_X+j-1];
        assign tile_west_wide_in[i*N_TILES_X+j]  = tile_east_wide_out[i*N_TILES_X+j-1];
        assign tile_south_req_in[i*N_TILES_X+j]  = tile_north_req_out[(i+1)*N_TILES_X+j];
        assign tile_south_rsp_in[i*N_TILES_X+j]  = tile_north_rsp_out[(i+1)*N_TILES_X+j];
        assign tile_south_wide_in[i*N_TILES_X+j] = tile_north_wide_out[(i+1)*N_TILES_X+j];
        assign tile_east_req_in[i*N_TILES_X+j]   = tile_west_req_out[i*N_TILES_X+j+1];
        assign tile_east_rsp_in[i*N_TILES_X+j]   = tile_west_rsp_out[i*N_TILES_X+j+1];
        assign tile_east_wide_in[i*N_TILES_X+j]  = tile_west_wide_out[i*N_TILES_X+j+1];
      end
    end
  end

/*******************************************************/
/**                  MAGIA Tiles End                  **/
/*******************************************************/
/**           FractalSync Network Beginning           **/
/*******************************************************/

  if ((N_TILES_Y == 2) && (N_TILES_X == 2)) begin: gen_2x2_fsync
    fractal_sync_2x2 i_mesh_fsync (
      .clk_i                                  ,
      .rst_ni                                 ,
      .h_1d_fsync_req_i  ( ht_tile_fsync_req ),
      .h_1d_fsync_rsp_o  ( ht_tile_fsync_rsp ),
      .v_1d_fsync_req_i  ( vt_tile_fsync_req ),
      .v_1d_fsync_rsp_o  ( vt_tile_fsync_rsp ),
      .h_nbr_fsycn_req_i ( hn_tile_fsync_req ),
      .h_nbr_fsycn_rsp_o ( hn_tile_fsync_rsp ),
      .v_nbr_fsycn_req_i ( vn_tile_fsync_req ),
      .v_nbr_fsycn_rsp_o ( vn_tile_fsync_rsp ),
      .h_2d_fsync_req_o  ( h_root_fsync_req  ),
      .h_2d_fsync_rsp_i  ( h_root_fsync_rsp  ),
      .v_2d_fsync_req_o  ( v_root_fsync_req  ),
      .v_2d_fsync_rsp_i  ( v_root_fsync_rsp  )
    );
  end else if ((N_TILES_Y == 4) && (N_TILES_X == 4)) begin: gen_4x4_fsync
    fractal_sync_4x4 i_mesh_fsync (
      .clk_i                                  ,
      .rst_ni                                 ,
      .h_1d_fsync_req_i  ( ht_tile_fsync_req ),
      .h_1d_fsync_rsp_o  ( ht_tile_fsync_rsp ),
      .v_1d_fsync_req_i  ( vt_tile_fsync_req ),
      .v_1d_fsync_rsp_o  ( vt_tile_fsync_rsp ),
      .h_nbr_fsycn_req_i ( hn_tile_fsync_req ),
      .h_nbr_fsycn_rsp_o ( hn_tile_fsync_rsp ),
      .v_nbr_fsycn_req_i ( vn_tile_fsync_req ),
      .v_nbr_fsycn_rsp_o ( vn_tile_fsync_rsp ),
      .h_2d_fsync_req_o  ( h_root_fsync_req  ),
      .h_2d_fsync_rsp_i  ( h_root_fsync_rsp  ),
      .v_2d_fsync_req_o  ( v_root_fsync_req  ),
      .v_2d_fsync_rsp_i  ( v_root_fsync_rsp  )
    );
  end else if ((N_TILES_Y == 8) && (N_TILES_X == 8)) begin: gen_8x8_fsync
    fractal_sync_8x8 i_mesh_fsync (
      .clk_i                                  ,
      .rst_ni                                 ,
      .h_1d_fsync_req_i  ( ht_tile_fsync_req ),
      .h_1d_fsync_rsp_o  ( ht_tile_fsync_rsp ),
      .v_1d_fsync_req_i  ( vt_tile_fsync_req ),
      .v_1d_fsync_rsp_o  ( vt_tile_fsync_rsp ),
      .h_nbr_fsycn_req_i ( hn_tile_fsync_req ),
      .h_nbr_fsycn_rsp_o ( hn_tile_fsync_rsp ),
      .v_nbr_fsycn_req_i ( vn_tile_fsync_req ),
      .v_nbr_fsycn_rsp_o ( vn_tile_fsync_rsp ),
      .h_2d_fsync_req_o  ( h_root_fsync_req  ),
      .h_2d_fsync_rsp_i  ( h_root_fsync_rsp  ),
      .v_2d_fsync_req_o  ( v_root_fsync_req  ),
      .v_2d_fsync_rsp_i  ( v_root_fsync_rsp  )
    );
  end else if ((N_TILES_Y == 16) && (N_TILES_X == 16)) begin: gen_16x16_fsync
    fractal_sync_16x16 i_mesh_fsync (
      .clk_i                                  ,
      .rst_ni                                 ,
      .h_1d_fsync_req_i  ( ht_tile_fsync_req ),
      .h_1d_fsync_rsp_o  ( ht_tile_fsync_rsp ),
      .v_1d_fsync_req_i  ( vt_tile_fsync_req ),
      .v_1d_fsync_rsp_o  ( vt_tile_fsync_rsp ),
      .h_nbr_fsycn_req_i ( hn_tile_fsync_req ),
      .h_nbr_fsycn_rsp_o ( hn_tile_fsync_rsp ),
      .v_nbr_fsycn_req_i ( vn_tile_fsync_req ),
      .v_nbr_fsycn_rsp_o ( vn_tile_fsync_rsp ),
      .h_2d_fsync_req_o  ( h_root_fsync_req  ),
      .h_2d_fsync_rsp_i  ( h_root_fsync_rsp  ),
      .v_2d_fsync_req_o  ( v_root_fsync_req  ),
      .v_2d_fsync_rsp_i  ( v_root_fsync_rsp  )
    );
  end else if ((N_TILES_Y == 32) && (N_TILES_X == 32)) begin: gen_32x32_fsync
    fractal_sync_32x32 i_mesh_fsync (
      .clk_i                                  ,
      .rst_ni                                 ,
      .h_1d_fsync_req_i  ( ht_tile_fsync_req ),
      .h_1d_fsync_rsp_o  ( ht_tile_fsync_rsp ),
      .v_1d_fsync_req_i  ( vt_tile_fsync_req ),
      .v_1d_fsync_rsp_o  ( vt_tile_fsync_rsp ),
      .h_nbr_fsycn_req_i ( hn_tile_fsync_req ),
      .h_nbr_fsycn_rsp_o ( hn_tile_fsync_rsp ),
      .v_nbr_fsycn_req_i ( vn_tile_fsync_req ),
      .v_nbr_fsycn_rsp_o ( vn_tile_fsync_rsp ),
      .h_2d_fsync_req_o  ( h_root_fsync_req  ),
      .h_2d_fsync_rsp_i  ( h_root_fsync_rsp  ),
      .v_2d_fsync_req_o  ( v_root_fsync_req  ),
      .v_2d_fsync_rsp_i  ( v_root_fsync_rsp  )
    );
  end else $fatal(1,"Unsupported Mesh configuration");

/*******************************************************/
/**              FractalSync Network End              **/
/*******************************************************/

endmodule: MagiaMesh
