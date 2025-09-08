#include "gemmini.h"
#include <riscv/mmu.h>
#include <riscv/trap.h>
#include <stdexcept>
#include <iostream>
#include <assert.h>
#include <math.h>

using namespace std;

REGISTER_EXTENSION(gemmini, []() { return new gemmini_t; })

#define dprintf(...) { if (p->get_log_commits_enabled()) printf(__VA_ARGS__); }

void gemmini_state_t::reset()
{
  enable = true;
  // mode = OS;
  // sys_act = NONE;
  // acc_act = NONE;
  // sys_shift = 0;
  // output_sp_addr = 0;
  // load_stride = DIM * sizeof(elem_t);
  // store_stride = DIM * sizeof(elem_t);
  // pool_stride = 0;
  // load_shrunk = false;

  spad.clear();
  spad.resize(sp_matrices*DIM, std::vector<elem_t>(DIM, 0));

  pe_state.clear();
  pe_state.resize(DIM, std::vector<acc_t>(DIM));

  accumulator.clear();
  accumulator.resize(accum_rows, std::vector<acc_t>(DIM, 0));

  // cisc reset
  a_addr = b_addr = c_addr = d_addr = 0;
  m = n = k = 0;
  repeating_bias = false;

  // Norm reset
  for (int i = 0; i < NORM_STAT_IDS; i++) norm_reset[i] = true;

  // Dummy counter reset
  snapshot_enable = false;
  op_in_progress = false;

  resetted = true;

  printf("Gemmini extension configured with:\n");
  printf("    dim = %u\n", DIM);
}

void gemmini_t::reset(processor_t &proc) {
  gemmini_state.reset();
}

void gemmini_t::reset() {
  gemmini_state.reset();
}

template <class T>
T gemmini_t::read_from_dram(reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load<uint8_t>(addr + byte_idx) << (byte_idx*8);
  }
  return value;
}

template <class T>
std::vector<std::vector<T>> *
matrix_zeroes(reg_t rows, reg_t cols) {
  return new std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0));
}

template <class T>
std::vector<std::vector<T>> *
gemmini_t::read_matrix_from_dram(reg_t addr, reg_t rows, reg_t cols,
                                  bool zeroable, bool repeating_bias) {
  // Read and return Matrix of size `rows*cols` from address `addr` in main
  // memory

  // Initialize to all zeroes
  auto result = matrix_zeroes<T>(rows, cols);

  // if an input matrix is at addr 0, it is NULL, so don't do anything with
  // it only the D matrix is zeroable; the A, B matrices must be valid
  if(addr == 0) {
    if(zeroable) {
      return result;
    }
    printf("ERROR: non-zeroable matrix given address zero!\n");
    exit(1);
  }

  // Load from memory
  for (size_t i = 0; i < rows; i++) {
    auto ii = repeating_bias ? 0 : i;
    auto const dram_row_addr = addr + ii*sizeof(T)*cols;
    for (size_t j = 0; j < cols; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(T);
#ifdef ELEM_T_IS_FLOAT
      result->at(i).at(j) = elem_t_bits_to_elem_t(gemmini_t::read_from_dram<elem_t_bits>(dram_byte_addr));
#else
      result->at(i).at(j) = gemmini_t::read_from_dram<elem_t>(dram_byte_addr);
#endif
    }
  }
  return result;
}

template <class T>
void gemmini_t::write_to_dram(reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store<uint8_t>(addr + byte_idx, (data >> (byte_idx*8)) & 0xFF);
  }
}

// Move a gemmini block from DRAM at dram_addr (byte addr) to
// the scratchpad/accumulator at sp_addr (gemmini-row addressed)
void gemmini_t::mvin(reg_t dram_addr, reg_t sp_addr, int state_id) {
  bool const accumulator = (sp_addr >> 31) & 0x1;
  bool const accumulate = (sp_addr >> 30) & 0x1;
  auto const base_row_addr = (sp_addr & 0x1FFFFFFF); // Strip accumulator addressing bits [31:29]
  auto const cols = (sp_addr >> addr_len) & 0xFFFF;
  auto const rows = (sp_addr >> (addr_len + 16)) & 0xFFFF;

  bool is_zeros = dram_addr == 0;

  auto const load_stride = gemmini_state.load_strides[state_id];
  auto const load_block_stride = gemmini_state.load_block_strides[state_id];
  auto const load_shrunk = gemmini_state.load_shrunks[state_id];
#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
  auto const load_scale = gemmini_state.load_scales[state_id];
#endif
  auto const pixels_per_row = gemmini_state.pixels_per_rows[state_id];

  dprintf("GEMMINI: mvin - 0x%02lx cols and 0x%02lx rows from 0x%08lx to addr 0x%08lx\n", cols, rows, dram_addr, sp_addr & 0xFFFFFFFF);

  for (size_t row = 0; row < rows; ++row) {
    auto const dram_row_addr = dram_addr + row*load_stride;

    for (size_t col = 0; col < cols; ++col) {
      const size_t block = col / DIM;
      const size_t spad_col = col % DIM;
      const size_t spad_row = base_row_addr + row + block*load_block_stride;

      for (size_t pixel = 0; pixel < pixels_per_row && pixel <= spad_row; pixel++) {

        if (accumulator) {
            auto const dram_byte_addr = dram_row_addr + col *
              (load_shrunk ? sizeof(elem_t) : sizeof(acc_t));

            acc_t value;
            if (is_zeros) {
              value = 0;
            } else if (!load_shrunk) {
#ifdef ELEM_T_IS_FLOAT
              value = acc_t_bits_to_acc_t(read_from_dram<acc_t_bits>(dram_byte_addr));
#else
              value = read_from_dram<acc_t>(dram_byte_addr);
#endif

#ifdef HAS_MVIN_ACC_SCALE
              value = mvin_scale_acc(value, load_scale);
#endif
            } else {
#ifdef ELEM_T_IS_FLOAT
              value = elem_t_bits_to_elem_t(read_from_dram<elem_t_bits>(dram_byte_addr));
#else
              value = read_from_dram<elem_t>(dram_byte_addr);
#endif

#ifdef HAS_MVIN_SCALE
              value = mvin_scale(value, load_scale);
#endif
            }

            if (accumulate) {
              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) += value;
            } else {
              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) = value;
            }

#ifdef ELEM_T_IS_FLOAT
            dprintf("%f ", gemmini_state.accumulator.at(spad_row).at(spad_col));
#else
            dprintf("%d ", gemmini_state.accumulator.at(spad_row).at(spad_col));
#endif
        } else {
            auto const dram_byte_addr = dram_row_addr + col*sizeof(elem_t);

            elem_t value;
            if (is_zeros) {
              value = 0;
            } else {
#ifdef ELEM_T_IS_FLOAT
              value = elem_t_bits_to_elem_t(read_from_dram<elem_t_bits>(dram_byte_addr));
#else
              value = read_from_dram<elem_t>(dram_byte_addr);
#endif

#ifdef HAS_MVIN_SCALE
              value = mvin_scale(value, load_scale);
#endif
            }

            gemmini_state.spad.at(spad_row - pixel).at(spad_col + pixel * cols) = value;

#ifdef ELEM_T_IS_FLOAT
            dprintf("%f ", gemmini_state.spad.at(spad_row).at(spad_col));
#else
            dprintf("%d", gemmini_state.spad.at(spad_row).at(spad_col));
#endif
        }
      }
    }
    dprintf("\n");
  }
}

void gemmini_t::mvout(reg_t dram_addr, reg_t sp_addr) {
  bool const accumulator = (sp_addr >> 31) & 0x1;
  bool const full = (sp_addr >> 29) & 0x1;
  auto const norm_cmd = static_cast<gemmini_state_t::NormCmd>((sp_addr >> 26) & 0x7);
  auto const base_row_addr = (sp_addr & 0x3FFFFFF); // Strip accumulator addressing bits [31:26]
  auto const cols = (sp_addr >> addr_len) & 0xFFFF;
  auto const rows = (sp_addr >> (addr_len + 16)) & 0xFFFF;

  const int block_stride = DIM;

  dprintf("GEMMINI: mvout - 0x%02lx cols and 0x%02lx rows from 0x%08lx to addr 0x%08lx\n", cols, rows, sp_addr, dram_addr);

  if (gemmini_state.pool_stride == 0) {
    for (size_t i = 0; i < rows; ++i) {
      auto const dram_row_addr = dram_addr + i*gemmini_state.store_stride;

      bool should_write = true;
      for (size_t j = 0; j < cols; j += DIM) {
        const size_t block = j / DIM;
        const size_t spad_row = base_row_addr + block*block_stride + i;
        const size_t len = cols - j > DIM ? DIM : cols - j;

        const bool is_last = j + DIM >= cols;
        const auto n_cmd = is_last ? norm_cmd : non_terminating_norm_cmd(norm_cmd);

        should_write = apply_norm(
            &gemmini_state.accumulator.at(spad_row).at(0),
            len, n_cmd);
      }

      if (!should_write)
        continue;

      for (size_t j = 0; j < cols; ++j) {
        const size_t block = j / DIM;
        const size_t spad_col = j % DIM;
        const size_t spad_row = base_row_addr + block*block_stride + i;

        if (accumulator) { // Apply shift and activation when moving out of accumulator
          acc_t acc_value = gemmini_state.accumulator.at(spad_row).at(spad_col);
          acc_value = apply_pre_activation_acc(acc_value);

          auto shifted = acc_scale(acc_value, gemmini_state.acc_shift);
          elem_t activated = apply_activation_acc(shifted); // Activation is always applied in either WS/OS mode

          auto const sizeof_output = full ? sizeof(acc_t) : sizeof(elem_t);

          auto const dram_byte_addr = dram_row_addr + j*sizeof_output;
#ifdef ELEM_T_IS_FLOAT
          if (full) {
            write_to_dram<acc_t_bits>(dram_byte_addr, acc_t_to_acc_t_bits(acc_value));
            dprintf("%f ", acc_value);
          } else {
            write_to_dram<elem_t_bits>(dram_byte_addr, elem_t_to_elem_t_bits(activated));
            dprintf("%f ", activated);
          }
#else
          if (full) {
            write_to_dram<acc_t>(dram_byte_addr, acc_value);
            dprintf("%d ", acc_value);
          } else {
            write_to_dram<elem_t>(dram_byte_addr, activated);
            dprintf("%d ", activated);
          }
#endif

        } else { // Scratchpad, write to DRAM directly
          auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
          elem_t value = gemmini_state.spad.at(spad_row).at(spad_col);

#ifdef ELEM_T_IS_FLOAT
          write_to_dram<elem_t_bits>(dram_byte_addr, elem_t_to_elem_t_bits(value));
          dprintf("%f ", value);
#else
          write_to_dram<elem_t>(dram_byte_addr, value);
          dprintf("%d ", value);
#endif
        }
      }
      dprintf("\n");
    }

  } else {

    // Perform pooling
    auto const pool_stride = gemmini_state.pool_stride;
    auto const pool_size = gemmini_state.pool_size;
    auto const pool_out_dim = gemmini_state.pool_out_dim;
    auto const porows = gemmini_state.pool_porows;
    auto const pocols = gemmini_state.pool_pocols;
    auto const orows = gemmini_state.pool_orows;
    auto const ocols = gemmini_state.pool_ocols;
    auto const plpad = gemmini_state.pool_lpad;
    auto const pupad = gemmini_state.pool_upad;
    auto const channels = cols;

    for (int porow = 0; porow < porows; porow++) {
      for (int pocol = 0; pocol < pocols; pocol++) {
        for (int poch = 0; poch < (int)channels; poch++) {
          elem_t value = elem_t_min;

          for (int wrow = 0; wrow < pool_size; wrow++) {
            for (int wcol = 0; wcol < pool_size; wcol++) {

              const int orow = porow * pool_stride + wrow - pupad;
              const int ocol = pocol * pool_stride + wcol - plpad;

              const int row_addr = base_row_addr + orow*ocols + ocol;

              elem_t elem;

              if (orow < 0 || ocol < 0 || orow >= orows || ocol >= ocols) {
                elem = 0;
              } else if (accumulator) {
                acc_t acc_value = gemmini_state.accumulator.at(row_addr).at(poch);
                auto shifted = acc_scale(acc_value, gemmini_state.acc_shift);
                elem = apply_activation_acc(shifted); // Activation is always applied in either WS/OS mode
              } else {
                elem = gemmini_state.spad.at(row_addr).at(poch);
              }

              if (elem > value) {
                value = elem;
              }
            }
          }

          auto const dram_byte_addr = dram_addr + (porow * pool_out_dim + pocol) * gemmini_state.store_stride + poch * sizeof(elem_t);

#ifdef ELEM_T_IS_FLOAT
          write_to_dram<elem_t_bits>(dram_byte_addr, elem_t_to_elem_t_bits(value));
#else
          write_to_dram<elem_t>(dram_byte_addr, value);
#endif
        }
      }
    }
  }
}

void gemmini_t::preload(reg_t bd_addr, reg_t c_addr) {
  // TODO: rename these state variables
  gemmini_state.preload_sp_addr = static_cast<uint32_t>(bd_addr & 0xFFFFFFFF);
  gemmini_state.output_sp_addr = static_cast<uint32_t>(c_addr & 0xFFFFFFFF);

  gemmini_state.preload_cols = (bd_addr >> addr_len) & 0xFFFF;
  gemmini_state.preload_rows = (bd_addr >> (addr_len + 16)) & 0xFFFF;
  gemmini_state.output_cols = (c_addr >> addr_len) & 0xFFFF;
  gemmini_state.output_rows = (c_addr >> (addr_len + 16)) & 0xFFFF;

  dprintf("GEMMINI: preload - scratchpad output addr = 0x%08x, scratchpad preload addr = 0x%08x\n",
            gemmini_state.output_sp_addr, gemmini_state.preload_sp_addr);
}

void gemmini_t::config(reg_t rs1, reg_t rs2) {
  if ((rs1 & 0b11) == 0) { // rs1[1:0] == 2'b00, config_ex, configure execute pipeline
    gemmini_state_t::Dataflow new_mode;
    gemmini_state_t::Activation new_act;
    reg_t new_sys_shift, new_sys_acc_shift, new_c_stride, new_a_stride, new_a_transpose, new_b_transpose;

    auto rs1_2 = (rs1 >> 2) & 0b1; // extract rs1[2], 0 = output stationary, 1 = weight stationary
    if (rs1_2 == 0) {
      new_mode = gemmini_state_t::OS;
    } else {
      new_mode = gemmini_state_t::WS;
    }

    auto rs1_4_3 = (rs1 >> 3) & 0b11; // extract rs1[4:3], 0 = no activation, 1 = ReLU, 2 = ReLU6
    if (rs1_4_3 == 0) {
      if (static_cast<int>(gemmini_state.acc_act) != 4) {
        new_act = gemmini_state_t::NONE;
      } else {
        new_act = gemmini_state_t::SOFTMAX;
      }
    } else if (rs1_4_3 == 1) {
      new_act = gemmini_state_t::RELU;
    } else if (rs1_4_3 == 2) {
      new_act = gemmini_state_t::LAYERNORM;
    } else if (rs1_4_3 == 3) {
      new_act = gemmini_state_t::IGELU;
    } else {
      assert(false);
    }

    new_sys_shift = (rs2) & 0xFFFFFFFF;
    new_sys_acc_shift = (rs1 >> 32) & 0xFFFFFFFF;
    new_c_stride = (rs2 >> 48) & 0xFFFF;
    new_a_stride = (rs1 >> 16) & 0xFFFF;
    new_a_transpose = (rs1 >> 8) & 0x1;
    new_b_transpose = (rs1 >> 9) & 0x1;

    const bool set_only_strides = (rs1 >> 7) & 0x1;

    dprintf("GEMMINI: config_ex - set dataflow mode from %d to %d\n", gemmini_state.mode, new_mode);
    dprintf("GEMMINI: config_ex - set activation function from %d to %d\n", gemmini_state.sys_act, new_act);
    dprintf("GEMMINI: config_ex - set sys_shift from %lu to %lu\n", gemmini_state.sys_shift, new_sys_shift);

    // assert(new_acc_shift >= 0 && new_acc_shift < sizeof(acc_t)*8);
    assert(new_sys_shift >= 0 && new_sys_shift < sizeof(output_t)*8);

    if (!set_only_strides) {
      gemmini_state.mode = new_mode;
      gemmini_state.sys_act = new_act;
      gemmini_state.sys_shift = new_sys_shift;
      gemmini_state.sys_acc_shift = new_sys_acc_shift;
      gemmini_state.a_transpose = new_a_transpose;
      gemmini_state.b_transpose = new_b_transpose;
    }

    gemmini_state.c_stride = new_c_stride;
    gemmini_state.a_stride = new_a_stride;

    assert(!(new_mode == gemmini_state_t::OS && !new_a_transpose && new_b_transpose) && !(new_mode == gemmini_state_t::WS && new_a_transpose && new_b_transpose));

  } else if ((rs1 & 0b11) == 1) { // rs1[1:0] == 2'b01, config_mvin, configure load pipeline
    const int state_id = (rs1 >> 3) & 0x3;
    dprintf("GEMMINI: config_mvin - set load stride from %lu to %lu\n", gemmini_state.load_strides[state_id], rs2);
    gemmini_state.load_strides[state_id] = rs2;
    gemmini_state.load_block_strides[state_id] = (rs1 >> 16) & 0xFFFF;
#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
    dprintf("GEMMINI: config_mvin - set load scale from %lu to %lu\n", gemmini_state.load_scales[state_id], scale_t_bits_to_scale_t(rs1 >> 32));
    gemmini_state.load_scales[state_id] = scale_t_bits_to_scale_t(rs1 >> 32);
    gemmini_state.load_shrunks[state_id] = (rs1 >> 2) & 1;
#endif
    gemmini_state.pixels_per_rows[state_id] = (rs1 >> 8) & 0xFF;

    if (gemmini_state.pixels_per_rows[state_id] == 0)
      gemmini_state.pixels_per_rows[state_id] = 1;

#ifndef HAS_FIRST_LAYER_OPTIMIZATIONS
    assert(gemmini_state.pixels_per_rows[state_id] <= 1, "If Gemmini is built without first-layer optimizations, then 'pixels_per_rows' cannot be larger than 1\n");
#endif
  } else if ((rs1 & 0b11) == 2) { // rs1[1:0] == 2'b10, config_mvout, configure store pipeline
    dprintf("GEMMINI: config_mvout - set store stride from %lu to %lu\n", gemmini_state.store_stride, rs2 & 0xFFFFFFFF);
    gemmini_state.store_stride = rs2 & 0xFFFFFFFF;

    gemmini_state_t::Activation new_act;
    auto rs1_3_2 = (rs1 >> 2) & 0b11; // extract rs1[3:2], 0 = no activation, 1 = ReLU, 2 = ReLU6
    if (rs1_3_2 == 0) {
      if (static_cast<int>(gemmini_state.acc_act) != 4) {
        new_act = gemmini_state_t::NONE;
      } else {
        new_act = gemmini_state_t::SOFTMAX;
      }
    } else if (rs1_3_2 == 1) {
      new_act = gemmini_state_t::RELU;
    } else if (rs1_3_2 == 2) {
      new_act = gemmini_state_t::LAYERNORM;
    } else if (rs1_3_2 == 3) {
      new_act = gemmini_state_t::IGELU;
    } else {
      assert(false);
    }
    gemmini_state.acc_act = new_act;

    auto new_acc_shift = (rs2 >> 32) & 0xFFFFFFFF;
    gemmini_state.acc_shift = acc_scale_t_bits_to_acc_scale_t(new_acc_shift);

    gemmini_state.pool_stride = (rs1 >> 4) & 0x3;
    gemmini_state.pool_size = (rs1 >> 6) & 0x3;
    gemmini_state.pool_upad = (rs1 >> 8) & 0x3;
    gemmini_state.pool_lpad = (rs1 >> 10) & 0x3;
    gemmini_state.pool_out_dim = (rs1 >> 24) & 0xFF;
    gemmini_state.pool_porows = (rs1 >> 32) & 0xFF;
    gemmini_state.pool_pocols = (rs1 >> 40) & 0xFF;
    gemmini_state.pool_orows = (rs1 >> 48) & 0xFF;
    gemmini_state.pool_ocols = (rs1 >> 56) & 0xFF;

    if (gemmini_state.pool_stride == 0) {
        dprintf("GEMMINI: config_mvout - no pooling\n");
    } else {
        dprintf("GEMMINI: config_mvout - set pool_stride to %u\n", gemmini_state.pool_stride);
        dprintf("GEMMINI: config_mvout - set pool_size to %u\n", gemmini_state.pool_size);
        dprintf("GEMMINI: config_mvout - set pool_upad to %u\n", gemmini_state.pool_upad);
        dprintf("GEMMINI: config_mvout - set pool_lpad to %u\n", gemmini_state.pool_lpad);
        dprintf("GEMMINI: config_mvout - set pool_out_dim to %u\n", gemmini_state.pool_out_dim);
        dprintf("GEMMINI: config_mvout - set pool_porows to %u\n", gemmini_state.pool_porows);
        dprintf("GEMMINI: config_mvout - set pool_pocols to %u\n", gemmini_state.pool_pocols);
        dprintf("GEMMINI: config_mvout - set pool_orows to %u\n", gemmini_state.pool_orows);
        dprintf("GEMMINI: config_mvout - set pool_ocols to %u\n", gemmini_state.pool_ocols);
        dprintf("GEMMINI: config_mvout - rs1 is %llx\n", rs1);
    }
  } else if ((rs1 & 0b11) == 3) { // rs1[1:0] == 2'b11, config_norm, configure norm pipeline
    gemmini_state.norm_stat_id = (rs1 >> 8) & 0xFF;
    if (!((rs1 >> 17) & 1)) { // set stat_id only
      gemmini_state.igelu_qb = rs2 & 0xFFFFFFFF;
      gemmini_state.igelu_qc = (rs2 >> 32) & 0xFFFFFFFF;

      auto act_msb = (rs1 >> 16) & 0x1;
      gemmini_state.acc_act = static_cast<gemmini_state_t::Activation>((act_msb << 2) | static_cast<int>(gemmini_state.acc_act));
      assert(static_cast<int>(gemmini_state.acc_act) <= 4);

      if ((rs1 >> 18) & 1) { // q const type = 1
        gemmini_state.qln2_inv = (rs1 >> 32) & 0xFFFFFFFF;
      } else {
        gemmini_state.qln2 = (rs1 >> 32) & 0xFFFFFFFF;
      }
    }
  }
}

void gemmini_t::compute(reg_t a_addr, reg_t bd_addr, bool preload) {
  auto a_addr_real = static_cast<uint32_t>(a_addr & 0xFFFFFFFF);
  auto bd_addr_real = static_cast<uint32_t>(bd_addr & 0xFFFFFFFF);

  const uint16_t a_cols = (a_addr >> addr_len) & 0xFFFF;
  const uint16_t a_rows = (a_addr >> (addr_len + 16)) & 0xFFFF;

  const uint16_t bd_cols = (bd_addr >> addr_len) & 0xFFFF;
  const uint16_t bd_rows = (bd_addr >> (addr_len + 16)) & 0xFFFF;

  dprintf("GEMMINI: compute - preload = %d, scratchpad A addr = 0x%08x,"
           "scratchpad B addr 0x%08x\n", preload, a_addr_real, bd_addr_real);

  // Preload
  if (preload) {
    dprintf("GEMMINI: compute - PEs after preloading:\n");
    for (size_t i = 0; i < DIM; i++) {
      for (size_t j = 0; j < DIM; j++) {
        // TODO: Handle preloads from accumulator, values are shifted and activated before preload
        if (~gemmini_state.preload_sp_addr != 0) {
          assert(((gemmini_state.preload_sp_addr >> 30) & 0b11) == 0); // Preloads from accumulator not supported
        }

        bool preload_tranpose = (gemmini_state.mode == gemmini_state_t::WS) &&
          gemmini_state.b_transpose;
        size_t r = preload_tranpose ? j : i;
        size_t c = preload_tranpose ? i : j;

        // In OS mode, pe_state stores the accumulator values
        // In WS mode, pe_state stores the persistent weight matrix
        if (i < gemmini_state.preload_rows && j < gemmini_state.preload_cols) {
          auto preload_value = (~gemmini_state.preload_sp_addr == 0) ? 0 :
                  gemmini_state.spad.at(gemmini_state.preload_sp_addr + r).at(c);
          gemmini_state.pe_state.at(i).at(j) = preload_value;
        } else {
          gemmini_state.pe_state.at(i).at(j) = 0;
        }

#ifdef ELEM_T_IS_FLOAT
        dprintf("%f ", gemmini_state.pe_state.at(i).at(j));
#else
        dprintf("%d ", gemmini_state.pe_state.at(i).at(j));
#endif
      }
      dprintf("\n");
    }
  }

  // Compute
  // For OS, accumulate the PE results internally in pe_state
  // For WS, allocate a new results array which won't affect pe_state, seed the results array with the bias (D) matrix
  auto results = std::vector<std::vector<acc_t>>(DIM, std::vector<acc_t>(DIM));
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      if (i < bd_rows && j < bd_cols) {
        results.at(i).at(j) = (~bd_addr_real == 0) ? 0 : gemmini_state.spad.at(bd_addr_real + i).at(j);
      } else {
        results.at(i).at(j) = 0;
      }
    }
  }

  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      for (size_t k = 0; k < DIM; ++k) {
        elem_t a;
        if (~a_addr_real != 0) {
            const size_t r = gemmini_state.a_stride * (gemmini_state.a_transpose ? k : i);
            const size_t c = gemmini_state.a_transpose ? i : k;

            a = i < a_rows && k < a_cols ? gemmini_state.spad.at(a_addr_real + r).at(c) : 0;
        }

        if (gemmini_state.mode == gemmini_state_t::WS) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
          results.at(i).at(j) += a * gemmini_state.pe_state.at(k).at(j);
#pragma GCC diagnostic pop
        } else {
          elem_t b = 0;
          if (~bd_addr_real != 0) {
            const size_t r = gemmini_state.b_transpose ? j : k;
            const size_t c = gemmini_state.b_transpose ? k : j;

            b = k < bd_rows && j < bd_cols ? gemmini_state.spad.at(bd_addr_real + r).at(c) : 0;
          }

          gemmini_state.pe_state.at(i).at(j) += a * b;
        }
      }
    }
  }

  dprintf("GEMMINI: compute - PEs after matmul:\n");
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
#ifdef ELEM_T_IS_FLOAT
      dprintf("%f ", gemmini_state.pe_state.at(i).at(j));
#else
      dprintf("%d ", gemmini_state.pe_state.at(i).at(j));
#endif
    }
    dprintf("\n");
  }

  // Write results
  if (~gemmini_state.output_sp_addr != 0) {
    bool const acc = (gemmini_state.output_sp_addr >> 31) & 0x1;
    bool const acc_accum = (gemmini_state.output_sp_addr >> 30) & 0x1;
    auto const base_sp_addr = gemmini_state.output_sp_addr & 0x1FFFFFFF;
    dprintf("GEMMINI: compute - writing results to addr 0x%08x, :\n", gemmini_state.output_sp_addr);

    for (size_t i = 0; i < gemmini_state.output_rows; ++i) {
      for (size_t j = 0; j < gemmini_state.output_cols; ++j) {
        acc_t value = gemmini_state.mode == gemmini_state_t::OS ? gemmini_state.pe_state.at(i).at(j) : results.at(i).at(j);
        if (acc) {
          output_t shifted = gemmini_state.mode == gemmini_state_t::OS ?
                  sys_shift(value, gemmini_state.sys_shift) :
                  sys_shift(value, 0);

          if (acc_accum) {
            gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) += value;
          } else { // Overwrite
            gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) = value;
          }

#ifdef ELEM_T_IS_FLOAT
          dprintf("%f ", gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j));
#else
          dprintf("%d ", gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j));
#endif
        } else { // Move to scratchpad, apply activation along the way
          elem_t shifted = gemmini_state.mode == gemmini_state_t::OS ?
                             sys_shift(value, gemmini_state.sys_shift) :
                             sys_shift(value, 0);
          elem_t activated = apply_activation_sys(shifted);
          gemmini_state.spad.at(base_sp_addr + gemmini_state.c_stride * i).at(j) = activated;
#ifdef ELEM_T_IS_FLOAT
          dprintf("%f ", gemmini_state.spad.at(base_sp_addr + gemmini_state.c_stride * i).at(j));
#else
          dprintf("%d ", gemmini_state.spad.at(base_sp_addr + gemmini_state.c_stride * i).at(j));
#endif
        }
      }
      dprintf("\n");
    }
  }
}

void gemmini_t::loop_ws(reg_t rs1, reg_t rs2) {
  const bool ex_accumulate = rs1 & 1;
  const bool full_C = (rs1 >> 1) & 1;
  const bool low_D = (rs1 >> 2) & 1;
  const uint8_t act = (rs1 >> 8) & 0x7;
  const bool a_transpose = rs2 & 1;
  const bool b_transpose = (rs2 >> 1) & 1;
  uint8_t a_spad_id = (rs1 >> 18) & 0b11;
  uint8_t b_spad_id = (rs1 >> 16) & 0b11;
  const bool is_resadd = (rs2 >> 2) & 1;

  const uint16_t I = gemmini_state.loop_ws_I;
  const uint16_t J = gemmini_state.loop_ws_J;
  const uint16_t K = gemmini_state.loop_ws_K;

  const uint16_t pad_I = gemmini_state.loop_ws_pad_I;
  const uint16_t pad_J = gemmini_state.loop_ws_pad_J;
  const uint16_t pad_K = gemmini_state.loop_ws_pad_K;

  const uint32_t GARBAGE_ADDR = ~0;

  const int total_spad_rows = (I * K + K * J) * DIM;
  const int total_acc_rows = (I * J) * DIM;

  if (total_spad_rows > BANK_NUM * BANK_ROWS / 2 || total_acc_rows > ACC_ROWS / 2) {
    printf("LOOP_WS bounds were too large for double-buffering\n");
    exit(1);
  }

  uint32_t A_sp_addr_start = 0;
  uint32_t B_sp_addr_start = (BANK_NUM * BANK_ROWS / 2) - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));
  if(a_spad_id == 2) A_sp_addr_start = (BANK_NUM * BANK_ROWS) / 2;
  if(b_spad_id == 2) B_sp_addr_start = (BANK_NUM * BANK_ROWS) - K * J * DIM;
  if(is_resadd) {
    A_sp_addr_start = 1 << (ADDR_LEN - 1);
    B_sp_addr_start = 3 << (ADDR_LEN - 2);
    for(uint16_t i = 0; i < I; i++){
      for(uint16_t j = 0; j < J; j++){
        const uint32_t A_sp_addr = A_sp_addr_start + (i*J+j)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (i*J+j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J+j)*DIM;
        uint64_t dram_addr, cols, rows;
        dram_addr = gemmini_state.loop_ws_A + (i*gemmini_state.loop_ws_A_stride + j) * DIM * sizeof(elem_t);
        cols = DIM - (j == J-1 ? pad_J : 0);
        rows = DIM - (i == I-1 ? pad_I : 0);
        mvin(dram_addr, (rows << 48) | (cols << 32) | A_sp_addr, 0);

        dram_addr = gemmini_state.loop_ws_B + (i*gemmini_state.loop_ws_B_stride + j) * DIM * sizeof(elem_t);
        cols = DIM - (j == J-1 ? pad_J : 0);
        rows = DIM - (i == I-1 ? pad_I : 0);
        mvin(dram_addr, (rows << 48) | (cols << 32) | B_sp_addr, 1);
        if(gemmini_state.loop_ws_C != 0){
          const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
          const uint64_t C_dram_addr = gemmini_state.loop_ws_C +
            (i*gemmini_state.loop_ws_C_stride + j) * DIM * sizeof_C;
          const uint64_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const uint64_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

          mvout(C_dram_addr, (C_rows << 48) | (C_cols << 32) | C_sp_addr);
        }
      }
    }
    return;
  }

  if (gemmini_state.loop_ws_D != 0) {
    for (uint16_t i = 0; i < I; i++) {
      for (uint16_t j = 0; j < J; j++) {
        const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);

        const uint64_t dram_addr = gemmini_state.loop_ws_D +
          (i * gemmini_state.loop_ws_D_stride + j) * DIM * sizeof_D;

        const uint64_t sp_addr = D_sp_addr_start + (i*J + j)*DIM;

        const uint64_t cols = DIM - (j == J-1 ? pad_J : 0);
        const uint64_t rows = DIM - (i == I-1 ? pad_I : 0);

        mvin(dram_addr, (rows << 48) | (cols << 32) | sp_addr, 2);
      }
    }
  }

  for (uint16_t k = 0; k < K; k++) {
    for (uint16_t j = 0; j < J; j++) {
      for (uint16_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        // Mvin A
        if (j == 0 && gemmini_state.loop_ws_A != 0) {
          uint64_t dram_addr, cols, rows;

          if (a_transpose) {
            dram_addr = gemmini_state.loop_ws_A +
                (k*gemmini_state.loop_ws_A_stride + i) * DIM * sizeof(elem_t);
            cols = DIM - (i == I-1 ? pad_I : 0);
            rows = DIM - (k == K-1 ? pad_K : 0);
          } else {
            dram_addr = gemmini_state.loop_ws_A +
                (i*gemmini_state.loop_ws_A_stride + k) * DIM * sizeof(elem_t);
            cols = DIM - (k == K-1 ? pad_K : 0);
            rows = DIM - (i == I-1 ? pad_I : 0);
          }

          mvin(dram_addr, (rows << 48) | (cols << 32) | A_sp_addr, 0);
        }

        // Mvin B
        if (i == 0 && gemmini_state.loop_ws_B != 0) {
          uint64_t dram_addr, cols, rows;

          if (b_transpose) {
            dram_addr = gemmini_state.loop_ws_B +
                (j*gemmini_state.loop_ws_B_stride + k) * DIM * sizeof(elem_t);
            cols = DIM - (k == K-1 ? pad_K : 0);
            rows = DIM - (j == J-1 ? pad_J : 0);
          } else {
            dram_addr = gemmini_state.loop_ws_B +
                (k*gemmini_state.loop_ws_B_stride + j) * DIM * sizeof(elem_t);
            cols = DIM - (j == J-1 ? pad_J : 0);
            rows = DIM - (k == K-1 ? pad_K : 0);
          }

          mvin(dram_addr, (rows << 48) | (cols << 32) | B_sp_addr, 1);
        }

        // Compute
        if(!is_resadd) {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          if (!ex_accumulate && k == 0) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          const uint64_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const uint64_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const uint64_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const uint64_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const uint64_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const uint64_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

          preload((B_rows << 48) | (B_cols << 32) | pre_sp_addr,
              (C_rows << 48) | (C_cols << 32) | out_sp_addr);

          compute((A_rows << 48) | (A_cols << 32) | A_sp_addr,
              ((uint64_t)DIM << 48) | ((uint64_t)DIM << 32) | GARBAGE_ADDR, i == 0);
        }

        // Move-out C
        if (gemmini_state.loop_ws_C != 0 && k == K-1) {
          const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

          if ((act != gemmini_state_t::LAYERNORM) && (act != gemmini_state_t::SOFTMAX)) {
            const uint64_t C_dram_addr = gemmini_state.loop_ws_C +
                (i*gemmini_state.loop_ws_C_stride + j) * DIM * sizeof_C;

            const uint64_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
            const uint64_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

            mvout(C_dram_addr, (C_rows << 48) | (C_cols << 32) | C_sp_addr);
          }

          else if (((act == gemmini_state_t::LAYERNORM) || (act == gemmini_state_t::SOFTMAX)) && j == J - 1){
            int norm_cmds_size;
            uint32_t (*norm_cmds)[2];
            uint32_t ln_norm_cmds[][2] = {{1,2},{3,4},{0,0}};
            uint32_t sm_norm_cmds[][2] = {{5,5},{6,7},{0,0}};

            if (act == gemmini_state_t::LAYERNORM) {
              norm_cmds = ln_norm_cmds;
              norm_cmds_size = 3;
            } else {
              norm_cmds = sm_norm_cmds;
              norm_cmds_size = 3;
            }

            const size_t rows = DIM - (i == I-1 ? pad_I : 0);

            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
              const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                NORM_STAT_IDS : rows - row;

              for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                  config((1 << 17) | (stat_id << 8) | 3, 0);
                  const size_t r = row + stat_id;

                  for (size_t jj = 0; jj < J; jj++) {
                    uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                    if (jj + 1 == J) {
                      norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                    } else {
                      norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                    }

                    const uint64_t C_dram_addr = gemmini_state.loop_ws_C +
                      (i*gemmini_state.loop_ws_C_stride + jj) * DIM * sizeof_C +
                      r * gemmini_state.loop_ws_C_stride * sizeof_C;

                    const size_t cols = DIM - (jj + 1 == J ? pad_J : 0);

                    // gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                    mvout(C_dram_addr, ((uint64_t)1 << 48) | (cols << 32) | norm_C_sp_addr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void gemmini_t::compute_cisc() {
  // `compute` performs Gemmini's core function - matrix multiply-add -
  //  without referencing any underlying hardware detail.
  //
  // * Operands A, B, and D are loaded from memory
  // * Multiply, add, activation, and any requested shifts are performed
  // * Result D is written back to memory
  //
  // These computations are made independent of systolic array sizes,
  // scratchpad-memory sizes,
  // and any other microarchitectural detail (other than datatypes).

  // Load operands from memory
  auto A = read_matrix_from_dram<elem_t>(gemmini_state.a_addr,
                                          gemmini_state.m,
                                          gemmini_state.k,
                                          false, false);
  auto B = read_matrix_from_dram<elem_t>(gemmini_state.b_addr,
                                          gemmini_state.k,
                                          gemmini_state.n,
                                          false, false);
  auto D = read_matrix_from_dram<acc_t>(gemmini_state.d_addr,
                                          gemmini_state.m,
                                          gemmini_state.n,
                                          true,
                                          gemmini_state.repeating_bias);
  // Initialize an accumulator/ result
  auto C = matrix_zeroes<elem_t>(gemmini_state.m, gemmini_state.n);

  // Multiply & apply activation
  for (size_t i=0; i<gemmini_state.m; i++) {
    for (size_t j=0; j<gemmini_state.n; j++) {
      acc_t value = D->at(i).at(j);
      for (size_t k=0; k<gemmini_state.k; k++) {
        value += ((acc_t)A->at(i).at(k)) * ((acc_t)B->at(k).at(j));
      }
      elem_t shifted = acc_scale(value,
                          gemmini_state.acc_shift);
      elem_t activated = apply_activation_acc(shifted);
      C->at(i).at(j) = activated;
    }
  }

  // Write back to memory
  for (size_t i = 0; i < gemmini_state.m; i++) {
    auto const dram_row_addr = gemmini_state.c_addr +
                               i*sizeof(elem_t)*gemmini_state.n;
    for (size_t j = 0; j < gemmini_state.n; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
#ifdef ELEM_T_IS_FLOAT
      write_to_dram<elem_t_bits>(dram_byte_addr, C->at(i).at(j));
#else
      write_to_dram<elem_t>(dram_byte_addr, C->at(i).at(j));
#endif
    }
  }
}

// Union for counter operation argument extraction
union counter_op_param {
  reg_t arg;
  struct {
    uint64_t counter_reset:1;
    uint64_t snapshot_reset:1;
    uint64_t take_snapshot:1;
    uint64_t change_config:1;
    uint64_t counter_index:3;
    uint64_t padding2:5;
    uint64_t counter_addr:6;
    uint64_t padding1:13;
    uint64_t external_counter:1;
    uint64_t padding0:32;
  };
};

reg_t gemmini_t::counter_operation(reg_t rs1) {
  counter_op_param decoder;
  decoder.arg = rs1;

  if (decoder.counter_reset) {
    for (size_t i = 0; i < NUM_COUNTERS; i++)
      gemmini_state.counter_val[i] = 0;
    gemmini_state.op_in_progress = false;
  }
  if (decoder.snapshot_reset) gemmini_state.snapshot_enable = false;
  if (decoder.take_snapshot) {
    gemmini_state.snapshot_enable = true;
    for (size_t i = 0; i < NUM_COUNTERS; i++) {
      if (gemmini_state.counter_external_flag[i])
        gemmini_state.counter_snapshot_val[i] = gemmini_state.counter_external[gemmini_state.counter_config[i]];
      else
        gemmini_state.counter_snapshot_val[i] = gemmini_state.counter_val[i];
    }
  }
  if (decoder.change_config) {
    gemmini_state.counter_config[decoder.counter_index] = decoder.counter_addr;
    gemmini_state.counter_val[decoder.counter_index] = 0;
    gemmini_state.counter_external_flag[decoder.counter_index] = decoder.external_counter;
  }
  if (gemmini_state.snapshot_enable)
    return gemmini_state.counter_snapshot_val[decoder.counter_index];
  else if (gemmini_state.counter_external_flag[decoder.counter_index])
    return gemmini_state.counter_external[gemmini_state.counter_config[decoder.counter_index]];
  else
    return gemmini_state.counter_val[decoder.counter_index];
}

void gemmini_t::loop_ws_config_bounds(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_ws_I = rs2 & 0xFFFF;
  gemmini_state.loop_ws_J = (rs2 >> 16) & 0xFFFF;
  gemmini_state.loop_ws_K = (rs2 >> 32) & 0xFFFF;

  gemmini_state.loop_ws_pad_I = rs1 & 0xFFFF;
  gemmini_state.loop_ws_pad_J = (rs1 >> 16) & 0xFFFF;
  gemmini_state.loop_ws_pad_K = (rs1 >> 32) & 0xFFFF;
}

void gemmini_t::loop_ws_config_addrs_AB(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_ws_A = rs1;
  gemmini_state.loop_ws_B = rs2;
}

void gemmini_t::loop_ws_config_addrs_DC(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_ws_D = rs1;
  gemmini_state.loop_ws_C = rs2;
}

void gemmini_t::loop_ws_config_strides_AB(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_ws_A_stride = rs1;
  gemmini_state.loop_ws_B_stride = rs2;
}

void gemmini_t::loop_ws_config_strides_DC(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_ws_D_stride = rs1;
  gemmini_state.loop_ws_C_stride = rs2;
}

void gemmini_t::loop_conv_ws(reg_t rs1, reg_t rs2) {
  const bool no_bias = rs1 & 1;
  const bool wrot180 = (rs1 >> 1) & 1;
  const bool trans_output_1203 = (rs1 >> 2) & 1;
  const bool trans_weight_1203 = (rs1 >> 3) & 1;
  const bool trans_weight_0132 = (rs1 >> 4) & 1;
  const bool trans_input_3120 = (rs1 >> 5) & 1;
  const bool dw = (rs1 >> 6) & 1;
  uint8_t max_pixels_per_row = (rs1 >> 8) & 0xFF;
  const bool no_pool = rs2 & 1;
  const bool downsample = (rs2 >> 1) & 1;
  const bool input_dilated = (rs2 >> 2) & 1;
  const bool activation = (rs2 >> 3) & 3;
  uint8_t a_spad_id = (rs1 >> 18) & 0b11;
  uint8_t b_spad_id = (rs1 >> 16) & 0b11;

  if (max_pixels_per_row == 0)
    max_pixels_per_row = 1;

#ifndef HAS_FIRST_LAYER_OPTIMIZATIONS
  assert(max_pixels_per_row <= 1, "If Gemmini is built without first-layer optimizations, then 'max_pixels_per_row' cannot be larger than 1\n");
#endif

  const uint16_t batch_size = gemmini_state.loop_conv_ws_batch_size;
  const uint16_t in_col_dim = gemmini_state.loop_conv_ws_in_col_dim;
  const uint16_t in_row_dim = gemmini_state.loop_conv_ws_in_row_dim;
  const uint16_t in_channels = gemmini_state.loop_conv_ws_in_channels;
  const uint16_t out_channels = gemmini_state.loop_conv_ws_out_channels;
  const uint16_t in_stride = gemmini_state.loop_conv_ws_in_stride;
  const uint16_t out_stride = gemmini_state.loop_conv_ws_out_stride;
  const uint16_t weight_stride = gemmini_state.loop_conv_ws_weight_stride;
  const uint16_t out_col_dim = gemmini_state.loop_conv_ws_out_col_dim;
  const uint16_t pool_out_col_dim = gemmini_state.loop_conv_ws_pool_out_col_dim;
  const uint16_t out_row_dim = gemmini_state.loop_conv_ws_out_row_dim;
  const uint16_t pool_out_row_dim = gemmini_state.loop_conv_ws_pool_out_row_dim;
  const uint16_t stride = gemmini_state.loop_conv_ws_stride;
  const uint16_t padding = gemmini_state.loop_conv_ws_padding;
  const uint16_t kernel_dim = gemmini_state.loop_conv_ws_kernel_dim;
  const uint16_t kernel_dilation = gemmini_state.loop_conv_ws_kernel_dilation;
  const uint16_t pool_size = gemmini_state.loop_conv_ws_pool_size;
  const uint16_t pool_stride = gemmini_state.loop_conv_ws_pool_stride;
  const uint16_t pool_padding = gemmini_state.loop_conv_ws_pool_padding;
  const uint16_t batches = gemmini_state.loop_conv_ws_batches;
  const uint16_t porows = gemmini_state.loop_conv_ws_porows;
  const uint16_t pocols = gemmini_state.loop_conv_ws_pocols;
  const uint16_t pochs = gemmini_state.loop_conv_ws_pochs;
  const uint16_t krows = gemmini_state.loop_conv_ws_krows;
  const uint16_t kcols = gemmini_state.loop_conv_ws_kcols;
  const uint16_t kchs = gemmini_state.loop_conv_ws_kchs;
  const uint16_t lpad = gemmini_state.loop_conv_ws_lpad;
  const uint16_t rpad = gemmini_state.loop_conv_ws_rpad;
  const uint16_t upad = gemmini_state.loop_conv_ws_upad;
  const uint16_t dpad = gemmini_state.loop_conv_ws_dpad;
  const uint16_t plpad = gemmini_state.loop_conv_ws_plpad;
  const uint16_t prad = gemmini_state.loop_conv_ws_prad;
  const uint16_t pupad = gemmini_state.loop_conv_ws_pupad;
  const uint16_t pdpad = gemmini_state.loop_conv_ws_pdpad;
  const uint16_t orows = gemmini_state.loop_conv_ws_orows;
  const uint16_t ocols = gemmini_state.loop_conv_ws_ocols;
  const uint64_t weights = gemmini_state.loop_conv_ws_weights;
  const uint64_t output = gemmini_state.loop_conv_ws_output;
  const uint64_t bias = gemmini_state.loop_conv_ws_bias;
  const uint64_t input = gemmini_state.loop_conv_ws_input;

  const uint16_t ochs = pochs;

  assert(!input_dilated || stride == 1);

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int16_t dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int16_t dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  const int16_t irows_without_dilation = orows * stride + dilated_krows - 1; // - 2 * padding;
  const int16_t icols_without_dilation = ocols * stride + dilated_kcols - 1; // - 2 * padding;
  const int16_t irows_unpadded_without_dilation = irows_without_dilation - upad - dpad;
  const int16_t icols_unpadded_without_dilation = icols_without_dilation - lpad - rpad;
  const int16_t ichs = kchs;

#define UNDILATED(x) (((x) + (input_dilated)) >> (input_dilated))

  const int16_t irows_unpadded = input_dilated ? (irows_unpadded_without_dilation+1)/2 : irows_unpadded_without_dilation;
  const int16_t icols_unpadded = input_dilated ? (icols_unpadded_without_dilation+1)/2 : icols_unpadded_without_dilation;

  const int16_t irows = input_dilated ? irows_unpadded + UNDILATED(upad) + UNDILATED(dpad) : irows_without_dilation;
  const int16_t icols = input_dilated ? icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad) : icols_without_dilation;

  const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

  static uint32_t D_sp_addr_row = 0;
  static uint32_t C_sp_addr_row = 0;

  uint32_t A_sp_addr_start = 0;
  uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS / 2 - B_rows;
  const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;
  if(a_spad_id == 2) A_sp_addr_start = (BANK_NUM * BANK_ROWS) / 2;
  if(b_spad_id == 2) B_sp_addr_start = (BANK_NUM * BANK_ROWS) - B_rows;
  if (bias != 0) {
    D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if (output != 0) {
    C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  const uint32_t GARBAGE_ADDR = ~0;

  // mvin bias
  if (bias != 0) {
    // TODO we probably don't need quite this many nested loops for this part

    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
      MAX_BLOCK_LEN_ACC * DIM;

    // gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);
    config(((uint64_t)scale_t_to_scale_t_bits(MVIN_SCALE_IDENTITY) << 32) |
      ((uint64_t)(batches * orows * ocols) << 16) |
      (1 << 8) |
      (2 << 3) | 1,
      0);

    for (int b = 0; b < batches; b++)
      for (int orow = 0; orow < orows; orow++)
        for (int ocol = 0; ocol < ocols; ocol += DIM) {
          const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

          for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
            const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

            const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

            mvin(no_bias ? 0 : bias + och * sizeof(acc_t),
              ((uint64_t)I << 48) | ((uint64_t)J << 32) | D_sp_addr,
              2);
          }
        }
  }

#define DS(x) ((x) >> (downsample))
#define US(x) ((x) << (downsample))

  // mvin input
  if(input != 0){
    int16_t max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
      MAX_BLOCK_LEN * DIM;
    if (trans_input_3120) {
      max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
        MAX_BLOCK_LEN * DIM;
    }

    const uint32_t dram_stride = trans_input_3120 ?
      batch_size * sizeof(elem_t) :
      in_stride * sizeof(elem_t);

    const int spad_stride = trans_input_3120 ?
      ichs * DS(irows) * DS(icols) :
      batches * DS(irows) * DS(icols);

    // gemmini_extended5_config_ld(dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);
    config(((uint64_t)scale_t_to_scale_t_bits(MVIN_SCALE_IDENTITY) << 32) |
      ((uint64_t)(spad_stride) << 16) |
      ((uint64_t)(max_pixels_per_row) << 8) |
      (0 << 3) | 1,
      US(dram_stride));

    const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
    const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

    for (int16_t b = 0; b < batches; b += b_it)
      for (int16_t irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += US(1)) {
        const int16_t irow_padded = irow + UNDILATED(upad);

        for (int16_t icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
          // TODO There might be some unnecessary mvins here at the edge of the image

          int16_t I = icols_unpadded - icol > US(DIM) ? US(DIM) : icols_unpadded - icol;

          if (icol < 0) {
            I = -icol > DIM ? DIM : -icol;
          } else if (icol >= icols_unpadded) {
            I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
          }

          const int16_t icol_padded = icol + UNDILATED(lpad);

          for (int16_t ich = 0; ich < ichs; ich += ich_it) {
            int16_t K = ichs - ich > max_chs_per_mvin ?
              max_chs_per_mvin : ichs - ich;
            if (trans_input_3120) {
              K = batches - b > max_chs_per_mvin ?
                max_chs_per_mvin : batches - b;
            }

            uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * spad_stride + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            if (trans_input_3120) {
              A_sp_addr = A_sp_addr_start + (b / DIM) * spad_stride + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            }

            const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

            uint64_t in = input + ((b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_stride + ich) * sizeof(elem_t);
            if (is_zeros) {
              in = 0;
            } else if (trans_input_3120) {
              in = input + ((ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b) * sizeof(elem_t);
            }

            // gemmini_extended_mvin(in,
            //     A_sp_addr,
            //     K, I);
            mvin(in,
              ((uint64_t)(DS(I)) << 48) | ((uint64_t)K << 32) | A_sp_addr,
              0);
          }

          icol += I;
        }
      }
  }

  // mvin weights
  if (weights != 0){
    uint16_t max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
      MAX_BLOCK_LEN * DIM;
    if (trans_weight_0132) {
      max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
          MAX_BLOCK_LEN * DIM;
    }

    size_t dram_stride = weight_stride * sizeof(elem_t);
    if (dw) {
      dram_stride = sizeof(elem_t);
    } else if (trans_weight_1203) {
      dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
    } else if (trans_weight_0132) {
      dram_stride = in_channels * sizeof(elem_t);
    }

    const size_t spad_block_stride = trans_weight_0132 ?
      krows * kcols * ochs : krows * kcols * kchs;

    // gemmini_extended4_config_ld(out_channels * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, krows * kcols * kchs, 1);
    config(((uint64_t)scale_t_to_scale_t_bits(MVIN_SCALE_IDENTITY) << 32) |
      ((uint64_t)(spad_block_stride) << 16) |
      (1 << 8) |
      (1 << 3) | 1,
      dram_stride);

    const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
    const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

    for (uint16_t och = 0; och < ochs; och += och_it) {
      for (uint16_t krow = 0; krow < krows; krow++)
        for (uint16_t kcol = 0; kcol < kcols; kcol++)
          for (uint16_t kch = 0; kch < kchs; kch += kch_it) {
            uint16_t K = kchs - kch > DIM ? DIM : kchs - kch;
            uint16_t J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
            if (trans_weight_0132) {
              K = ochs - och > DIM ? DIM : ochs - och;
              J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
            }

            uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
            if (trans_weight_0132) {
              B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
            }

            auto w = weights + ((krow*kernel_dim*in_channels + kcol*in_channels + kch) * weight_stride + och)*sizeof(elem_t);
            if (dw) {
              w = (weights + krow * kernel_dim + kcol)*sizeof(elem_t);
            } else if (trans_weight_1203) {
              w = weights + ((kch * kernel_dim * kernel_dim  + krow * kernel_dim + kcol) * out_channels + och)*sizeof(elem_t);
            } else if (trans_weight_0132) {
              w = weights + ((krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch)*sizeof(elem_t);
            }

            // gemmini_extended_mvin2(w,
            //   B_sp_addr,
            //   J, K);

            mvin(w,
              ((uint64_t)K << 48) | ((uint64_t)J << 32) | B_sp_addr,
              1);
          }
    }
  }

  // Compute
  {
    const int b_it = trans_input_3120 ? DIM : 1;
    const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

    if (trans_input_3120) {
      // gemmini_extended3_config_ex(0, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true);
      const uint16_t a_stride = irows * icols;
      const uint16_t c_stride = orows * ocols;
      const bool set_only_strides = true;
      config(((uint64_t)a_stride << 16) | (set_only_strides << 7) | 0,
        ((uint64_t)c_stride << 48));
    }

    for (uint16_t och = 0; och < ochs; och += DIM) {
      for (uint16_t krow = 0; krow < krows; krow++) {
        for (uint16_t kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
          for (uint16_t kch = 0; kch < kchs; kch += DIM) {
            bool new_weights = true;

            for (uint16_t b = 0; b < batches; b += b_it) {
              for (uint16_t orow = 0; orow < orows; orow++) {
                if (input_dilated && ((krow * kernel_dilation + orow - upad) % 2 != 0)) {
                  continue;
                }

                for (uint16_t ocol = 0; ocol < ocols;) {
                  if (input_dilated && ((kcol * kernel_dilation + ocol - lpad) % 2 != 0)) {
                    ocol++;
                    continue;
                  }

                  const uint16_t irow = UNDILATED(orow * stride + krow * kernel_dilation);
                  const uint16_t icol = UNDILATED(ocol * stride + kcol * kernel_dilation);

                  const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

                  const uint8_t pixels = kcols - kcol > max_pixels_per_row ?
                    max_pixels_per_row : kcols - kcol;

                  // Over here, construct a new matrix
                  //
                  // Let us assume that we only ever operate on
                  // one pixel in one row.
                  // Thus, krow == kcol == 1
                  //
                  // Then, for every set of I, J, and K values
                  //   - I = ocol
                  //   - J = och
                  //   - K = kch

                  uint16_t I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
                  const uint16_t J = ochs - och > DIM ? DIM : ochs - och;
                  const uint16_t K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

                  if (trans_input_3120) {
                    I = batches - b > DIM ? DIM : batches - b;
                  }

                  uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  if (trans_input_3120) {
                    A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  }

                  const int krow_ = wrot180 ? krows - krow - 1 : krow;
                  const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

                  uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
                  if (trans_weight_0132) {
                    B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
                  }

                  const uint32_t pre_sp_addr = new_weights ? B_sp_addr : GARBAGE_ADDR;

                  // perform matmul
                  const uint32_t out_sp_addr = C_sp_addr;

                  /*
                  gemmini_extended_preload(pre_sp_addr, out_sp_addr,
                          J, K, J, I);

                  if (new_weights) {
                    gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  } else {
                    gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  }
                  */

                  preload(((uint64_t)K << 48) | ((uint64_t)J << 32) | pre_sp_addr,
                      ((uint64_t)I << 48) | ((uint64_t)J << 32) | out_sp_addr);
                  compute(((uint64_t)I << 48) | ((uint64_t)K << 32) | A_sp_addr,
                      ((uint64_t)I << 48) | ((uint64_t)J << 32) | GARBAGE_ADDR, new_weights);

                  ocol += ocol_it;
                  new_weights = false;
                }
              }
            }
          }
        }
      }
    }
  }

#undef DS
#undef US
#undef UNDILATED

  // Mvout results
  if (output != 0 && no_pool) {
    for (uint16_t b = 0; b < batches; b++)
      for (uint16_t orow = 0; orow < orows; orow++)
        for (uint16_t ocol = 0; ocol < ocols; ocol += DIM) {
          const uint16_t I = ocols - ocol > DIM ? DIM : ocols - ocol;

          for (uint16_t och = 0; och < ochs; och += DIM) {
            const uint16_t J = ochs - och > DIM ? DIM : ochs - och;

            const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

            auto out = output + ((b*out_row_dim*out_col_dim + orow*out_col_dim + ocol) * out_stride + och) * sizeof(elem_t);
            if (trans_output_1203) {
              out = output + ((orow*out_col_dim*batch_size + ocol*batch_size + b) * out_channels + och) * sizeof(elem_t);
            }

            mvout(out,
                ((uint64_t)I << 48) | ((uint64_t)J << 32) | C_sp_addr);
          }
        }
  } else if (output != 0 && !no_pool) {
    auto acc_scale = gemmini_state.acc_shift;

    // gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
    config(
      ((uint64_t)ocols << 56) |
      ((uint64_t)orows << 48) |
      ((uint64_t)pocols << 40) |
      ((uint64_t)porows << 32) |
      ((uint64_t)pool_out_col_dim << 24) |
      ((uint64_t)plpad << 10) |
      ((uint64_t)pupad << 8) |
      ((uint64_t)pool_size << 6) |
      ((uint64_t)pool_stride << 4) |
      (activation << 2) |
      2,
      ((uint64_t)(acc_scale_t_to_acc_scale_t_bits(acc_scale)) << 32) |
      (out_stride * sizeof(elem_t)));

    for (int b = 0; b < batches; b++) {
      for (int poch = 0; poch < pochs; poch += DIM) {
        const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

        const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

        mvout(output + ((b * pool_out_row_dim * pool_out_col_dim)*out_stride + poch) * sizeof(elem_t),
          ((uint64_t)channels << 32) | C_sp_addr);
      }
    }

    // gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
    config(
      (activation << 2) |
      2,
      ((uint64_t)(acc_scale_t_to_acc_scale_t_bits(acc_scale)) << 32) |
      out_stride * sizeof(elem_t));
  }
}

void gemmini_t::loop_conv_ws_config_1(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_batch_size = rs1 & 0xFFFF;
  gemmini_state.loop_conv_ws_in_row_dim = (rs1 >> 16) & 0xFFFF;
  gemmini_state.loop_conv_ws_in_channels = (rs1 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_out_channels = (rs1 >> 48) & 0xFFFF;

  gemmini_state.loop_conv_ws_out_row_dim = rs2 & 0xFFFF;
  gemmini_state.loop_conv_ws_pool_out_row_dim = (rs2 >> 16) & 0xFFFF;
  gemmini_state.loop_conv_ws_out_col_dim = (rs2 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_stride = (rs2 >> 48) & 0xFF;
  gemmini_state.loop_conv_ws_padding = (rs2 >> 56) & 0xFF;
}

void gemmini_t::loop_conv_ws_config_2(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_kernel_dim = (rs1 >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_pool_out_col_dim = (rs1 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_pool_size = (rs1 >> 16) & 0xFFFF;
  gemmini_state.loop_conv_ws_pool_stride = (rs1 >> 8) & 0xFF;
  gemmini_state.loop_conv_ws_pool_padding = rs1 & 0xFF;

  gemmini_state.loop_conv_ws_batches = (rs2 >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_porows = (rs2 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_pocols = (rs2 >> 16)  & 0xFFFF;
  gemmini_state.loop_conv_ws_pochs = rs2 & 0xFFFF;
}

void gemmini_t::loop_conv_ws_config_3(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_krows = (rs1 >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_kcols = (rs1 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_kchs = (rs1 >> 16) & 0xFFFF;
  gemmini_state.loop_conv_ws_lpad = rs1 & 0xFFFF;

  gemmini_state.loop_conv_ws_rpad = (rs2 >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_upad = (rs2 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_dpad = (rs2 >> 24) & 0xFF;
  gemmini_state.loop_conv_ws_plpad = (rs2 >> 16) & 0xFF;
  gemmini_state.loop_conv_ws_in_col_dim = rs2 & 0xFFFF;
}

void gemmini_t::loop_conv_ws_config_4(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_orows = (rs1  >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_prad = (rs1 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_pupad = (rs1 >> 21) & 0xFF;
  gemmini_state.loop_conv_ws_pdpad = (rs1 >> 10) & 0xFF;
  gemmini_state.loop_conv_ws_kernel_dilation = rs1 & 0xFF;

  gemmini_state.loop_conv_ws_in_stride = (rs2 >> 48) & 0xFFFF;
  gemmini_state.loop_conv_ws_weight_stride = (rs2 >> 32) & 0xFFFF;
  gemmini_state.loop_conv_ws_out_stride = (rs2 >> 16) & 0xFFFF;
  gemmini_state.loop_conv_ws_ocols = rs2 & 0xFFFF;
}

void gemmini_t::loop_conv_ws_config_5(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_weights = rs1;
  gemmini_state.loop_conv_ws_output = rs2;
}

void gemmini_t::loop_conv_ws_config_6(reg_t rs1, reg_t rs2) {
  gemmini_state.loop_conv_ws_bias = rs1;
  gemmini_state.loop_conv_ws_input = rs2;
}

reg_t gemmini_t::CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  if (!gemmini_state.resetted) {
    reset();
  }
  if (gemmini_state.op_in_progress)
    counter_increment_random();

  if (insn.funct == mvin_funct) {
    mvin(xs1, xs2, 0);
  } else if (insn.funct == mvin2_funct) {
    mvin(xs1, xs2, 1);
  } else if (insn.funct == mvin3_funct) {
    mvin(xs1, xs2, 2);
  } else if (insn.funct == mvout_funct) {
    mvout(xs1, xs2);
  } else if (insn.funct == preload_funct) {
    preload(xs1, xs2);
  } else if (insn.funct == config_funct) {
    config(xs1, xs2);
  } else if (insn.funct == compute_preloaded_funct) {
    compute(xs1, xs2, true);
  } else if (insn.funct == compute_accumulated_funct) {
    compute(xs1, xs2, false);
  } else if (insn.funct == loop_ws_config_bounds_funct) {
    loop_ws_config_bounds(xs1, xs2);
  } else if (insn.funct == loop_ws_config_addrs_AB_funct) {
    loop_ws_config_addrs_AB(xs1, xs2);
  } else if (insn.funct == loop_ws_config_addrs_DC_funct) {
    loop_ws_config_addrs_DC(xs1, xs2);
  } else if (insn.funct == loop_ws_config_strides_AB_funct) {
    loop_ws_config_strides_AB(xs1, xs2);
  } else if (insn.funct == loop_ws_config_strides_DC_funct) {
    loop_ws_config_strides_DC(xs1, xs2);
  } else if (insn.funct == loop_ws_funct) {
    loop_ws(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_1_funct) {
    loop_conv_ws_config_1(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_2_funct) {
    loop_conv_ws_config_2(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_3_funct) {
    loop_conv_ws_config_3(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_4_funct) {
    loop_conv_ws_config_4(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_5_funct) {
    loop_conv_ws_config_5(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_config_6_funct) {
    loop_conv_ws_config_6(xs1, xs2);
  } else if (insn.funct == loop_conv_ws_funct) {
    loop_conv_ws(xs1, xs2);
  } else if (insn.funct == counter_op_funct) {
    return counter_operation(xs1);
  }
  //==========================================================================
  // gemmini-cisc opcodes
  //==========================================================================
  /*
  else if (insn.funct == config_cisc_ex_funct) {
    config(xs1, xs2);
  }
  else if (insn.funct == config_addr_AB_funct) {
    gemmini_state.a_addr = xs1;
    gemmini_state.b_addr = xs2;
  }
  else if (insn.funct == config_addr_CD_funct ){
    gemmini_state.c_addr = xs1;
    gemmini_state.d_addr = xs2;
  }
  else if (insn.funct == config_size0_funct ){
    gemmini_state.m = xs1;
    gemmini_state.n = xs2;
  }
  else if (insn.funct == config_size1_funct ){
    gemmini_state.k = xs1;
  }
  else if (insn.funct == config_repeating_bias_funct){
    gemmini_state.repeating_bias = (bool)xs1;
  }
  else if (insn.funct == config_reset_funct) {
    reset();
  }
  else if (insn.funct == compute_cisc_funct) {
    compute_cisc();
  }
  */
  //==========================================================================
  else if (insn.funct == flush_funct) {
    dprintf("GEMMINI: flush\n");
  } else if (insn.funct == fence_funct) {
    dprintf("GEMMINI: fence\n");
  } else {
    dprintf("GEMMINI: encountered unknown instruction with funct: %d\n", insn.funct);
    illegal_instruction(*this->p);
  }
  gemmini_state.op_in_progress = (insn.funct != flush_funct);
  return 0;
}

#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
scale_t_bits gemmini_t::scale_t_to_scale_t_bits(scale_t scale) {
    union {
        scale_t scale;
        scale_t_bits bits;
    } un;

    un.scale = scale;
    return un.bits;
}

scale_t gemmini_t::scale_t_bits_to_scale_t(scale_t_bits bits) {
    union {
        scale_t scale;
        scale_t_bits bits;
    } un;

    un.bits = bits;
    return un.scale;
}
#endif

acc_scale_t_bits gemmini_t::acc_scale_t_to_acc_scale_t_bits(acc_scale_t scale) {
    union {
        acc_scale_t scale;
        acc_scale_t_bits bits;
    } un;

    un.scale = scale;
    return un.bits;
}

acc_scale_t gemmini_t::acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits bits) {
    union {
        acc_scale_t scale;
        acc_scale_t_bits bits;
    } un;

    un.bits = bits;
    return un.scale;
}

// Applying activation from PE post-shifted output to scratchpad (for OS dataflow)
// or from accumulator to DRAM (after shifting, for WS dataflow)
elem_t gemmini_t::apply_activation(elem_t value, enum gemmini_state_t::Activation act) {
  if (act == gemmini_state_t::RELU) {
    return value > 0 ? static_cast<elem_t>(value) : static_cast<elem_t>(0);
  } else {
    return static_cast<elem_t>(value);
  }
}

elem_t gemmini_t::apply_activation_sys(elem_t value) {
  return apply_activation(value, gemmini_state.sys_act);
}

elem_t gemmini_t::apply_activation_acc(elem_t value) {
  return apply_activation(value, gemmini_state.acc_act);
}

acc_t apply_igelu(acc_t q, acc_t qb, acc_t qc) {
  const acc_t q_sign = q < 0 ? -1 : 1;
  const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
  const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
  const acc_t q_erf = q_sign * q_poly;

  return q * (q_erf + qc);
}

acc_t apply_iexp(acc_t q, acc_t qb, acc_t qc, acc_t qln2, acc_t qln2_inv) {
#ifdef ELEM_T_IS_FLOAT
  const acc_t z = (acc_t) (-q * qln2_inv) / (1 << 16);
#else
  const acc_t z = (acc_t) (-q * qln2_inv) >> 16;
#endif
  const acc_t qp = q + z * qln2;
  const acc_t q_exp = (qp + qb) * (qp + qb) + qc;
#ifdef ELEM_T_IS_FLOAT
  return q_exp / std::pow(2, z);
#else
  return q_exp >> z;
#endif
}

acc_t gemmini_t::apply_pre_activation_acc(acc_t value) {
  if (gemmini_state.acc_act == gemmini_state_t::IGELU) {
    return apply_igelu(value, gemmini_state.igelu_qb, gemmini_state.igelu_qc);
  } else if (gemmini_state.acc_act == gemmini_state_t::LAYERNORM) {
    const auto stat_id = gemmini_state.norm_stat_id;
    const auto norm_mean = gemmini_state.norm_mean[stat_id];
    const auto norm_inv_stddev = gemmini_state.norm_inv_stddev[stat_id];
    return (value - norm_mean) * norm_inv_stddev;
  } else if (gemmini_state.acc_act == gemmini_state_t::SOFTMAX) {
    const auto stat_id = gemmini_state.norm_stat_id;
    const auto norm_max = gemmini_state.norm_max[stat_id];
    const auto norm_inv_sum_exp = gemmini_state.norm_inv_sum_exp[stat_id];
    return acc_scale(apply_iexp(value - norm_max, gemmini_state.igelu_qb, gemmini_state.igelu_qc,
        gemmini_state.qln2, gemmini_state.qln2_inv), norm_inv_sum_exp);
  } else {
    return value;
  }
}

bool gemmini_t::apply_norm(const acc_t * x, size_t len, enum gemmini_state_t::NormCmd cmd) {
  // The return value is whether or not the command should write to main memory

  const auto stat_id = gemmini_state.norm_stat_id;

  auto& norm_sum = gemmini_state.norm_sum[stat_id];
  auto& norm_count = gemmini_state.norm_count[stat_id];
  auto& norm_mean = gemmini_state.norm_mean[stat_id];
  auto& norm_inv_stddev = gemmini_state.norm_inv_stddev[stat_id];
  auto& norm_max = gemmini_state.norm_max[stat_id];
  auto& norm_running_max = gemmini_state.norm_running_max[stat_id];
  auto& norm_inv_sum_exp = gemmini_state.norm_inv_sum_exp[stat_id];
  auto& norm_reset = gemmini_state.norm_reset[stat_id];

  if (norm_reset) {
    norm_sum = 0;
    norm_count = 0;
    norm_running_max = -2147483648;
  }

  norm_reset = cmd == gemmini_state_t::RESET || cmd == gemmini_state_t::MEAN || cmd == gemmini_state_t::INV_STDDEV;

  if (cmd == gemmini_state_t::SUM || cmd == gemmini_state_t::MEAN) {
    for (size_t i = 0; i < len; i++)
      norm_sum += x[i];
    norm_count += len;
  } else if (cmd == gemmini_state_t::VARIANCE || cmd == gemmini_state_t::INV_STDDEV) {
    for (size_t i = 0; i < len; i++)
      norm_sum += (x[i] - norm_mean)*(x[i] - norm_mean);
    norm_count += len;
  } else if (cmd == gemmini_state_t::MAX) {
    for (size_t i = 0; i < len; i++) {
      norm_running_max = x[i] > norm_running_max ? x[i] : norm_running_max;
    }
  } else if (cmd == gemmini_state_t::SUM_EXP || cmd == gemmini_state_t::INV_SUM_EXP) {
    norm_max = norm_running_max;
    for (size_t i = 0; i < len; i++) {
      norm_sum += apply_iexp(x[i] - norm_max, gemmini_state.igelu_qb, gemmini_state.igelu_qc,
          gemmini_state.qln2, gemmini_state.qln2_inv);
    }
  }

  if (cmd == gemmini_state_t::MEAN) {
    norm_mean = norm_sum / norm_count;
  } else if (cmd == gemmini_state_t::INV_STDDEV) {
    acc_t variance = norm_sum / norm_count;
    acc_t norm_stddev = sqrt((double)variance);
    if (variance == 0) norm_stddev = 1;
    norm_inv_stddev = (acc_scale_t)1.0 / (acc_scale_t)norm_stddev;
  } else if (cmd == gemmini_state_t::INV_SUM_EXP) {
    norm_running_max = -2147483648;
    norm_inv_sum_exp = ((acc_scale_t) 127.0) / norm_sum;
  }

  return cmd == gemmini_state_t::RESET;
}

enum gemmini_state_t::NormCmd gemmini_t::non_terminating_norm_cmd(enum gemmini_state_t::NormCmd cmd) {
  if (cmd == gemmini_state_t::RESET)
    cmd = gemmini_state_t::RESET;
  else if (cmd == gemmini_state_t::MEAN)
    cmd = gemmini_state_t::SUM;
  else if (cmd == gemmini_state_t::INV_STDDEV)
    cmd = gemmini_state_t::VARIANCE;
  else if (cmd == gemmini_state_t::INV_SUM_EXP)
    cmd = gemmini_state_t::SUM_EXP;
  return cmd;
}

#ifdef HAS_MVIN_SCALE
elem_t gemmini_t::mvin_scale(elem_t value, scale_t scale) {
  acc_t scaled = MVIN_SCALE(value, scale);

#ifndef ELEM_T_IS_FLOAT
  // Saturate and cast element
  const auto elem_t_max = std::numeric_limits<elem_t>::max();
  const auto elem_t_min = std::numeric_limits<elem_t>::min();
  scaled = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
#endif

  return scaled;
}
#endif

#ifdef HAS_MVIN_ACC_SCALE
acc_t gemmini_t::mvin_scale_acc(acc_t value, scale_acc_t scale) {
  acc_t scaled = MVIN_SCALE_ACC(value, scale);

#ifndef ELEM_T_IS_FLOAT
  // Saturate and cast element
  const auto elem_t_max = std::numeric_limits<acc_t>::max();
  const auto elem_t_min = std::numeric_limits<acc_t>::min();
  scaled = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
#endif

  return scaled;
}
#endif

elem_t gemmini_t::acc_scale(acc_t value, acc_scale_t scale) {
  acc_t scaled = ACC_SCALE(value, scale);

#ifndef ELEM_T_IS_FLOAT
  // Saturate and cast element
  const auto elem_t_max = std::numeric_limits<elem_t>::max();
  const auto elem_t_min = std::numeric_limits<elem_t>::min();
  scaled = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
#endif

  return scaled;
}

elem_t gemmini_t::sys_shift(acc_t value, unsigned int shift) {
  acc_t shifted = ROUNDING_RIGHT_SHIFT(value, shift);

#ifndef ELEM_T_IS_FLOAT
  // Saturate and cast element
  const auto elem_t_max = std::numeric_limits<elem_t>::max();
  const auto elem_t_min = std::numeric_limits<elem_t>::min();
  shifted = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
#endif

  return shifted;
}

#ifdef ELEM_T_IS_FLOAT
elem_t gemmini_t::elem_t_bits_to_elem_t(elem_t_bits x) {
  union {
    elem_t_bits b;
    elem_t f;
  } un;

  un.b = x;
  return un.f;
}

elem_t_bits gemmini_t::elem_t_to_elem_t_bits(elem_t x) {
  union {
    elem_t_bits b;
    elem_t f;
  } un;

  un.f = x;
  return un.b;
}

acc_t gemmini_t::acc_t_bits_to_acc_t(acc_t_bits x) {
  union {
    acc_t_bits b;
    acc_t f;
  } un;

  un.b = x;
  return un.f;
}

acc_t_bits gemmini_t::acc_t_to_acc_t_bits(acc_t x) {
  union {
    acc_t_bits b;
    acc_t f;
  } un;

  un.f = x;
  return un.b;
}
#endif

void gemmini_t::counter_increment(unsigned int counter_id) {
  for (size_t i = 0; i < NUM_COUNTERS; i++) {
    if (gemmini_state.counter_config[i] == counter_id) {
      gemmini_state.counter_val[i]++;
      break;
    }
  }
}
void gemmini_t::counter_increment_random() {
  // So basically, once any Gemmini command is executed, the counter will increments every time a Gemmini command got called
  // until we hit a counter reset
  for (size_t i = 0; i < NUM_COUNTERS; i++) {
    gemmini_state.counter_val[i] += rand() & 0x1ff; // So that the increment < 512
  }
  for (size_t i = 0; i < NUM_EXTERNAL_COUNTERS; i++) {
    gemmini_state.counter_external[i] = rand() & 0xf; // So that the increment < 16
  }
}

static reg_t gemmini_custom(processor_t* p, insn_t insn, reg_t pc) {
  gemmini_t* gemmini = static_cast<gemmini_t*>(p->get_extension("gemmini"));
  rocc_insn_union_t u;
  state_t* state = p->get_state();
  gemmini->set_processor(p);
  u.i = insn;
  reg_t xs1 = u.r.xs1 ? state->XPR[insn.rs1()] : -1;
  reg_t xs2 = u.r.xs2 ? state->XPR[insn.rs2()] : -1;
  reg_t xd = gemmini->CUSTOMFN(XCUSTOM_ACC)(u.r, xs1, xs2);
  if (u.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc+4;
}

std::vector<insn_desc_t> gemmini_t::get_instructions(const processor_t &proc)
{
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, gemmini_custom);
  return insns;
}

std::vector<disasm_insn_t*> gemmini_t::get_disasms(const processor_t *proc)
{
  std::vector<disasm_insn_t*> insns;
  return insns;
}
