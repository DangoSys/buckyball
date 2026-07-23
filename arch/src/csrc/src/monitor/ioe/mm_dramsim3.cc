// See LICENSE for license details.

#include "ioe/mm_dramsim3.h"
#include "ioe/mm.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <queue>
#include <string>

// #define DEBUG_DRAMSIM3

static std::string dramsim3_config_path(const std::string &memory_ini,
                                        const std::string &ini_dir) {
  if (memory_ini.empty())
    return ini_dir;
  if (!memory_ini.empty() && memory_ini[0] == '/')
    return memory_ini;
  if (ini_dir.empty())
    return memory_ini;
  if (ini_dir.size() >= 4 && ini_dir.substr(ini_dir.size() - 4) == ".ini")
    return ini_dir;
  if (ini_dir.back() == '/')
    return ini_dir + memory_ini;
  return ini_dir + "/" + memory_ini;
}

void mm_dramsim3_t::read_complete(uint64_t address) {
  assert(!rreq[address].empty());
  auto req = rreq[address].front();
  uint64_t start_addr = (req.addr / word_size) * word_size;
  for (size_t i = 0; i < req.len; i++) {
    auto dat = read(start_addr + i * word_size);
    rresp.push(mm_rresp_t(req.id, dat, (i == req.len - 1)));
  }
  read_id_busy[req.id] = false;
  rreq[address].pop();
}

void mm_dramsim3_t::write_complete(uint64_t address) {
  assert(!wreq[address].empty());
  auto b_id = wreq[address].front();
  bresp.push(b_id);
  write_id_busy[b_id] = false;
  wreq[address].pop();
}

mm_dramsim3_t::mm_dramsim3_t(size_t mem_base, size_t mem_sz, size_t word_sz,
                             size_t line_sz, backing_data_t &dat,
                             std::string memory_ini, std::string ini_dir,
                             int axi4_ids, size_t clock_hz)
    : mm_t(mem_base, mem_sz, word_sz, line_sz, dat),
      read_id_busy(axi4_ids, false), write_id_busy(axi4_ids, false) {

  assert(line_sz == 64); // assumed by dramsim3
  assert(mem_sz % (1024 * 1024) == 0);
  (void)clock_hz;
  auto config_file = dramsim3_config_path(memory_ini, ini_dir);
  mem = dramsim3::GetMemorySystem(
      config_file, "results",
      [this](uint64_t address) { this->read_complete(address); },
      [this](uint64_t address) { this->write_complete(address); });
};

bool mm_dramsim3_t::ar_ready() { return ar_ready_cache; }

bool mm_dramsim3_t::aw_ready() { return aw_ready_cache && !store_inflight; }

void mm_dramsim3_t::tick(bool reset,

                         bool ar_valid, uint64_t ar_addr, uint64_t ar_id,
                         uint64_t ar_size, uint64_t ar_len,

                         bool aw_valid, uint64_t aw_addr, uint64_t aw_id,
                         uint64_t aw_size, uint64_t aw_len,

                         bool w_valid, uint64_t w_strb, void *w_data,
                         bool w_last,

                         bool r_ready, bool b_ready) {
  ar_ready_cache = !reset && mem->WillAcceptTransaction(ar_addr, false);
  aw_ready_cache =
      !reset && !store_inflight && mem->WillAcceptTransaction(aw_addr, true);

  bool ar_fire = ar_valid && ar_ready_cache;
  bool aw_fire = aw_valid && aw_ready_cache;
  bool w_fire = !reset && w_valid && w_ready();
  bool r_fire = !reset && r_valid() && r_ready;
  bool b_fire = !reset && b_valid() && b_ready;

  for (auto it = rreq_queue.begin(); it != rreq_queue.end(); it++) {
    if (!read_id_busy[it->id] && mem->WillAcceptTransaction(it->addr, false)) {
      read_id_busy[it->id] = true;
      auto transaction = *it;
      rreq[transaction.addr].push(transaction);
      mem->AddTransaction(transaction.addr, false);
      rreq_queue.erase(it);
      break;
    }
  }

  if (ar_fire) {
    rreq_queue.push_back(mm_req_t(ar_id, 1 << ar_size, ar_len + 1, ar_addr));
  }

  if (aw_fire) {
    store_addr = aw_addr;
    store_id = aw_id;
    store_count = aw_len + 1;
    store_size = 1 << aw_size;
    store_inflight = true;
  }

  if (w_fire) {
    write(store_addr, (uint8_t *)w_data, w_strb, store_size);
    store_addr += store_size;
    store_count--;

    if (store_count == 0) {
      store_inflight = false;
      mem->AddTransaction(store_addr, true);
      wreq[store_addr].push(store_id);
      assert(w_last);
    }
  }

  if (b_fire)
    bresp.pop();

  if (r_fire)
    rresp.pop();

  mem->ClockTick();
  cycle++;

  if (reset) {
    while (!bresp.empty())
      bresp.pop();
    while (!rresp.empty())
      rresp.pop();
    cycle = 0;
    ar_ready_cache = false;
    aw_ready_cache = false;
  }
}
