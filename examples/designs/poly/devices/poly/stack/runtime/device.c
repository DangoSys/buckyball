/* The poly device runtime: the executor of a static device program
 * (program.h, lowered by lower-device) on the poly harts, whose values move
 * through the device SHM.
 *
 * Every hart of the device (DEVICE_HARTS of design.pb) has one worker, its
 * hart, which runs the entries of its hart in program order, following the
 * table's list of the hart. The program runs on the hart of the control hart
 * (DEVICE_CONTROL_HART) and dispatches the entries; that hart runs its own
 * entries while the program waits, in program_wait and program_finish. An
 * entry runs once it is dispatched and every wakeup it waits for has come:
 * its completion decrements the count of each entry it wakes and wakes the
 * hart of one that reaches zero. Entries retire in program order. A failed
 * entry squashes every later entry that has not started.
 *
 * An entry pulls the values other harts sent it at its front and pushes the
 * values it wrote at its tail. Every hart has a device SHM node of its own
 * (framework/device/shm, wired by arch/device/DeviceTop.scala); its CPU
 * submits bulk payload transfers through its memory service, and the node's
 * copy kernel moves bytes from it into the storage of any other hart, routed
 * by the hart links (arch/device/HartLinks.scala). The storage is not in the
 * DDR. Its low half stages what the hart sends, and holds at its top the
 * acknowledgement word of every receiving hart; its high half has a slot per
 * sending hart, in RING parts, each a chunk and a header word after it. A
 * link (a sending and a receiving hart) carries chunks in order: the sender
 * copies a chunk into the next part, then its header, the chunk's sequence
 * number; the receiver waits for the header, reads the chunk and copies the
 * count it has read into the sender's acknowledgement word, which frees the
 * part. A value sent to several harts is staged once per chunk. Messages on
 * a link are read in the order they are sent: convert-linalg-to-device lets
 * no entry of a receiving hart between a value's producer and its reader
 * wait for the producer's hart, so a sender that waits for a free part waits
 * for a reader that can run. When the chunks of a push do not fit in the free
 * parts, the entry wakes its readers before it pushes (program_release); then
 * they read while it sends.
 *
 * Latency is counted on a virtual clock per hart, in the cycles of its hart
 * (mcycle): an entry starts when its hart is free, the program dispatched it
 * and the entries it waits for ended (or released it), and lasts the cycles
 * its hart spends in it; the program's hart advances by the cycles the
 * program spends itself. Waiting for the shared allocator's lock is left
 * out. */
#ifdef PROGRAM_LINUX
#define _GNU_SOURCE
#endif
#include "device.h"

#include <runtime.h>
#include <stdlib.h>
#include <string.h>
#ifdef PROGRAM_LINUX
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <baremetal.h>
#endif

/* The stack of every hart's worker: the boot stack of a hart is 4 KiB, too
 * small for compiled graphs. */
#define PROGRAM_STACK_BYTES (1 << 20)

/* The harts of the device, in the order of design.pb. */
#define X(hartid)                                                              \
  {DEVICE_HARTS_##hartid##_HART_ID, DEVICE_HARTS_##hartid##_SIGNATURE},
static const struct {
  uint32_t hartid;
  uint64_t signature;
} device_harts[] = {DEVICE_HARTS};
#undef X
#define HART_COUNT (sizeof(device_harts) / sizeof(device_harts[0]))

#define LINK(op, field, value)                                                 \
  ({                                                                           \
    uint64_t result;                                                           \
    asm volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2"                        \
                 : "=r"(result)                                                \
                 : "r"((uint64_t)(field)), "r"((uint64_t)(value))              \
                 : "memory");                                                  \
    result;                                                                    \
  })
#define LINK_CAPACITY 0
#define RING 2

/* The storage layout, the same on every hart: the low half stages, the
 * high half receives; a part is a chunk and a 16-byte header. */
static uint64_t half, slot, stride;
/* The chunks each link sent and read so far: the sending hart counts the
 * first, the receiving hart the second. */
static uint64_t sent[HART_COUNT][HART_COUNT], received[HART_COUNT][HART_COUNT];
/* The harts that laid out their storage. */
static uint32_t laid;

static uint64_t chunk_bytes(void) { return stride - 16; }
static uint64_t part_of(uint64_t from, uint64_t count) {
  return half + from * slot + count % RING * stride;
}
static uint64_t acknowledgement(uint64_t to) { return half - 16 * (to + 1); }
static uint64_t staged_header(void) { return stride; }
static uint64_t staged_count(void) { return stride + 16; }
static uint32_t prepare(size_t hart);
static void park(size_t hart, uint32_t seen);
static void wake(size_t hart);

static void copy(uint64_t source, uint64_t to, uint64_t target,
                 uint64_t bytes) {
  LINK(13, 0, source);
  LINK(13, 1, to);
  LINK(13, 2, target);
  LINK(13, 3, (bytes + 7) & ~(uint64_t)7);
  if (LINK(14, 0, 0))
    __builtin_trap();
}

/* Headers and acknowledgements of this hart start at zero. */
static void lay_out(size_t hart) {
  half = LINK(15, LINK_CAPACITY, 0) / 2;
  slot = half / HART_COUNT & ~(uint64_t)15;
  stride = slot / RING & ~(uint64_t)15;
  for (size_t from = 0; from < HART_COUNT; ++from)
    for (uint64_t part = 0; part < RING; ++part)
      if (LINK(17, part_of(from, part) + chunk_bytes(), 0))
        __builtin_trap();
  for (size_t to = 0; to < HART_COUNT; ++to)
    if (LINK(17, acknowledgement(to), 0))
      __builtin_trap();
  (void)hart;
  __atomic_add_fetch(&laid, 1, __ATOMIC_ACQ_REL);
}

/* Reads a transfer sent to `to`: its words are source, target, bytes. */
static void pull(uint64_t from, uint64_t to, const uint64_t *moved) {
  uint8_t *target = (uint8_t *)(uintptr_t)moved[1];
  const uint64_t bytes = moved[2];
  for (uint64_t offset = 0; offset < bytes; offset += chunk_bytes()) {
    const uint64_t chunk =
        bytes - offset < chunk_bytes() ? bytes - offset : chunk_bytes();
    const uint64_t part = part_of(from, received[from][to]);
    while (LINK(16, part + chunk_bytes(), 0) != received[from][to] + 1) {
      uint32_t seen = prepare(to);
      if (LINK(16, part + chunk_bytes(), 0) != received[from][to] + 1)
        park(to, seen);
    }
    if (LINK(18, (uintptr_t)(target + offset), part) || LINK(19, 0, chunk))
      __builtin_trap();
    ++received[from][to];
    if (LINK(17, staged_count(), received[from][to]))
      __builtin_trap();
    copy(staged_count(), from, acknowledgement(to), 8);
    wake(from);
  }
}

/* Stages bytes of `source` at the start of the storage. */
static void stage(const uint8_t *source, uint64_t chunk) {
  if (LINK(18, (uintptr_t)source, 0) || LINK(19, 1, chunk))
    __builtin_trap();
}

static uint64_t chunks(uint64_t bytes) {
  return (bytes + chunk_bytes() - 1) / chunk_bytes();
}

enum { WAITING, DONE, FAILED, SQUASHED };
/* The state of an entry. */
enum { PENDING, STATUS, AFTER, END, JOB };

static struct runtime_hart harts[HART_COUNT];
/* The run: its table, words and state, its counts. */
static const uint64_t *table;
static uint64_t *words, *state;
static uint64_t entry_count;
static const uint64_t *entries, *lists;
static const struct program_transfer *transfers;
static uint64_t dispatched, retired, failed;
/* The next entry of each hart, and the entry the program waits for. */
static uint64_t heads[HART_COUNT];
static uint64_t waited;
static size_t control;
static int (*program_entry)(void *, void *);
static void *program_argument;
static int program_status;
/* The runs the control hart started, each hart's runs, the harts' exits. */
static uint32_t started, left;
static uint32_t entered[HART_COUNT];
#ifdef PROGRAM_LINUX
/* The caller of program_run waits on it for the end of the run. */
static uint32_t over;
#endif

static uint64_t clocks[HART_COUNT], busy[HART_COUNT];
/* The entry each hart runs, when it began on the virtual clock and on its
 * hart, and whether it woke its stream wakeups. */
static uint64_t running[HART_COUNT], running_begin[HART_COUNT],
    running_first[HART_COUNT];
static int released[HART_COUNT];
/* The program's clock mark: the cycles of its hart when it last advanced. */
static uint64_t mark;
#ifndef PROGRAM_LINUX
static uint64_t allocator_wait[HART_COUNT];
static unsigned allocation_depth[HART_COUNT];
static uint64_t allocator_waiters;
static _Thread_local size_t allocator_hart = SIZE_MAX;
static uint32_t heap;
#endif

static size_t hart_of(uint32_t hartid) {
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (device_harts[hart].hartid == hartid)
      return hart;
  __builtin_trap();
}

static uint64_t cycles(size_t hart) {
  uint64_t value;
#ifdef PROGRAM_LINUX
  (void)hart;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
#else
  asm volatile("csrr %0, mcycle" : "=r"(value));
  return value - allocator_wait[hart];
#endif
}

/* Waiting: a hart parks after `prepare` until woken; a wake between them is
 * not lost. */
#ifdef PROGRAM_LINUX
static uint32_t signals[HART_COUNT];
static uint32_t prepare(size_t hart) {
  return __atomic_load_n(&signals[hart], __ATOMIC_ACQUIRE);
}
static void park(size_t hart, uint32_t seen) {
  syscall(SYS_futex, &signals[hart], FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
}
static void wake(size_t hart) {
  __atomic_add_fetch(&signals[hart], 1, __ATOMIC_RELEASE);
  syscall(SYS_futex, &signals[hart], FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
}
#else
static uint32_t prepare(size_t hart) {
  runtime_baremetal_prepare_wait(device_harts[hart].hartid);
  return 0;
}
static void park(size_t hart, uint32_t seen) {
  (void)hart;
  (void)seen;
  asm volatile("wfi" ::: "memory");
}
static void wake(size_t hart) {
  runtime_baremetal_wake(device_harts[hart].hartid);
}
#endif

static const uint64_t *entry_of(uint64_t entry) {
  return &entries[entry * PROGRAM_ENTRY_WORDS];
}
static uint64_t *state_of(uint64_t entry) {
  return &state[entry * PROGRAM_STATE_WORDS];
}

static void raise_after(uint64_t entry, uint64_t stamp) {
  uint64_t *after = &state_of(entry)[AFTER];
  uint64_t seen = __atomic_load_n(after, __ATOMIC_RELAXED);
  while (seen < stamp &&
         !__atomic_compare_exchange_n(after, &seen, stamp, 1, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
  }
}

/* The entries of `list` (an offset and a count in the entry) hear that an
 * entry ended at `stamp`. */
static void wakeups(const uint64_t *list, uint64_t stamp) {
  for (uint64_t index = 0; index < list[1]; ++index) {
    const uint64_t woken = lists[list[0] + index];
    raise_after(woken, stamp);
    if (!__atomic_sub_fetch(&state_of(woken)[PENDING], 1, __ATOMIC_ACQ_REL))
      wake(entry_of(woken)[0]);
  }
}

static void retire(void) {
  uint64_t oldest = __atomic_load_n(&retired, __ATOMIC_ACQUIRE);
  while (oldest < entry_count && __atomic_load_n(&state_of(oldest)[STATUS],
                                                 __ATOMIC_ACQUIRE) != WAITING) {
    if (__atomic_compare_exchange_n(&retired, &oldest, oldest + 1, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      ++oldest;
  }
  if (oldest == entry_count)
    wake(control);
}

static void finish_entry(uint64_t entry, uint64_t status) {
  __atomic_store_n(&state_of(entry)[STATUS], status, __ATOMIC_RELEASE);
  if (__atomic_load_n(&waited, __ATOMIC_ACQUIRE) == entry)
    wake(control);
  retire();
}

/* Sends the transfers of `list` (an offset and a count) from `hart`. Receivers
 * progress independently; those at the same offset share a staged chunk.
 * If the values do not fit in the
 * free parts of their links, the readers are woken first. */
static void push(size_t hart, const uint64_t *list) {
  uint64_t needed[HART_COUNT] = {0};
  int fits = 1;
  for (uint64_t index = 0; index < list[1]; ++index) {
    const struct program_transfer *moved = &transfers[lists[list[0] + index]];
    needed[moved->to] += chunks(words[moved->words + 2]);
  }
  for (size_t to = 0; to < HART_COUNT; ++to)
    fits &= !needed[to] || sent[hart][to] + needed[to] <=
                               LINK(16, acknowledgement(to), 0) + RING;
  if (!fits) {
    const uint64_t stamp =
        running_begin[hart] + cycles(hart) - running_first[hart];
    wakeups(&entry_of(running[hart])[6], stamp);
    released[hart] = 1;
  }
  for (uint64_t index = 0; index < list[1];) {
    const uint64_t *value = &words[transfers[lists[list[0] + index]].words];
    // The compiler identifies recipients of the same logical value.
    uint64_t end = index + 1;
    while (end < list[1] &&
           words[transfers[lists[list[0] + end]].words + 3] == value[3])
      ++end;
    const uint8_t *source = (const uint8_t *)(uintptr_t)value[0];
    // Point-to-point transfers need no table of recipient progress.
    if (end == index + 1) {
      const uint64_t to = transfers[lists[list[0] + index]].to;
      for (uint64_t offset = 0; offset < value[2]; offset += chunk_bytes()) {
        const uint64_t chunk = value[2] - offset < chunk_bytes()
                                   ? value[2] - offset
                                   : chunk_bytes();
        stage(source + offset, chunk);
        while (sent[hart][to] >= LINK(16, acknowledgement(to), 0) + RING) {
          const uint32_t seen = prepare(hart);
          if (sent[hart][to] >= LINK(16, acknowledgement(to), 0) + RING)
            park(hart, seen);
        }
        const uint64_t part = part_of(hart, sent[hart][to]);
        copy(0, to, part, chunk);
        if (LINK(17, staged_header(), sent[hart][to] + 1))
          __builtin_trap();
        copy(staged_header(), to, part + chunk_bytes(), 8);
        ++sent[hart][to];
        wake(to);
      }
      index = end;
      continue;
    }
    uint64_t offsets[end - index];
    memset(offsets, 0, sizeof(offsets));
    uint64_t remaining = value[2] ? end - index : 0;
    while (remaining) {
      const uint32_t seen = prepare(hart);
      uint64_t offset = value[2];
      for (uint64_t moved = index; moved < end; ++moved) {
        const uint64_t to = transfers[lists[list[0] + moved]].to;
        if (offsets[moved - index] < offset &&
            sent[hart][to] < LINK(16, acknowledgement(to), 0) + RING)
          offset = offsets[moved - index];
      }
      if (offset == value[2]) {
        park(hart, seen);
        continue;
      }
      const uint64_t chunk =
          value[2] - offset < chunk_bytes() ? value[2] - offset : chunk_bytes();
      stage(source + offset, chunk);
      for (uint64_t moved = index; moved < end; ++moved) {
        const uint64_t to = transfers[lists[list[0] + moved]].to;
        if (offsets[moved - index] != offset ||
            sent[hart][to] >= LINK(16, acknowledgement(to), 0) + RING)
          continue;
        const uint64_t part = part_of(hart, sent[hart][to]);
        copy(0, to, part, chunk);
        if (LINK(17, staged_header(), sent[hart][to] + 1))
          __builtin_trap();
        copy(staged_header(), to, part + chunk_bytes(), 8);
        ++sent[hart][to];
        wake(to);
        offsets[moved - index] += chunk;
        if (offsets[moved - index] == value[2])
          --remaining;
      }
    }
    index = end;
  }
}

/* Runs the next entry of `hart` if it can: 1 if it ran or was squashed, 0 if
 * it waits, -1 if the hart has none left. */
static int step(size_t hart, void *context) {
  const uint64_t entry = heads[hart];
  if (entry >= entry_count)
    return -1;
  const uint64_t *fields = entry_of(entry);
  if (entry > __atomic_load_n(&failed, __ATOMIC_ACQUIRE)) {
    heads[hart] = fields[3];
    finish_entry(entry, SQUASHED);
    return 1;
  }
  if (entry >= __atomic_load_n(&dispatched, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&state_of(entry)[PENDING], __ATOMIC_ACQUIRE))
    return 0;
  heads[hart] = fields[3];
  uint64_t begin = clocks[hart];
  const uint64_t after =
      __atomic_load_n(&state_of(entry)[AFTER], __ATOMIC_ACQUIRE);
  if (after > begin)
    begin = after;
  const uint64_t first = cycles(hart);
  running[hart] = entry;
  running_begin[hart] = begin;
  running_first[hart] = first;
  released[hart] = 0;
  for (uint64_t index = 0; index < fields[9]; ++index) {
    const struct program_transfer *moved = &transfers[lists[fields[8] + index]];
    pull(moved->from, hart, &words[moved->words]);
  }
  int (*job)(void *, void *) =
      (int (*)(void *, void *))(uintptr_t)state_of(entry)[JOB];
  const int status = job(context, &words[fields[1]]);
  if (!status)
    push(hart, &fields[10]);
  const uint64_t spent = cycles(hart) - first;
  const uint64_t end = begin + spent;
  state_of(entry)[END] = end;
  clocks[hart] = end;
  busy[hart] += spent;
  if (status) {
    uint64_t seen = __atomic_load_n(&failed, __ATOMIC_ACQUIRE);
    while (entry < seen &&
           !__atomic_compare_exchange_n(&failed, &seen, entry, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    }
    // Every hart squashes its later entries.
    for (size_t other = 0; other < HART_COUNT; ++other)
      wake(other);
  }
  wakeups(&fields[4], end);
  if (!released[hart])
    wakeups(&fields[6], end);
  finish_entry(entry, status ? FAILED : DONE);
  return 1;
}

/* The program's clock, advanced by the cycles it ran since `mark`. */
static void advance(void) {
  const uint64_t now = cycles(control);
  clocks[control] += now - mark;
  busy[control] += now - mark;
  mark = now;
}

void program_start(const uint64_t *program, uint64_t *run_words,
                   uint64_t *run_state) {
  table = program;
  words = run_words;
  state = run_state;
  entry_count = table[0];
  const uint64_t hart_count = table[2];
  if (hart_count > HART_COUNT)
    __builtin_trap();
  entries = &table[3 + hart_count];
  transfers = (const struct program_transfer
                   *)&entries[entry_count * PROGRAM_ENTRY_WORDS];
  lists = (const uint64_t *)&transfers[table[1]];
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    heads[hart] = hart < hart_count ? table[3 + hart] : entry_count;
  for (uint64_t entry = 0; entry < entry_count; ++entry) {
    uint64_t *fields = state_of(entry);
    fields[PENDING] = entry_of(entry)[2];
    fields[STATUS] = WAITING;
    fields[AFTER] = 0;
    fields[END] = 0;
  }
  dispatched = 0;
  retired = 0;
  failed = entry_count;
  waited = entry_count;
  mark = cycles(control);
  __atomic_store_n(&started, started + 1, __ATOMIC_RELEASE);
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (hart != control)
      wake(hart);
}

void program_dispatch(uint64_t entry, int (*job)(void *, void *)) {
  advance();
  if (entry != dispatched)
    __builtin_trap();
  state_of(entry)[JOB] = (uintptr_t)job;
  raise_after(entry, clocks[control]);
  __atomic_store_n(&dispatched, entry + 1, __ATOMIC_RELEASE);
  if (!__atomic_load_n(&state_of(entry)[PENDING], __ATOMIC_ACQUIRE))
    wake(entry_of(entry)[0]);
  mark = cycles(control);
}

/* The program's hart runs its own entries until `done`. */
static void serve(void *context, int (*done)(void)) {
  for (;;) {
    const uint32_t seen = prepare(control);
    if (done())
      return;
    if (step(control, context) <= 0)
      park(control, seen);
  }
}

static void *program_context;

static int waited_done(void) {
  return __atomic_load_n(&state_of(waited)[STATUS], __ATOMIC_ACQUIRE) !=
         WAITING;
}

void program_wait(uint64_t entry) {
  advance();
  __atomic_store_n(&waited, entry, __ATOMIC_RELEASE);
  serve(program_context, waited_done);
  if (__atomic_load_n(&state_of(entry)[STATUS], __ATOMIC_ACQUIRE) != DONE)
    __builtin_trap();
  waited = entry_count;
  if (state_of(entry)[END] > clocks[control])
    clocks[control] = state_of(entry)[END];
  mark = cycles(control);
}

static int all_retired(void) {
  return __atomic_load_n(&retired, __ATOMIC_ACQUIRE) == entry_count;
}

void program_finish(void) {
  advance();
  serve(program_context, all_retired);
  mark = cycles(control);
}

int program_copy_job(void *hart, void *argument) {
  (void)hart;
  void *const *copy = argument;
  memcpy(copy[0], copy[1], (uintptr_t)copy[2]);
  return 0;
}

uint64_t program_latency(void) {
  uint64_t latency = 0;
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (clocks[hart] > latency)
      latency = clocks[hart];
  return latency;
}

uint64_t program_busy(uint64_t hart) { return busy[hart]; }

/* A worker: the program on the control hart, the entries elsewhere, for one
 * run. */
static void work_run(size_t hart, void *context, uint32_t run) {
  if (run == 1) {
    lay_out(hart);
    while (__atomic_load_n(&laid, __ATOMIC_ACQUIRE) != HART_COUNT) {
    }
  }
  if (hart == control) {
    clocks[hart] = busy[hart] = 0;
    program_context = context;
    program_status = program_entry(context, program_argument);
    advance();
  } else {
    for (;;) {
      const uint32_t seen = prepare(hart);
      if (__atomic_load_n(&started, __ATOMIC_ACQUIRE) == run)
        break;
      park(hart, seen);
    }
    clocks[hart] = busy[hart] = 0;
    for (;;) {
      const uint32_t seen = prepare(hart);
      const int progress = step(hart, context);
      if (progress < 0)
        break;
      if (!progress)
        park(hart, seen);
    }
  }
#ifdef PROGRAM_LINUX
  if (__atomic_add_fetch(&left, 1, __ATOMIC_ACQ_REL) == HART_COUNT) {
    __atomic_add_fetch(&over, 1, __ATOMIC_RELEASE);
    syscall(SYS_futex, &over, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
  }
#else
  __atomic_fetch_add(&left, 1, __ATOMIC_ACQ_REL);
#endif
}

static int result(void) {
  if (program_status)
    return program_status;
  return failed < entry_count ? -1 - (int)failed : 0;
}

#ifdef PROGRAM_LINUX
static pthread_t threads[HART_COUNT];
/* The run the workers are asked for. */
static uint32_t requested;

static int work(void *context, void *argument) {
  const size_t hart = (uintptr_t)argument;
  for (uint32_t run = 1;; ++run) {
    for (;;) {
      const uint32_t seen = prepare(hart);
      if (__atomic_load_n(&requested, __ATOMIC_ACQUIRE) >= run)
        break;
      park(hart, seen);
    }
    work_run(hart, context, run);
  }
  return 0;
}

static void *thread(void *argument) {
  const size_t hart = (uintptr_t)argument;
  if (harts[hart].execute(harts[hart].context, work, argument))
    abort();
  return NULL;
}

int program_run(int (*program)(void *, void *), void *argument) {
  control = hart_of(DEVICE_CONTROL_HART);
  program_entry = program;
  program_argument = argument;
  const uint32_t run = requested + 1;
  if (run == 1) {
    unsigned cpus[HART_COUNT];
    for (size_t index = 0; index < HART_COUNT; ++index)
      cpus[index] = UINT32_MAX;
    FILE *info = fopen("/proc/cpuinfo", "r");
    if (!info)
      abort();
    char line[256];
    unsigned cpu = UINT32_MAX, hartid;
    while (fgets(line, sizeof(line), info)) {
      if (sscanf(line, "processor : %u", &cpu) == 1)
        continue;
      if (sscanf(line, "hart : %u", &hartid) == 1)
        for (size_t index = 0; index < HART_COUNT; ++index)
          if (device_harts[index].hartid == hartid)
            cpus[index] = cpu;
    }
    fclose(info);
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) ||
        pthread_attr_setstacksize(&attributes, PROGRAM_STACK_BYTES))
      abort();
    for (size_t hart = 0; hart < HART_COUNT; ++hart) {
      if (cpus[hart] == UINT32_MAX)
        abort();
      harts[hart] = runtime_hart(device_harts[hart].hartid,
                                 device_harts[hart].signature, 0);
      cpu_set_t affinity;
      CPU_ZERO(&affinity);
      CPU_SET(cpus[hart], &affinity);
      if (pthread_attr_setaffinity_np(&attributes, sizeof(affinity),
                                      &affinity) ||
          pthread_create(&threads[hart], &attributes, thread,
                         (void *)(uintptr_t)hart))
        abort();
    }
    pthread_attr_destroy(&attributes);
  }
  left = 0;
  __atomic_store_n(&requested, run, __ATOMIC_RELEASE);
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    wake(hart);
  // The caller is no worker: it waits on a word of its own.
  for (;;) {
    const uint32_t seen = __atomic_load_n(&over, __ATOMIC_ACQUIRE);
    if (__atomic_load_n(&left, __ATOMIC_ACQUIRE) == HART_COUNT)
      break;
    syscall(SYS_futex, &over, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
  }
  return result();
}
#else
/* Newlib's shared heap needs a recursive lock. Only acquisition waiting is
 * excluded from hart-local timing; allocator execution remains counted. */
static size_t this_hart(void) {
  if (allocator_hart == SIZE_MAX) {
    uintptr_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    allocator_hart = hart_of(hartid);
  }
  return allocator_hart;
}

void __malloc_lock(void *reent) {
  (void)reent;
  const size_t hart = this_hart();
  if (allocation_depth[hart]++)
    return;
  uint64_t begin, acquired;
  asm volatile("csrr %0, mcycle" : "=r"(begin));
  __atomic_fetch_or(&allocator_waiters, UINT64_C(1) << hart, __ATOMIC_RELEASE);
  for (;;) {
    runtime_baremetal_prepare_wait(device_harts[hart].hartid);
    if (!__atomic_exchange_n(&heap, 1, __ATOMIC_ACQUIRE))
      break;
    asm volatile("wfi" ::: "memory");
  }
  __atomic_fetch_and(&allocator_waiters, ~(UINT64_C(1) << hart),
                     __ATOMIC_RELEASE);
  asm volatile("csrr %0, mcycle" : "=r"(acquired));
  allocator_wait[hart] += acquired - begin;
}

void __malloc_unlock(void *reent) {
  (void)reent;
  const size_t hart = this_hart();
  if (--allocation_depth[hart])
    return;
  __atomic_store_n(&heap, 0, __ATOMIC_RELEASE);
  uint64_t waiters = __atomic_load_n(&allocator_waiters, __ATOMIC_ACQUIRE);
  for (size_t waiting = 0; waiters; ++waiting, waiters >>= 1)
    if (waiters & 1)
      runtime_baremetal_wake(device_harts[waiting].hartid);
}

static uint8_t stacks[HART_COUNT][PROGRAM_STACK_BYTES]
    __attribute__((aligned(16), section(".noinit")));

struct worker {
  size_t hart;
  uint32_t run;
};

static int work(void *context, void *argument) {
  const struct worker *worker = argument;
  work_run(worker->hart, context, worker->run);
  return 0;
}

static void run_on_stack(size_t hart, uint32_t run) {
  struct worker worker = {hart, run};
  harts[hart] =
      runtime_hart(device_harts[hart].hartid, device_harts[hart].signature, 0);
  if (harts[hart].execute(harts[hart].context, work, &worker))
    __builtin_trap();
}

int program_run(int (*program)(void *, void *), void *argument) {
  const size_t hart = this_hart();
  const uint32_t run = ++entered[hart];
  control = hart_of(DEVICE_CONTROL_HART);
  if (hart == control) {
    // A run starts once every hart left the run before.
    while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != (run - 1) * HART_COUNT) {
    }
    program_entry = program;
    program_argument = argument;
  }
  // The worker runs on the hart's stack; the boot stack pointer is kept on
  // it and restored.
  register uintptr_t top asm("t1") = (uintptr_t)stacks[hart + 1];
  register size_t worker_hart asm("a0") = hart;
  register uint32_t worker_run asm("a1") = run;
  asm volatile("mv t0, sp\n"
               "addi sp, t1, -16\n"
               "sd t0, 0(sp)\n"
               "jalr %3\n"
               "ld sp, 0(sp)\n"
               : "+r"(worker_hart), "+r"(top), "+r"(worker_run)
               : "r"(run_on_stack)
               : "ra", "t0", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4",
                 "a5", "a6", "a7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5",
                 "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa0", "fa1",
                 "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "memory");
  if (hart != control)
    return 0;
  while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != run * HART_COUNT) {
  }
  return result();
}
#endif
