"""Bit-exact Python model of CheckerOnChip and its unit test.

usage: model.py <buckets> <tb binary> <work dir> <seed>...

For each seed it writes a stimulus, runs the Verilator testbench, and checks every verdict field
(hash, max, min, sum d^2, sum d^4, fail bits) against the model, plus the expected outcome of each case:
clean subgraphs pass, a planted deviation trips SC, a flipped byte trips HC.
"""

import json
import random
import struct
import subprocess
import sys
from pathlib import Path

m32 = 0xFFFFFFFF
acc_bits = 48
dev_bits = 16
address_bits = 44


def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & m32


def xorshift(x):
    a = x ^ ((x << 13) & m32)
    b = a ^ (a >> 17)
    return b ^ ((b << 5) & m32)


def signed(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def fp_to_fixed(bits, exp):
    e = (bits >> 23) & 0xFF
    mantissa = (1 << 23) | (bits & 0x7FFFFF)
    shift = e - 150 - exp
    if shift > 7 or e == 255:
        magnitude = 0x7FFFFFFF
    elif shift < -24 or e == 0:
        magnitude = 0
    elif shift >= 0:
        magnitude = (mantissa << shift) & 0x7FFFFFFF
    else:
        magnitude = mantissa >> -shift
    return -magnitude if bits >> 31 else magnitude


def lane_values(data, strb, fmt, fp_exp):
    if fmt == 0:
        return [signed(data >> (8 * i), 8) if (strb >> i) & 1 else 0 for i in range(16)]
    values = []
    for i in range(4):
        word = (data >> (32 * i)) & m32
        full = ((strb >> (4 * i)) & 0xF) == 0xF
        value = signed(word, 32) if fmt == 1 else fp_to_fixed(word, fp_exp)
        values.append(value if full else 0)
    return values


def beat_sum(beat_addr, data, strb, fmt, fp_exp, seed):
    signs = xorshift(xorshift((beat_addr & m32) ^ seed))
    lanes = lane_values(data, strb, fmt, fp_exp)
    return sum(-v if (signs >> i) & 1 else v for i, v in enumerate(lanes))


def row_hash(beat_addr, data, strb):
    mask = sum(0xFF << (8 * i) for i in range(16) if (strb >> i) & 1)
    row = data & mask
    key = (beat_addr & m32) ^ ((beat_addr >> 32) & 0xFF)
    w = [(row >> (32 * i)) & m32 for i in range(4)]
    return w[0] ^ rotl(w[1], 7) ^ rotl(w[2], 13) ^ rotl(w[3], 21) ^ rotl(key, 11)


class Entry:
    def __init__(self, base=0, size=0, fmt=0, fp_exp=0, sc=False, t_r=0, t_k=0):
        self.base, self.size, self.fmt, self.fp_exp, self.sc = (
            base,
            size,
            fmt,
            fp_exp,
            sc,
        )
        self.t_r, self.t_k, self.hash = t_r, t_k, 0

    def words(self):
        control = (
            self.size
            | (self.fmt << 32)
            | ((self.fp_exp & 0xFF) << 40)
            | (int(self.sc) << 48)
        )
        return [self.hash, self.base, control, self.t_r | (self.t_k << 32)]

    def in_window(self, beat_addr):
        return self.sc and self.base >> 4 <= beat_addr < (self.base + self.size) >> 4


class Model:
    """Mirrors the RTL state: hash, bucket accumulators, current entry."""

    def __init__(self, buckets, seed):
        self.buckets, self.seed = buckets, seed
        self.acc = [0] * buckets
        self.hash = 0

    def load_refs(self, refs):
        self.acc = [(-v) & ((1 << acc_bits) - 1) for v in refs]

    def beat(self, entry, beat_addr, data, strb):
        if entry.in_window(beat_addr):
            j = beat_addr % self.buckets
            s = beat_sum(beat_addr, data, strb, entry.fmt, entry.fp_exp, self.seed)
            self.acc[j] = (self.acc[j] + s) & ((1 << acc_bits) - 1)
        else:
            self.hash = (self.hash + row_hash(beat_addr, data, strb)) & m32

    def fence(self, index, entry):
        top = (1 << (dev_bits - 1)) - 1
        high, low, s2, s4 = -top - 1, top, 0, 0
        if entry.sc:
            for value in self.acc:
                d = max(-top - 1, min(top, signed(value, acc_bits)))
                high, low = max(high, d), min(low, d)
                s2 += d * d
                s4 += d**4
        bucket_bits = self.buckets.bit_length() - 1
        verdict = {
            "entry": index,
            "hash_fail": int(self.hash != entry.hash),
            "range_fail": int(entry.sc and high - low > entry.t_r),
            "kurtosis_fail": int(
                entry.sc and (s4 << (bucket_bits + 8)) > entry.t_k * s2 * s2
            ),
            "hash": self.hash,
            "max": high,
            "min": low,
            "s2": s2,
            "s4": s4,
        }
        self.hash = 0
        return verdict


def statistics(buckets, deviations):
    """Range and kurtosis of the bucket deviations (for calibration and reporting)."""
    s2 = sum(d * d for d in deviations)
    s4 = sum(d**4 for d in deviations)
    return max(deviations) - min(deviations), buckets * s4 / (s2 * s2) if s2 else 0.0


def beats_of(base, payload):
    """Splits a payload into 16-byte beats: (beat address, data, strobe)."""
    out = []
    for offset in range(0, len(payload), 16):
        chunk = payload[offset : offset + 16]
        out.append(
            (
                (base + offset) >> 4,
                int.from_bytes(chunk.ljust(16, b"\0"), "little"),
                (1 << len(chunk)) - 1,
            )
        )
    return out


def reference_buckets(buckets, seed, base, fmt, fp_exp, exact):
    """GPU side: V_j = round(sum s_i * v_i / Delta) over the elements of bucket j."""
    lanes = 16 if fmt == 0 else 4
    delta = 2.0**fp_exp if fmt == 2 else 1.0
    refs = [0.0] * buckets
    for k in range(0, len(exact), lanes):
        beat_addr = (base >> 4) + k // lanes
        signs = xorshift(xorshift((beat_addr & m32) ^ seed))
        for i, v in enumerate(exact[k : k + lanes]):
            refs[beat_addr % buckets] += (-v if (signs >> i) & 1 else v) / delta
    return [round(r) for r in refs]


class Stimulus:
    def __init__(self, path, model):
        self.out = open(path, "w")
        self.model = model
        self.rng = random.Random(model.seed)

    def config(self, addr, data):
        self.out.write(f"c {addr:x} {data & ((1 << 64) - 1):x}\n")

    def start(self, name, entries, refs=None):
        self.out.write(f"s case {name}\n")
        self.config(0x2000, self.model.seed)
        self.config(0x2001, 0)
        for index, entry in enumerate(entries):
            for word, value in enumerate(entry.words()):
                self.config(4 * index + word, value)
        if refs is not None:
            self.load_refs(refs)

    def load_refs(self, refs):
        for j, v in enumerate(refs):
            self.config(0x1000 + j, v)
        self.model.load_refs(refs)

    def write(self, entry, base, payload, flip=None):
        """AXI bursts of up to 16 beats; every other burst sends two W beats before its AW."""
        beats = beats_of(base, payload)
        for start in range(0, len(beats), 16):
            burst = beats[start : start + 16]
            lead = 2 if (start // 16) % 2 and len(burst) > 2 else 0
            for i, (beat_addr, data, strb) in enumerate(burst):
                if flip == start + i:
                    data ^= 0x80
                aw = i == lead
                self.out.write(
                    f"b {int(aw)} {burst[0][0] << 4:x} {len(burst) - 1} 1 {data:x} {strb:x}\n"
                )
                self.model.beat(entry, beat_addr, data, strb)

    def fence(self, index, entry):
        self.out.write("f\n")
        return self.model.fence(index, entry)


def expected_hash(base, payload):
    h = 0
    for beat_addr, data, strb in beats_of(base, payload):
        h = (h + row_hash(beat_addr, data, strb)) & m32
    return h


def int8_tensor(rng, n):
    """GPU exact values v (in Delta units) and a legal NPU output y = floor(v) (truncating requantization)."""
    exact = [rng.gauss(0.0, 30.0) for _ in range(n)]
    y = [max(-128, min(127, int(v // 1))) for v in exact]
    exact = [max(-128.0, min(127.0, v)) for v in exact]
    return exact, y


def fp32_tensor(rng, n, fp_exp):
    """v exact; y = v + up to one Delta of legal error, stored as FP32."""
    delta = 2.0**fp_exp
    exact = [rng.gauss(0.0, 4.0) for _ in range(n)]
    y = [v + rng.uniform(-1.0, 1.0) * delta for v in exact]
    return exact, y


def int32_tensor(rng, n):
    exact = [rng.gauss(0.0, 5000.0) for _ in range(n)]
    y = [round(v) + rng.choice((-1, 0, 0, 1)) for v in exact]
    return exact, y


def pack(fmt, values):
    if fmt == 0:
        return bytes(v & 0xFF for v in values)
    if fmt == 1:
        return b"".join(struct.pack("<i", v) for v in values)
    return b"".join(struct.pack("<f", v) for v in values)


def calibrate(buckets, seed, base, fmt, fp_exp, make, instances=8):
    """Thresholds from clean instances of the same subgraph: 1.25 x the largest range and kurtosis seen."""
    rng = random.Random(seed * 7919 + fmt)
    ranges, kurts = [], []
    for _ in range(instances):
        exact, y = make(rng)
        model = Model(buckets, seed)
        model.load_refs(reference_buckets(buckets, seed, base, fmt, fp_exp, exact))
        entry = Entry(base, len(y) * (1 if fmt == 0 else 4), fmt, fp_exp, True)
        for beat_addr, data, strb in beats_of(base, pack(fmt, y)):
            model.beat(entry, beat_addr, data, strb)
        r, k = statistics(buckets, [signed(v, acc_bits) for v in model.acc])
        ranges.append(r)
        kurts.append(k)
    return int(1.25 * max(ranges)), int(1.25 * max(kurts) * 256)


def run_seed(buckets, tb, work, seed):
    stim_path = work / f"stim_b{buckets}_s{seed}.txt"
    model = Model(buckets, seed)
    stim = Stimulus(stim_path, model)
    rng = random.Random(seed)
    expect = []  # (case, model verdict, expected outcome)

    warm_base, target_base, follow_base = 0x8000_0000, 0x8010_0000, 0x8020_0000
    n_int8 = 65536
    t_r, t_k = calibrate(
        buckets, seed, target_base, 0, 0, lambda r: int8_tensor(r, n_int8)
    )
    warm = bytes(rng.getrandbits(8) for _ in range(4096))
    follow = bytes(rng.getrandbits(8) for _ in range(2048))
    exact, y = int8_tensor(rng, n_int8)
    refs = reference_buckets(buckets, seed, target_base, 0, 0, exact)

    for case in ("clean", "planted", "hash"):
        entries = [
            Entry(),
            Entry(target_base, n_int8, 0, 0, True, t_r, t_k),
            Entry(),
        ]
        entries[0].hash = expected_hash(warm_base, warm)
        entries[2].hash = expected_hash(follow_base, follow)
        model.hash = 0
        stim.start(f"int8_{case}", entries, refs)
        stim.write(entries[0], warm_base, warm, flip=37 if case == "hash" else None)
        expect.append(
            (
                case + ":warmup",
                stim.fence(0, entries[0]),
                "hash" if case == "hash" else "pass",
            )
        )
        out = list(y)
        if case == "planted":
            for i in rng.sample(range(n_int8), 4):
                out[i] = max(-128, min(127, out[i] + 64))
        stim.write(entries[1], target_base, pack(0, out))
        expect.append(
            (
                case + ":target",
                stim.fence(1, entries[1]),
                "sc" if case == "planted" else "pass",
            )
        )
        stim.write(entries[2], follow_base, follow)
        expect.append((case + ":followup", stim.fence(2, entries[2]), "pass"))

    # FP32 target, then INT32 target with references reloaded between the fences.
    fp_exp, n_word = -6, 4096
    fp_tr, fp_tk = calibrate(
        buckets, seed, target_base, 2, fp_exp, lambda r: fp32_tensor(r, n_word, fp_exp)
    )
    i32_tr, i32_tk = calibrate(
        buckets, seed, follow_base, 1, 0, lambda r: int32_tensor(r, n_word)
    )
    fp_exact, fp_y = fp32_tensor(rng, n_word, fp_exp)
    i32_exact, i32_y = int32_tensor(rng, n_word)
    for case in ("clean", "planted"):
        entries = [
            Entry(target_base, 4 * n_word, 2, fp_exp, True, fp_tr, fp_tk),
            Entry(follow_base, 4 * n_word, 1, 0, True, i32_tr, i32_tk),
        ]
        model.hash = 0
        stim.start(
            f"word_{case}",
            entries,
            reference_buckets(buckets, seed, target_base, 2, fp_exp, fp_exact),
        )
        out = list(fp_y)
        if case == "planted":
            out[rng.randrange(n_word)] += 400 * 2.0**fp_exp
        stim.write(entries[0], target_base, pack(2, out))
        expect.append(
            (
                "fp32_" + case,
                stim.fence(0, entries[0]),
                "sc" if case == "planted" else "pass",
            )
        )
        stim.load_refs(reference_buckets(buckets, seed, follow_base, 1, 0, i32_exact))
        out = list(i32_y)
        if case == "planted":
            out[rng.randrange(n_word)] += 3000
        stim.write(entries[1], follow_base, pack(1, out))
        expect.append(
            (
                "int32_" + case,
                stim.fence(1, entries[1]),
                "sc" if case == "planted" else "pass",
            )
        )
    stim.out.close()

    result = subprocess.run(
        [str(tb), str(stim_path)], capture_output=True, text=True, check=True
    )
    lines = [
        line.split() for line in result.stdout.splitlines() if line.startswith("v ")
    ]
    if len(lines) != len(expect):
        raise RuntimeError(f"expected {len(expect)} verdicts, got {len(lines)}")
    rows = []
    for (case, m, outcome), f in zip(expect, lines):
        rtl = {
            "entry": int(f[1]),
            "hash_fail": int(f[2]),
            "range_fail": int(f[3]),
            "kurtosis_fail": int(f[4]),
            "hash": int(f[5], 16),
            "max": int(f[6]),
            "min": int(f[7]),
            "s2": int(f[8], 16),
            "s4": int(f[9], 16),
        }
        if rtl != m:
            raise RuntimeError(
                f"B={buckets} seed={seed} {case}: RTL {rtl} != model {m}"
            )
        sc = m["range_fail"] or m["kurtosis_fail"]
        got = "hash" if m["hash_fail"] else "sc" if sc else "pass"
        if got != outcome:
            raise RuntimeError(
                f"B={buckets} seed={seed} {case}: outcome {got}, expected {outcome}"
            )
        kurt = (buckets * m["s4"] / (m["s2"] ** 2)) if m["s2"] else 0.0
        rows.append(
            {
                "buckets": buckets,
                "seed": seed,
                "case": case,
                "outcome": got,
                "range": m["max"] - m["min"] if m["s2"] else 0,
                "kurtosis": round(kurt, 4),
                "fence_to_verdict_cycles": int(f[10]),
                "last_beat_to_verdict_cycles": int(f[11]),
            }
        )
    return rows


def main():
    buckets, tb, work = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for seed in sys.argv[4:]:
        rows += run_seed(buckets, tb, work, int(seed))
    for r in rows:
        print(json.dumps(r))
    print(f"PASS B={buckets}: {len(rows)} verdicts match the model")


if __name__ == "__main__":
    main()
