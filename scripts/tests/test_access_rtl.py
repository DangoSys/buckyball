import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class AccessRTLTest(unittest.TestCase):
    def test_accepted_writes_and_simultaneous_producers(self):
        with tempfile.TemporaryDirectory(prefix="access-rtl-test-") as directory:
            work = Path(directory)
            (work / "test.sv").write_text("""module AccessObserverTest;
  bit clock=0; always #5 clock=~clock;
  bit reset=1, valid=0, ready=0, second=0;
  bit [127:0] data=128'h1234;
  bit [31:0] mask=32'hffff;
  AccessWriteDPI a(.clock(clock), .reset(reset), .fire(valid && ready), .idle(!valid),
    .stream_hart(64'd0), .hart(64'd0), .inst(64'd7), .shared(32'd0),
    .physical(32'd0), .bank(32'd3), .group(32'd0), .addr(32'd5), .mask(mask), .data(data));
  AccessWriteDPI b(.clock(clock), .reset(reset), .fire(second), .idle(!second),
    .stream_hart(64'd0), .hart(64'd0), .inst(64'd7), .shared(32'd1),
    .physical(32'd1), .bank(32'd4), .group(32'd0), .addr(32'd5), .mask(mask), .data(data));
  import "DPI-C" function void check_events();
  int unsigned owner_lo, owner_hi, shared, producer, lo, hi, idle;
  initial begin
    @(negedge clock); reset=0; valid=1;
    repeat(3) @(negedge clock);
    ready=1; second=1;
    @(negedge clock); ready=0; second=0;
    repeat(2) @(negedge clock);
    ready=1; mask=1; data=128'h42;
    @(negedge clock); ready=0; valid=0; #1;
    a.access_snapshot(owner_lo,owner_hi,shared,producer,lo,hi,idle);
    assert(lo==2 && hi==0 && producer==0 && idle==1) else $fatal;
    b.access_snapshot(owner_lo,owner_hi,shared,producer,lo,hi,idle);
    assert(lo==1 && producer==1 && shared==1) else $fatal;
    check_events();
    $finish;
  end
endmodule
""")
            (work / "callbacks.cpp").write_text("""#include <cassert>
#include <cstdint>
static unsigned counts[2] = {};
extern "C" void dpi_access_write(uint32_t ol,uint32_t oh,uint32_t hl,uint32_t hh,
 uint32_t il,uint32_t ih,uint32_t shared,uint32_t pbank,uint32_t bank,uint32_t group,
 uint32_t sl,uint32_t sh,uint32_t addr,uint32_t mask,
 uint32_t d0,uint32_t d1,uint32_t d2,uint32_t d3) {
 const auto producer = pbank;
 assert(producer < 2 && sl == counts[producer]++ && sh == 0);
 assert(ol == 0 && oh == 0 && hl == 0 && hh == 0 && il == 7 && ih == 0);
 assert(pbank == producer && bank == producer + 3 && shared == producer && group == 0 && addr == 5);
 assert(d1 == 0 && d2 == 0 && d3 == 0);
 if (sl == 0) { assert(mask == 0xffff && d0 == 0x1234); }
 else { assert(mask == 1 && d0 == 0x42); }
}
extern "C" void check_events() { assert(counts[0] == 2 && counts[1] == 1); }
""")
            command = [
                "verilator",
                "--binary",
                "--timing",
                "--assert",
                "--top-module",
                "AccessObserverTest",
                "--Mdir",
                str(work / "obj"),
                str(work / "test.sv"),
                str(ROOT / "arch/src/main/resources/vsrc/AccessWriteDPI.sv"),
                str(work / "callbacks.cpp"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [str(work / "obj/VAccessObserverTest")], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_private_and_shared_backend_observation(self):
        with tempfile.TemporaryDirectory(prefix="access-backend-test-") as directory:
            work = Path(directory)
            result = subprocess.run(
                [
                    "mill",
                    "-i",
                    "buckyball.runMain",
                    "sims.p2e.ElaborateAccessBackendTest",
                    str(work / "rtl"),
                ],
                cwd=ROOT / "arch",
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (work / "test.cpp").write_text("""#include "VAccessBackendTest.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
static unsigned counts[4] = {};
extern "C" void dpi_access_write(uint32_t ol,uint32_t oh,uint32_t hl,uint32_t hh,
 uint32_t il,uint32_t ih,uint32_t shared,uint32_t pbank,uint32_t bank,uint32_t group,
 uint32_t sl,uint32_t sh,uint32_t addr,uint32_t mask,
 uint32_t d0,uint32_t d1,uint32_t d2,uint32_t d3) {
 assert(shared < 2 && pbank < 2);
 unsigned i = shared*2+pbank;
 assert(sl == counts[i]++ && sh == 0);
 assert(hl == (shared ? 4+pbank : 4) && hh == 0 && ol == hl && oh == 0);
 assert(bank == (shared ? 32+pbank : 2+pbank) && group == 0 && addr == 5);
 assert(ih == 0x1234 && il == (sl == 0 ? 0x56789abc : 0x56789abd));
 assert(d1 == 0 && d2 == 0 && d3 == 0);
 assert(mask == (sl == 0 ? 0xffff : 1) && d0 == (sl == 0 ? 0x1234 : 0x42));
}
int main(int argc, char** argv) {
 Verilated::commandArgs(argc,argv);
 VAccessBackendTest top;
 auto tick = [&](unsigned accepted) {
   top.clock=0; top.eval();
   if(top.io_accepted != accepted) fprintf(stderr,"accepted: got %u expected %u\\n",top.io_accepted,accepted);
   assert(top.io_accepted == accepted);
   top.clock=1; top.eval();
 };
 top.reset=1; top.io_valid=0; top.io_allocate=0; top.io_responseReady=0;
 top.io_data[0]=0x1234; top.io_data[1]=top.io_data[2]=top.io_data[3]=0;
 top.io_mask=0xffff; top.io_inst=0x123456789abcULL;
 tick(0); tick(0); top.reset=0;
 top.io_allocate=1; top.io_second=0; tick(0);
 top.io_second=1; tick(0); top.io_allocate=0;
 top.io_valid=15; tick(12); tick(3);
 for (int j=0; j<5; ++j) tick(0);
 top.io_valid=0; top.io_responseReady=1;
 for (int j=0; j<4; ++j) tick(0);
 top.io_inst++; top.io_mask=1; top.io_data[0]=0x42; top.io_valid=15; tick(12); tick(3);
 top.io_valid=0;
 for (int j=0; j<5; ++j) tick(0);
 for (auto count : counts) assert(count == 2);
 top.final();
}
""")
            sources = sorted(
                p for p in (work / "rtl").iterdir() if p.suffix in (".sv", ".v")
            )
            result = subprocess.run(
                [
                    "verilator",
                    "--cc",
                    "--exe",
                    "--build",
                    "--assert",
                    "-Wno-fatal",
                    "--top-module",
                    "AccessBackendTest",
                    "--Mdir",
                    str(work / "obj"),
                    "+define+BUCKYBALL_DISABLE_TRACE_DPI",
                    *map(str, sources),
                    str(work / "test.cpp"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [str(work / "obj/VAccessBackendTest")], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
