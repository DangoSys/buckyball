// Format Version: 2
// Reviewed against the exact exported object signatures and the declared depth4 always-on profile.

CHECKSUM: "2163818788 95260501"
INSTANCE: rx_tb.dut
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit on inactive link"
Block 2 "1857595103" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit on inactive link"
Block 3 "971991619" "$error(\"Assertion failed: CHI RX flit on inactive link\n    at Rx.scala:64 assert(io.active, \\"CHI RX flit on inactive link\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit on inactive link"
Block 5 "1857595103" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit on inactive link"
Block 6 "3528662196" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI FLITPEND must precede FLITV"
Block 10 "3823759045" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI FLITPEND must precede FLITV"
Block 11 "2663934422" "$error(\"Assertion failed: CHI FLITPEND must precede FLITV\n    at Rx.scala:65 assert(previousPend, \\"CHI FLITPEND must precede FLITV\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI FLITPEND must precede FLITV"
Block 13 "3823759045" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI FLITPEND must precede FLITV"
Block 14 "2045170107" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit without granted credit"
Block 18 "798018255" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit without granted credit"
Block 19 "74350349" "$error(\"Assertion failed: CHI RX flit without granted credit\n    at Rx.scala:66 assert(advertised =/= 0.U, \\"CHI RX flit without granted credit\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit without granted credit"
Block 21 "798018255" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX flit without granted credit"
Block 22 "2469915260" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX buffer overflow"
Block 26 "2435529851" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX buffer overflow"
Block 27 "2348576011" "$error(\"Assertion failed: CHI RX buffer overflow\n    at Rx.scala:67 assert(enqueueReady, \\"CHI RX buffer overflow\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX buffer overflow"
Block 29 "2435529851" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX buffer overflow"
Block 30 "773203258" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX credit conservation"
Block 34 "3002688943" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX credit conservation"
Block 35 "2461347743" "$error(\"Assertion failed: CHI RX credit conservation\n    at Rx.scala:69 assert(reserved <= depth.U, \\"CHI RX credit conservation\\")\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX credit conservation"
Block 37 "3002688943" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX credit conservation"
Block 38 "1033059400" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX requires coordinated reset to stop"
Block 42 "2653430091" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX requires coordinated reset to stop"
Block 43 "1798953395" "$error(\"Assertion failed: CHI RX requires coordinated reset to stop\n    at Rx.scala:70 when(wasActive)(assert(io.active, \\"CHI RX requires coordinated reset to stop\\"))\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX requires coordinated reset to stop"
Block 45 "2653430091" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI RX requires coordinated reset to stop"
Block 46 "2212642507" "$fatal;"


CHECKSUM: "2163818788 2138553741"
INSTANCE: rx_tb.dut
ANNOTATION: "Assertion-failure vector: physical FLITV requires the activated link."
Condition 2 "3562691098" "(_GEN_0 & ((~io_active))) 1 -1" (3 "11")
ANNOTATION: "Assertion-failure vector: FLITPEND must precede FLITV by a cycle."
Condition 3 "2541789932" "(_GEN_0 & ((~previousPend))) 1 -1" (3 "11")
ANNOTATION: "Assertion-failure vector: sender must hold a granted credit; independently checked by rx_bad_credit."
Condition 4 "959243910" "(_GEN_0 & (advertised == 3'b0)) 1 -1" (3 "11")
ANNOTATION: "Credit conservation proves any legal arriving flit has a reserved FIFO entry; overflow is outside legal peer behavior."
Condition 6 "183470145" "(_GEN_0 & ((~enqueueReady))) 1 -1" (3 "11")
ANNOTATION: "Depth4 invariant reserved=advertised+occupancy<=4, including coordinated reset boundary."
Condition 7 "311107905" "(((~reset)) & (reserved > 4'h4)) 1 -1" (1 "01")
ANNOTATION: "Depth4 invariant reserved<=4: grant is issued only below depth and legal enqueue replaces one advertised credit with one occupied entry."
Condition 7 "311107905" "(((~reset)) & (reserved > 4'h4)) 1 -1" (3 "11")
ANNOTATION: "Always-on profile forbids runtime deactivation except under coordinated reset; assertion-failure vector only."
Condition 8 "3706255149" "(wasActive & ((~reset)) & ((~io_active))) 1 -1" (4 "111")
ANNOTATION: "A legal arriving flit owns advertised credit>0; advertised+occupancy<=4 implies occupancy<4 and enqueueReady. This functional vector is unreachable."
Condition 9 "241394234" "(io_link_flitv & enqueueReady) 1 -1" (2 "10")

CHECKSUM: "2163818788 1283625097"
INSTANCE: rx_tb.dut
ANNOTATION: "Only reserved[3]: reset starts at zero; legal FLITV consumes an advertised entry, enqueue moves it to occupancy, and grant cannot exceed depth4. Hence reserved<=4 and bit3 is invariant zero. Bits2:0 remain measured."
Toggle reserved [3] "net reserved[3:0]"
