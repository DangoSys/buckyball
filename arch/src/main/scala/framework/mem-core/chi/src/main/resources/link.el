// Format Version: 2
// Reviewed against the exact exported object signatures and the declared depth4 always-on profile.

CHECKSUM: "1668538250 4134186348"
INSTANCE: link_tb.dut
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit overflow"
Block 2 "239387013" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit overflow"
Block 3 "4289838947" "$error(\"Assertion failed: CHI TX credit overflow\n    at Tx.scala:43 when(returned && !send)(assert(credits < maxCredits.U, \\"CHI TX credit overflow\\"))\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit overflow"
Block 5 "239387013" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit overflow"
Block 6 "1750606003" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit underflow"
Block 10 "1503763571" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit underflow"
Block 11 "556320376" "$error(\"Assertion failed: CHI TX credit underflow\n    at Tx.scala:44 when(send)(assert(credits =/= 0.U, \\"CHI TX credit underflow\\"))\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit underflow"
Block 13 "1503763571" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX credit underflow"
Block 14 "2166023300" "$fatal;"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX requires coordinated reset to stop"
Block 18 "2653430091" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX requires coordinated reset to stop"
Block 19 "597017625" "$error(\"Assertion failed: CHI TX requires coordinated reset to stop\n    at Tx.scala:45 when(wasActive)(assert(io.active, \\"CHI TX requires coordinated reset to stop\\"))\n\");"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX requires coordinated reset to stop"
Block 21 "2653430091" "if (1)"
ANNOTATION: "Assertion-failure diagnostic only; assertion remains enabled: CHI TX requires coordinated reset to stop"
Block 22 "2212642507" "$fatal;"


CHECKSUM: "1668538250 4069976122"
INSTANCE: link_tb.dut
ANNOTATION: "Tx32/depth4 peer: credit=4 means all four credits are available; another returned credit cannot simultaneously be pending. This vector would violate peer credit conservation."
Condition 1 "284300870" "(returned & ((~send)) & ((~reset)) & credits[2]) 1 -1" (2 "1011")
ANNOTATION: "Tx32/depth4 peer: no returned credit can coexist with all four credits available, including the coordinated reset boundary."
Condition 1 "284300870" "(returned & ((~send)) & ((~reset)) & credits[2]) 1 -1" (3 "1101")
ANNOTATION: "Assertion-failure vector: overgrant violates the declared four-credit peer contract. Independently checked by tx_bad_credit."
Condition 1 "284300870" "(returned & ((~send)) & ((~reset)) & credits[2]) 1 -1" (5 "1111")
ANNOTATION: "Structural invariant: send requires ready, and ready requires nonzero credits; send with zero credits is impossible even on reset."
Condition 2 "3481575346" "(send & ((~reset)) & ((~(|credits)))) 1 -1" (2 "101")
ANNOTATION: "Structural invariant: send implies credits != 0 through io.in.ready."
Condition 2 "3481575346" "(send & ((~reset)) & ((~(|credits)))) 1 -1" (4 "111")
ANNOTATION: "Always-on profile forbids deactivation without coordinated reset; this is the assertion-failure vector, not normal shutdown support."
Condition 3 "3706255149" "(wasActive & ((~reset)) & ((~io_active))) 1 -1" (4 "111")
