# MatAddBall

`MatAddBall` adds INT32 data from two virtual banks into a third virtual bank.

```
MATADD(a, b, c, iter)
```

`a`, `b`, and `c` must be different allocated vbank IDs. Their `col` values
must match. `iter` is the number of 128-bit physical lines in every group.
The ball processes group 0 through the final group in order. For each line it
reads one 128-bit word from `a` and `b`, adds four INT32 lanes modulo 2^32,
then writes one 128-bit word to `c`.

The ball has two SRAM read ports and one SRAM write port. It has no internal
SRAM. It holds all three bank channels until the command response is accepted.
