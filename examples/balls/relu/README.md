# ReluBall

`ReluBall` computes signed INT32 `max(x, 0)` in place in one physical pbank.
It has no internal SRAM: a 128-bit register holds one pbank line between its
read and write requests.

## Command

`RELU(vbank, group, iter, stride)` selects physical pbank `group` in virtual
bank `vbank`. `group` is a pbank number, never a data-layout term.

`iter` is the number of 128-bit physical lines processed in each segment.
`stride` is the distance between segments. Both are positive; `iter` is a
multiple of 16, `iter <= stride`, and `stride` divides the pbank depth. The
command processes `iter` lines at every `stride` boundary until it reaches the
end of the selected pbank.

For every line the Ball performs one single-port sequence:

```text
pbank read -> 4 signed int32 ReLUs -> pbank write
```

The selected group must be allocated. There is no output vbank, no fallback
layout, and no tail path. Tile lowering pads logical matrices to 16-aligned
shapes; compiler packing fills the final pbank chunk before issuing ReLU.
