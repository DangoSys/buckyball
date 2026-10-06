/* Single-precision e^x function.
   Copyright (C) 2017-2025 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#pragma once
#include <stdint.h>
static const uint64_t exp_table[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f,
    0x3fef9301d0125b51, 0x3fef72b83c7d517b, 0x3fef54873168b9aa,
    0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
    0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74,
    0x3feea11473eb0187, 0x3feea589994cce13, 0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c,
    0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};
static const uint32_t inv_pio4[24] = {

    0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};
