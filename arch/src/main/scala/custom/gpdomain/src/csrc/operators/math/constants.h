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
static const float exp_table[32] = {
    0x1.0000000000000p+0f, 0x1.059b0e0000000p+0f, 0x1.0b55860000000p+0f,
    0x1.11301e0000000p+0f, 0x1.172b840000000p+0f, 0x1.1d48740000000p+0f,
    0x1.2387a60000000p+0f, 0x1.29e9e00000000p+0f, 0x1.306fe00000000p+0f,
    0x1.371a740000000p+0f, 0x1.3dea640000000p+0f, 0x1.44e0860000000p+0f,
    0x1.4bfdae0000000p+0f, 0x1.5342b60000000p+0f, 0x1.5ab07e0000000p+0f,
    0x1.6247ec0000000p+0f, 0x1.6a09e60000000p+0f, 0x1.71f75e0000000p+0f,
    0x1.7a11480000000p+0f, 0x1.82589a0000000p+0f, 0x1.8ace540000000p+0f,
    0x1.93737c0000000p+0f, 0x1.9c49180000000p+0f, 0x1.a5503c0000000p+0f,
    0x1.ae89fa0000000p+0f, 0x1.b7f7700000000p+0f, 0x1.c199be0000000p+0f,
    0x1.cb720e0000000p+0f, 0x1.d5818e0000000p+0f, 0x1.dfc9740000000p+0f,
    0x1.ea4afa0000000p+0f, 0x1.f507660000000p+0f,
};
static const uint32_t inv_pio4[24] = {

    0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};
