/*
 * Copyright (C) 2018-2019 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 */

SEARCH_DIR(.)
__DYNAMIC  =  0;

MEMORY
{
    instrram    : ORIGIN = 0xcc000000, LENGTH = 0x8000
    dataram     : ORIGIN = 0xcc010000, LENGTH = 0xF00000  /* MAGIA: data at 0xCC010000 */
    stack       : ORIGIN = 0x00010000, LENGTH = 0x0C000   /* CV32/PULP stacks below Spatz [0x1C000, 0x20000) */
}

/* Stack information variables.
 * Spatz owns [0x1C000, 0x20000) in the tile SPM (_stack_start = 0x1FFF8 in
 * spatz_program.ld); this region stops at 0x1C000 to avoid overlap.
 * Each hart gets _stack_slice_size bytes; crt0 computes:
 *   sp = stack - local_id * _stack_slice_size
 * Slot 0 = CV32 main core, slots 1..8 = PULP cluster cores. */
_stack_slice_size = 0x800;                       /* 2 KB per hart */
_stack_hart_count = 9;                           /* 1 main + 8 PULP cores */
_min_stack        = _stack_slice_size * _stack_hart_count;  /* 18 KB total */
_stack_len        = LENGTH(stack);
_stack_start      = ORIGIN(stack) + LENGTH(stack);
ASSERT(_min_stack <= _stack_len, "Configured stacks exceed stack memory");

/* We have to align each sector to word boundaries as our current s19->slm
 * conversion scripts are not able to handle non-word aligned sections. */

PHDRS
{
    text PT_LOAD;
    data PT_LOAD;
}

SECTIONS
{
    .vectors :
    {
        . = ALIGN(4);
        KEEP(*(.vectors))
    } > instrram :text

    .text : {
        . = ALIGN(4);
        _stext = .;
        *(.text)
        _etext  =  .;
        __CTOR_LIST__ = .;
        LONG((__CTOR_END__ - __CTOR_LIST__) / 4 - 2)
        *(.ctors)
        LONG(0)
        __CTOR_END__ = .;
        __DTOR_LIST__ = .;
        LONG((__DTOR_END__ - __DTOR_LIST__) / 4 - 2)
        *(.dtors)
        LONG(0)
        __DTOR_END__ = .;
        *(.lit)
        *(.shdata)
        _endtext = .;
    }  > instrram :text

    /* Spatz embedded binary - inlined from header file */
    /* Positioned right after CV32 .text in instrram */

    . = ALIGN(4);
    _spatz_binary_start = .;

    .spatz_binary : {
        KEEP(*(.spatz_binary))
    } > instrram :text

    _spatz_binary_end = .;
    _spatz_binary_size = _spatz_binary_end - _spatz_binary_start;

    /* PULP embedded binary - inlined from header file (magia-sdk flow).
       CV32E40P stores mtvec as mtvec_addr_i[31:8], so keep the PULP
       boot address and trap base in the same 256-byte page. */
    .pulp_binary ALIGN(256) : {
        _pulp_binary_start = .;
        KEEP(*(.pulp_binary))
        _pulp_binary_end = .;
    } > instrram :text

    _pulp_binary_size = _pulp_binary_end - _pulp_binary_start;

    /*--------------------------------------------------------------------*/
    /* Global constructor/destructor segment                              */
    /*--------------------------------------------------------------------*/

    .preinit_array     :
    {
      PROVIDE_HIDDEN (__preinit_array_start = .);
      KEEP (*(.preinit_array))
      PROVIDE_HIDDEN (__preinit_array_end = .);
    } > dataram :data

    .init_array     :
    {
      PROVIDE_HIDDEN (__init_array_start = .);
      KEEP (*(SORT(.init_array.*)))
      KEEP (*(.init_array ))
      PROVIDE_HIDDEN (__init_array_end = .);
    } > dataram :data

    .fini_array     :
    {
      PROVIDE_HIDDEN (__fini_array_start = .);
      KEEP (*(SORT(.fini_array.*)))
      KEEP (*(.fini_array ))
      PROVIDE_HIDDEN (__fini_array_end = .);
    } > dataram :data

    .data : {
        . = ALIGN(4);
        sdata  =  .;
        _sdata  =  .;
        KEEP(*(.data.bb_started))
        *(.data);
        *(.data.*)
        edata  =  .;
        _edata  =  .;
    } > dataram :data
    ASSERT(bb_started == ORIGIN(dataram), "MAGIA boot flag must start the data segment")

    .rodata : {
        . = ALIGN(4);
        *(.rodata);
        *(.rodata.*)
    } > dataram :data

    .shbss :
    {
        . = ALIGN(4);
        *(.shbss)
    } > dataram :data

    .bss :
    {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
        _bss_end = .;
    } > dataram :data

    /* Stack placed at the TOP of the CV32/PULP region so crt0 can compute
     * per-hart SP = stack - local_id * _stack_slice_size. */
    .stack ORIGIN(stack) + LENGTH(stack) - _min_stack (NOLOAD): {
        . = ALIGN(4);
        _stack_bottom = .;
        . = . + _min_stack ;
        . = ALIGN(4);
        stack = . ;
        _stack = . ;
    } > stack :NONE

}
