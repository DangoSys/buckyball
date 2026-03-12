#ifndef __MACRO_H__
#define __MACRO_H__

// macro stringizing
#define str_temp(x) #x
#define str(x) str_temp(x)

// array length
#define ARRLEN(arr) (int)(sizeof(arr) / sizeof(arr[0]))

// bit manipulation
#define BITMASK(bits) ((1ull << (bits)) - 1)
#define BITS(x, hi, lo) (((x) >> (lo)) & BITMASK((hi) - (lo) + 1))

#if !defined(likely)
#define likely(cond) __builtin_expect(cond, 1)
#define unlikely(cond) __builtin_expect(cond, 0)
#endif

#endif
