#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define ROWS 16
#define N_GROUPS 1
#define COLS (N_GROUPS * ((BANK_WIDTH) / 8))
#define ELEM_BITS 8

static elem_t input_matrix[ROWS * COLS] __attribute__((aligned(64))) = {
    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,
    13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,
    25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,
    37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,
    49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,
    61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,
    73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
    85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,
    97,   98,   99,   100,  101,  102,  103,  104,  105,  106,  107,  108,
    109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,
    121,  122,  123,  124,  125,  126,  127,  -128, -127, -126, -125, -124,
    -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112,
    -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101, -100,
    -99,  -98,  -97,  -96,  -95,  -94,  -93,  -92,  -91,  -90,  -89,  -88,
    -87,  -86,  -85,  -84,  -83,  -82,  -81,  -80,  -79,  -78,  -77,  -76,
    -75,  -74,  -73,  -72,  -71,  -70,  -69,  -68,  -67,  -66,  -65,  -64,
    -63,  -62,  -61,  -60,  -59,  -58,  -57,  -56,  -55,  -54,  -53,  -52,
    -51,  -50,  -49,  -48,  -47,  -46,  -45,  -44,  -43,  -42,  -41,  -40,
    -39,  -38,  -37,  -36,  -35,  -34,  -33,  -32,  -31,  -30,  -29,  -28,
    -27,  -26,  -25,  -24,  -23,  -22,  -21,  -20,  -19,  -18,  -17,  -16,
    -15,  -14,  -13,  -12,  -11,  -10,  -9,   -8,   -7,   -6,   -5,   -4,
    -3,   -2,   -1,   0,
};

static elem_t output_matrix[COLS * ROWS] __attribute__((aligned(64)));
static elem_t expected_matrix[COLS * ROWS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  transpose_u8_matrix(input_matrix, expected_matrix, ROWS, COLS);

  const uint32_t src = 0;
  const uint32_t dst = 1;

  bb_mem_alloc(src, 1, N_GROUPS);
  bb_mem_alloc(dst, 1, N_GROUPS);
  bb_mvin((uintptr_t)input_matrix, src, ROWS, 1);
  bb_transpose(src, dst, ROWS, ELEM_BITS);
  bb_mvout((uintptr_t)output_matrix, dst, ROWS, 1);
  bb_fence();

  if (compare_u8_matrices(output_matrix, expected_matrix, COLS, ROWS)) {
    printf("Transpose i8 16x16 PASSED\n");
    return 0;
  }
  printf("Transpose i8 16x16 FAILED\n");
  return 1;
}
