#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A masked attention of 2 key heads, 24 query rows each (3 query heads of 8
 * queries) and 96 keys of 64 values in two parts (64, then 32). Query row i
 * of head h has 16 at value i and key (5 i + 7 h + 3) % 96 has 64 there, so
 * its score is 1024 at that key and 0 at the others, 128 and 0 scaled by
 * 1/8. The bias row of query row i is row i % 8, 0 for a key the row sees
 * and -1e30 for the others: an even row sees every key
 * but one that is no query's of the row, so the softmax is exactly 1 at the
 * query's key and the output row is that key's row of v; an odd row sees
 * four keys, none a query's of the row, so the softmax is exactly 1/4 at
 * each and the output row is the mean of their rows of v. v holds small
 * integers, which MXINT8 holds exactly, as it does 1 and 1/4. */
struct tensor3 {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
struct tensor4 {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
extern void _mlir_ciface_compute(struct tensor3 *, struct tensor4 *,
                                 struct tensor4 *, struct tensor4 *,
                                 struct tensor4 *, struct tensor3 *,
                                 struct tensor3 *);
static float q[2][24][64] __attribute__((aligned(64)));
static float kp[1][64][2][64] __attribute__((aligned(64)));
static float kn[1][32][2][64] __attribute__((aligned(64)));
static float bias[1][8][96] __attribute__((aligned(64)));
/* v and the output are written whole: from the heap, which the start code
 * does not clear (clearing them took most of the test's cycles). */
static float (*vp)[64][2][64], (*vn)[32][2][64], (*o)[24][64];

static float pattern[13][64];

static int key(int h, int i) { return (5 * i + 7 * h + 3) % 96; }

/* Whether key t is the key of a query row of mask row r. */
static int taken(int r, int t) {
  for (int h = 0; h < 2; ++h)
    for (int i = r; i < 24; i += 8)
      if (key(h, i) == t)
        return 1;
  return 0;
}

/* Row t of v of head h. */
static const float *row(int h, int t) {
  return t < 64 ? vp[0][t][h] : vn[0][t - 64][h];
}

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[16];
  kernel_init(&kernel, slots, 16);
  vp = aligned_alloc(64, sizeof(float[1][64][2][64]));
  vn = aligned_alloc(64, sizeof(float[1][32][2][64]));
  o = aligned_alloc(64, sizeof(float[2][24][64]));
  struct tensor3 qm = {&q[0][0][0], &q[0][0][0], 0, {2, 24, 64}, {1536, 64, 1}};
  struct tensor4 kpm = {
      &kp[0][0][0][0], &kp[0][0][0][0], 0, {1, 64, 2, 64}, {8192, 128, 64, 1}};
  struct tensor4 knm = {
      &kn[0][0][0][0], &kn[0][0][0][0], 0, {1, 32, 2, 64}, {4096, 128, 64, 1}};
  struct tensor4 vpm = {
      &vp[0][0][0][0], &vp[0][0][0][0], 0, {1, 64, 2, 64}, {8192, 128, 64, 1}};
  struct tensor4 vnm = {
      &vn[0][0][0][0], &vn[0][0][0][0], 0, {1, 32, 2, 64}, {4096, 128, 64, 1}};
  struct tensor3 bm = {
      &bias[0][0][0], &bias[0][0][0], 0, {1, 8, 96}, {768, 96, 1}};
  struct tensor3 om = {&o[0][0][0], &o[0][0][0], 0, {2, 24, 64}, {1536, 64, 1}};
  for (int s = 0; s < 13; ++s)
    for (int d = 0, x = s; d < 64; ++d, x = x == 12 ? 0 : x + 1)
      pattern[s][d] = (float)(x - 6);
  for (int h = 0; h < 2; ++h) {
    for (int i = 0; i < 24; ++i) {
      int t = key(h, i);
      q[h][i][i] = 16;
      if (t < 64)
        kp[0][t][h][i] = 64;
      else
        kn[0][t - 64][h][i] = 64;
    }
    /* v[t][d] = (7 t + d + h) % 13 - 6: one of 13 rows, copied. */
    for (int t = 0; t < 96; ++t)
      memcpy(t < 64 ? vp[0][t][h] : vn[0][t - 64][h], pattern[(t * 7 + h) % 13],
             sizeof(pattern[0]));
  }
  /* The four keys an odd row sees, the one an even row does not. */
  int four[8][4], hidden[8];
  for (int r = 0; r < 8; ++r) {
    int count = 0;
    for (int t = 95 - 9 * r; count < 4; t = (t + 29) % 96)
      if (!taken(r, t))
        four[r][count++] = t;
    hidden[r] = four[r][0];
    for (int t = 0; t < 96; ++t)
      bias[0][r][t] = (r % 2 == 0 ? t != hidden[r]
                                  : t == four[r][0] || t == four[r][1] ||
                                        t == four[r][2] || t == four[r][3])
                          ? 0.0f
                          : -1e30f;
  }
  _mlir_ciface_compute(&qm, &kpm, &knm, &vpm, &vnm, &bm, &om);
  for (int h = 0; h < 2; ++h)
    for (int i = 0; i < 24; ++i) {
      int r = i % 8;
      const float *own = row(h, key(h, i)), *a = row(h, four[r][0]),
                  *b = row(h, four[r][1]), *c = row(h, four[r][2]),
                  *e = row(h, four[r][3]);
      for (int d = 0; d < 64; ++d) {
        float expected = r % 2 ? (a[d] + b[d] + c[d] + e[d]) * 0.25f : own[d];
        if (o[h][i][d] != expected) {
          printf("Masked attention mismatch: head %d query %d value %d: "
                 "%d/4 != %d/4\n",
                 h, i, d, (int)(o[h][i][d] * 4), (int)(expected * 4));
          return 1;
        }
      }
    }
  puts("Compiled masked attention in the thread's banks passed");
  return 0;
}
