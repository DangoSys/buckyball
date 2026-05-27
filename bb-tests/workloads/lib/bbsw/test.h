#ifndef BBSW_TEST_H
#define BBSW_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

void bb_test_pass(void);
void bb_test_fail(void) __attribute__((noreturn));

#ifdef __cplusplus
}
#endif

#endif
