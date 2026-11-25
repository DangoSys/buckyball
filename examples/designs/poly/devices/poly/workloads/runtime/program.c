#include <program.h>
#include <signal.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

// Two entries on hart 1, in order; the program waits for the second.
static const uint64_t table[] = {2, 0, 2, 2, 0, // counts, the first entries
                                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, // entry 0
                                 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, // entry 1
                                 0};                                 // lists
static uint64_t words[2], state[2 * PROGRAM_STATE_WORDS];
static uint64_t value;

static int increment(void *hart, void *argument) {
  (void)hart;
  value += *(const uint64_t *)argument;
  return 0;
}

static int fail(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 7;
}

static int submit(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  program_start(table, words, state);
  words[0] = 1;
  words[1] = 2;
  // Let the other worker sleep before the program dispatches.
  usleep(1000);
  program_dispatch(0, increment);
  program_dispatch(1, increment);
  program_wait(1);
  const uint64_t seen = value;
  program_finish();
  return seen % 3 != 0;
}

// The first entry fails: the second is squashed, waiting for it traps.
static int cancelled(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  program_start(table, words, state);
  program_dispatch(0, fail);
  usleep(10000);
  program_dispatch(1, fail);
  program_wait(1);
  return 0;
}

int main(void) {
  const pid_t child = fork();
  if (child < 0)
    return 1;
  if (child == 0) {
    alarm(2);
    program_run(cancelled, 0);
    _exit(1);
  }
  int status;
  if (waitpid(child, &status, 0) != child)
    return 1;
  if (!WIFSIGNALED(status) || WTERMSIG(status) == SIGALRM)
    return 1;
  for (unsigned run = 0; run < 32; ++run)
    if (program_run(submit, 0) || value != (run + 1) * 3)
      return 1;
  return 0;
}
