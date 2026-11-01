#include <program.h>
#include <signal.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

static uint64_t value;

static int increment(void *core, void *argument) {
  (void)core;
  void *const *words = argument;
  value += (uintptr_t)words[0];
  return 0;
}

static int submit(void *core, void *argument) {
  (void)core;
  (void)argument;
  for (unsigned index = 0; index < 32; ++index) {
    // Let the other worker sleep before submitting another job.
    usleep(1000);
    program_job_begin(1, increment);
    program_job_argument(1);
    const uint64_t first = program_job_submit();
    program_job_begin(1, increment);
    program_job_argument(2);
    program_job_after(first);
    program_wait(program_job_submit());
    if (value != (index + 1) * 3)
      return 1;
  }
  return 0;
}

static int fail(void *core, void *argument) {
  (void)core;
  (void)argument;
  return 7;
}

static int cancelled(void *core, void *argument) {
  (void)core;
  (void)argument;
  program_job_begin(1, fail);
  const uint64_t first = program_job_submit();
  program_job_begin(1, fail);
  program_job_after(first);
  program_wait(program_job_submit());
  return 0;
}

int main(void) {
  for (unsigned run = 0; run < 2; ++run) {
    value = 0;
    if (program_run(submit, 0) || value != 96 || program_run_index() != run + 1)
      return 1;
  }
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
  return !WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT;
}
