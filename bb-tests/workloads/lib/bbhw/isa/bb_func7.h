#ifndef FUNCT7_H
#define FUNCT7_H

// The funct7 of instruction `mne` of the Ball in directory `ball`, from the
// compile definitions of the target (DESIGN_TARGET_DEFS in design.cmake).
#define FUNCT7(ball, mne)                                                      \
  THREAD_BALLDOMAIN_BALLS_##ball##_INSTRUCTIONS_##mne##_FUNCT7

#endif
