# ===----------------------------------------------------------------------===
# The workloads of a prefab thread alone: the tests and performance workloads
# of the thread (<thread>/workloads/thread), programs of the thread's virtual
# environment (bb-tests/workloads/lib/sims/thread).
#
#   bbdev workload --build "--thread <thread>"
#
# configures it with DESIGN_CMAKE, generated from the thread's design.pb. The
# programs are built into ${DESIGN_OUTPUT}/workloads.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(thread-workloads LANGUAGES NONE)
include("${DESIGN_CMAKE}")

add_subdirectory("${DESIGN_REPO}/${DESIGN_THREADS_0_SOURCE_PATH}/workloads/thread" thread)
