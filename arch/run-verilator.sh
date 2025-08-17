/home/mio/Code/buckyball/arch/build/obj_dir/VTestHarness \
    +permissive +loadmem=/home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal \
    +loadmem_addr=80000000 +custom_boot_pin=1  \
    +permissive-off /home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal \
    > >(tee /home/mio/Code/buckyball/arch/log/stdout.log) \
    2> >(spike-dasm > /home/mio/Code/buckyball/arch/log/disasm.log)
    