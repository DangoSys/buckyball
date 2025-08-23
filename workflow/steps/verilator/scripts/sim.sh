#!/bin/bash

/home/mio/Code/buckyball/arch/build/obj_dir/VTestHarness \                                                                                                                    [0]
    +permissive +loadmem=/home/mio/Code/buckyball/arch/workload/hello \
    +loadmem_addr=800000000  \
    +permissive-off /home/mio/Code/buckyball/arch/workload/hello 