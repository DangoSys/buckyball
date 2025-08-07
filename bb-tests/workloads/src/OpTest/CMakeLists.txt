set(RISCV_GNU_TOOLCHAIN ${CYDIR}/.conda-env/riscv-tools)
set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)

set(BUCKYBALL_OPTEST_DIR ${NPU_WORKLOAD_DIR}/buckyball/OpTest)

set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-g++")  

add_custom_command(
  OUTPUT  bb_mvin_mvout_multi-baremetal bb_mvin_mvout_single-baremetal bb_mvin_mvout-linux
  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_mvin_mvout.mlir 
            -lower-buckyball="hartId=3"  |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -relocation-model=pic
            -o log.o
  COMMAND ${ELF_CC} -c ${BUCKYBALL_OPTEST_DIR}/start.S -o start.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o start.o -o bb_mvin_mvout_multi-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_mvin_mvout.mlir 
          -lower-buckyball  |
        ${BUDDY_TRANSLATE} --buddy-to-llvmir |
        ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
          -mattr=+buddyext,+D -float-abi=hard
          -relocation-model=pic
          -o log.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o -o bb_mvin_mvout_single-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_mvin_mvout.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -o log.o
  COMMAND ${LINUX_CC} -O2 -static log.o -o bb_mvin_mvout-linux
  DEPENDS ${BUCKYBALL_OPTEST_DIR}/bb_mvin_mvout.mlir
  COMMENT "Generating bb_mvin_mvout"
)

add_custom_command(
  OUTPUT  bb_dma1-baremetal bb_dma1-linux
  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma1.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -relocation-model=pic
            -o log.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o -o bb_dma1-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma1.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -o log.o
  COMMAND ${LINUX_CC} -O2 -static log.o -o bb_dma1-linux
  DEPENDS ${BUCKYBALL_OPTEST_DIR}/bb_dma1.mlir
  COMMENT "Generating bb_dma1"
)

add_custom_command(
  OUTPUT  bb_dma2-baremetal bb_dma2-linux
  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma2.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -relocation-model=pic
            -o log.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o -o bb_dma2-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma2.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -o log.o
  COMMAND ${LINUX_CC} -O2 -static log.o -o bb_dma2-linux
  DEPENDS ${BUCKYBALL_OPTEST_DIR}/bb_dma2.mlir
  COMMENT "Generating bb_dma2"
)

add_custom_command(
  OUTPUT  bb_dma3-baremetal bb_dma3-linux
  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma3.mlir 
            -expand-strided-metadata
            -convert-linalg-to-loops 
            -lower-buckyball |
            ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -relocation-model=pic
            -o log.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o -o bb_dma3-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_dma3.mlir 
            -expand-strided-metadata
            -convert-linalg-to-loops 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -o log.o
  COMMAND ${LINUX_CC} -O2 -static log.o -o bb_dma3-linux
  DEPENDS ${BUCKYBALL_OPTEST_DIR}/bb_dma3.mlir
  COMMENT "Generating bb_dma3"
) 

add_custom_command(
  OUTPUT  bb_mul_warp16-baremetal bb_mul_warp16-linux
  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_mul_warp16.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -relocation-model=pic
            -o log.o
  COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs log.o -o bb_mul_warp16-baremetal

  COMMAND ${BUDDY_OPT} ${BUCKYBALL_OPTEST_DIR}/bb_mul_warp16.mlir 
            -lower-buckyball |
          ${BUDDY_TRANSLATE} --buddy-to-llvmir |
          ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
            -mattr=+buddyext,+D -float-abi=hard
            -o log.o
  COMMAND ${LINUX_CC} -O2 -static log.o -o bb_mul_warp16-linux
          DEPENDS ${BUCKYBALL_OPTEST_DIR}/bb_mul_warp16.mlir
  COMMENT "Generating bb_mul_warp16"
)


add_custom_target(buckyball-OpTest ALL DEPENDS
  bb_mvin_mvout_multi-baremetal 
  bb_mvin_mvout_single-baremetal
  bb_mvin_mvout-linux
  bb_dma1-baremetal
  bb_dma1-linux
  bb_dma2-baremetal
  bb_dma2-linux
  bb_dma3-baremetal
  bb_dma3-linux
  bb_mul_warp16-baremetal
  bb_mul_warp16-linux
  )
