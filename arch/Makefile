BBDIR = $(shell git rev-parse --show-toplevel)
ARCHDIR = $(BBDIR)/arch


BUILD_DIR = $(ARCHDIR)/build

export PATH := $(PATH):$(abspath ./utils)

# ARGS
NUM_JOBs = 16

# ARGS += +max_core_cycles=0 +uart_tx=1 +uart_tx_printf=0 +custom_boot_pin=0 +jtag_rbb_enable=0 
ARGS += +permissive +loadmem=/home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal +loadmem_addr=80000000 +custom_boot_pin=1 +permissive-off /home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal -b
# ========= verilator ==============
TOPNAME = TestHarness
INC_PATH ?= ${ARCHDIR}/src/csrc/include \
			/usr/lib/llvm-11/include \
			$(BBDIR)/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim \
			$(BBDIR)/thirdparty/chipyard/tools/DRAMSim2 \
			${ARCHDIR}/build 
# /home/mio/waste/anaconda3/envs/buckyball/share/verilator/include
# $(shell dirname $(shell which verilator))/../../share/verilator/include

VERILATOR = verilator
VERILATOR_CFLAGS += -MMD --build -cc  --trace \
				-O3 --x-assign fast --x-initial fast --noassert \
				-Wno-fatal \
				-j ${NUM_JOBs} \
				--timing
# --cc 参数告诉 Verilator 转换为 C++。 Verilator 还支持转换为 SystemC，这可以通过使用 --sc 来完成，但我们暂时不会使用此功能。
# -Wall - 打开所有 C++ 警告。不是必需的，但在您刚开始时很有用
# --trace - 启用波形跟踪	
OBJ_DIR = $(BUILD_DIR)/obj_dir
BIN = $(OBJ_DIR)/V$(TOPNAME) # 可执行文件在这

# project source
VSRCS = $(shell find $(BUILD_DIR) -name "*.v" -or -name "*.sv")
CSRCS = $(shell find $(ARCHDIR)/src/csrc $(BUILD_DIR) -name "*.c" -or -name "*.cc" -or -name "*.cpp") \

# rules for verilator
INCFLAGS = $(addprefix -I, $(INC_PATH))
CFLAGS  += $(INCFLAGS) -DTOP_NAME="\"V$(TOPNAME)\"" -std=c++17 #-Og -ggdb3 # -Og -ggdb3为添加gdb调试
LDFLAGS += -lreadline -ldramsim -lfesvr  \
			-L$(BBDIR)/thirdparty/chipyard/tools/DRAMSim2 \
			-L$(BBDIR)/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build \
			-L$(BBDIR)/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/lib 
# 使用readline函数的是否需要链接-lreadline
# ====================================

$(BIN): ${VSRCS} ${CSRCS} 
	@rm -rf $(OBJ_DIR)
	verilator $(VERILATOR_CFLAGS) +incdir+$(BUILD_DIR)\
		--top $(TOPNAME) $^\
		$(addprefix -CFLAGS , $(CFLAGS)) $(addprefix -LDFLAGS , $(LDFLAGS)) \
		--Mdir $(OBJ_DIR) --exe 
	$(MAKE) -C ${OBJ_DIR} -f V$(TOPNAME).mk $(OBJ_DIR)/V$(TOPNAME) 

.PHONY: run test verilog help compile bsp reformat checkformat clean sim wave

run: verilog $(BIN) sim
# 生成verilog代码->构建bin->仿真

test:
	mill -i __.test

verilog:
	$(shell mkdir -p $(BUILD_DIR))
#	JAVA_OPTS="-Xmx64g" -td $(BUILD_DIR)	-ll debug --jobs ${NUM_JOBs}  --output-file $(TOPNAME).sv
	mill -i __.test.runMain Elaborate  \
		--disable-annotation-unknown \
		-strip-debug-info \
		-O=debug \
		--split-verilog \
		-o=$(BUILD_DIR)
	rm $(ARCHDIR)/$(TOPNAME).sv 
# I dont know why, but it will generate a V$(TOPNAME).sv file in the arch directory, which will cause the verilog file to be generated in the arch directory

# mill -i __.test.runMain Elaborate -td /home/shiroha/Code/Backend/ml-accelerator/build -X mverilog -ll debug

# -X mverilog
# mill -i __.test.runMain Elaborate -td $(BUILD_DIR) 
# -X mverilog 生成verilog取消优化

LOG_DIR=$(ARCHDIR)/log

sim:
# LD_LIBRARY_PATH="/home/mio/Code/buckyball/thirdparty/chipyard/tools/DRAMSim2:$LD_LIBRARY_PATH"  && 
	mkdir -p $(LOG_DIR)
	$(BIN) $(ARGS) 

help:
	mill -i __.test.runMain Elaborate --help

compile:
	mill -i __.compile

bsp:
	mill -i mill.bsp.BSP/install

reformat:
	mill -i __.reformat

checkformat:
	mill -i __.checkFormat

clean:
	-rm -rf $(BUILD_DIR)

wave:
	gtkwave dump.vcd &


# include ../Makefile
