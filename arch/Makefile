BBDIR = $(shell git rev-parse --show-toplevel)
ARCHDIR = $(BBDIR)/arch


BUILD_DIR = $(ARCHDIR)/build
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

WAVEFORM_DIR = $(ARCHDIR)/waveform
$(WAVEFORM_DIR):
	mkdir -p $(WAVEFORM_DIR)

LOG_DIR = $(ARCHDIR)/log
$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# ARGS
NUM_JOBs = 16

# Default binary path, can be overridden with BINARY=xxx
BINARY ?= 

# Default VCD path, can be overridden with VCD_PATH=xxx
VCD_PATH ?= $(WAVEFORM_DIR)/dump.vcd

# ARGS += +max_core_cycles=0 +uart_tx=1 +uart_tx_printf=0 +custom_boot_pin=0 +jtag_rbb_enable=0 
ARGS += +permissive +verbose=0 +loadmem=$(BINARY) +loadmem_addr=80000000 +custom_boot_pin=1 +permissive-off $(BINARY)
# ========= verilator =============
TOPNAME = TestHarness
INC_PATH ?= $(RISCV)/include \
			$(ARCHDIR)/thirdparty/chipyard/tools/DRAMSim2 \
			${ARCHDIR}/build \
			${ARCHDIR}/src/csrc/include 
# /home/mio/waste/anaconda3/envs/buckyball/share/verilator/include
# $(shell dirname $(shell which verilator))/../../share/verilator/include

VERILATOR = verilator
VERILATOR_CFLAGS += -MMD --build -cc --trace \
				-O3 --x-assign fast --x-initial fast --noassert \
				-Wno-fatal --timing \
				-j ${NUM_JOBs} 

OBJ_DIR = $(BUILD_DIR)/obj_dir
BIN = $(OBJ_DIR)/V$(TOPNAME) # 可执行文件在这

# project source
VSRCS = $(shell find $(BUILD_DIR) -name "*.v" -or -name "*.sv")
CSRCS = $(shell find $(ARCHDIR)/src/csrc $(BUILD_DIR) -name "*.c" -or -name "*.cc" -or -name "*.cpp") \

# rules for verilator
INCFLAGS = $(addprefix -I, $(INC_PATH))
CFLAGS  += $(INCFLAGS) -DTOP_NAME="\"V$(TOPNAME)\"" -std=c++17 -DCONFIG_VCD_PATH='"$(VCD_PATH)"' -DCONFIG_LOG_PATH='"$(LOG_PATH)"' #-Og -ggdb3 # -Og -ggdb3为添加gdb调试
LDFLAGS += -lreadline -ldramsim -lfesvr  \
		   		 -L$(ARCHDIR)/thirdparty/chipyard/tools/DRAMSim2 \
		   		 -L$(ARCHDIR)/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build \
		   		 -L$(ARCHDIR)/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/lib 
# -lreadline is need by buckyball debuger
# ====================================

$(BIN): ${VSRCS} ${CSRCS} 
	@rm -rf $(OBJ_DIR)
	verilator $(VERILATOR_CFLAGS) +incdir+$(BUILD_DIR)\
		--top $(TOPNAME) $^\
		$(addprefix -CFLAGS , $(CFLAGS)) $(addprefix -LDFLAGS , $(LDFLAGS)) \
		--Mdir $(OBJ_DIR) --exe 
	$(MAKE) -C ${OBJ_DIR} -f V$(TOPNAME).mk $(OBJ_DIR)/V$(TOPNAME) 

.PHONY: run test verilog debug help compile bsp reformat checkformat clean sim wave

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
