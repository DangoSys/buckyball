BUILD_DIR = $(abspath ./build)

export PATH := $(PATH):$(abspath ./utils)

# ARGS
NUM_JOBs = 256

# ========= verilator ==============
TOPNAME = top
INC_PATH ?= ${PWD}/src/csrc/include \
			/usr/lib/llvm-11/include # for disasm 

VERILATOR = verilator
VERILATOR_CFLAGS += -MMD --build -cc  --trace \
				-O3 --x-assign fast --x-initial fast --noassert \
				-Wno-fatal \
				-j ${NUM_JOBs}
#  --cc 参数告诉 Verilator 转换为 C++。 Verilator 还支持转换为 SystemC，这可以通过使用 --sc 来完成，但我们暂时不会使用此功能。
# -Wall - 打开所有 C++ 警告。不是必需的，但在您刚开始时很有用
# --trace - 启用波形跟踪	
OBJ_DIR = $(BUILD_DIR)/obj_dir
BIN = $(OBJ_DIR)/V$(TOPNAME) # 可执行文件在这

# project source
VSRCS = $(shell find $(abspath ./build) -name "*.v")
CSRCS = $(shell find $(abspath ./src/csrc) -name "*.c" -or -name "*.cc" -or -name "*.cpp")

# rules for verilator
INCFLAGS = $(addprefix -I, $(INC_PATH))
CFLAGS  += $(INCFLAGS) -DTOP_NAME="\"V$(TOPNAME)\"" #-Og -ggdb3 # -Og -ggdb3为添加gdb调试
LDFLAGS += -lreadline
# 使用readline函数的是否需要链接-lreadline
# ====================================

$(BIN): ${VSRCS} ${CSRCS} 
	@rm -rf $(OBJ_DIR)
# $(call git_commit, "sim RTL") # DO NOT REMOVE THIS LINE!!!
	verilator $(VERILATOR_CFLAGS) +incdir+$(BUILD_DIR)\
		--top $(TOPNAME) $^\
		$(addprefix -CFLAGS , $(CFLAGS)) $(addprefix -LDFLAGS , $(LDFLAGS)) \
		--Mdir $(OBJ_DIR) --exe 
	$(MAKE) -C ${OBJ_DIR} -f V$(TOPNAME).mk V$(TOPNAME) 

.PHONY: run test verilog help compile bsp reformat checkformat clean sim wave

run: verilog $(BIN) sim
# 生成verilog代码->构建bin->仿真

test:
	mill -i __.test

verilog:
	$(shell mkdir -p $(BUILD_DIR))
#	JAVA_OPTS="-Xmx64g" 	
	mill -i __.test.runMain Elaborate -td $(BUILD_DIR) -X mverilog -ll debug --jobs ${NUM_JOBs}
# mill -i __.test.runMain Elaborate -td /home/shiroha/Code/Backend/ml-accelerator/build -X mverilog -ll debug

# -X mverilog
# mill -i __.test.runMain Elaborate -td $(BUILD_DIR) 
# -X mverilog 生成verilog取消优化


sim:
	$(BIN) $(ARGS)
	  @$^

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
