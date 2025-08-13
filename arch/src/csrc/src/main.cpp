#include "tet.h"
#include "utils/macro.h"
#include "utils/debug.h"

// #define MAX_SIM_TIME 50 最大仿真周期
vluint64_t sim_time = 0;

VerilatedContext* contextp = NULL;
VerilatedVcdC* tfp = NULL;
static VChipTop* top;

int tet_step = 1; // 记录一共走了多少步，出错时抛出，方便单步调试到周围 

//================ SIM FUNCTION =====================//
void step_and_dump_wave() {
    top->eval();
    contextp->timeInc(1);
    tfp->dump(contextp->time());
    sim_time++;
}

void sim_init(int argc, char** argv) {
    contextp = new VerilatedContext;
    contextp->commandArgs(argc, argv);
    tfp = new VerilatedVcdC;
    top = new VChipTop{contextp};

    contextp->traceEverOn(true);
    top->trace(tfp, 0);
    tfp->open("dump.vcd");

    top->reset_io = 1; top->clock_uncore = 0; step_and_dump_wave();
    top->reset_io = 1; top->clock_uncore = 1; step_and_dump_wave();
    top->reset_io = 0; top->clock_uncore = 0; step_and_dump_wave();   
} // 低电平复位

void tet_exec_once() {
    top->clock_uncore ^= 1; step_and_dump_wave();
    top->clock_uncore ^= 1; step_and_dump_wave();
    // dump_gpr(); 
    // npc_step++;
} // 翻转两次走一条指令

void sim_exit() {
    contextp->timeInc(1);
    tfp->dump(contextp->time());
    tfp->close();
    printf("The wave data has been saved to the dump.vcd\n");
}

// void init_tet() {
//     while (cpu_npc.pc != MEM_BASE) { 
//         // printf("%ld\n", cpu_npc.pc); 
//         npc_exec_once(); 
//         // npc_step--;
//     } // pc先走拍到第一条指令执行结束
// }


extern float softmax_input_float[datain_lines][datain_bandwidth];
extern fp16_t softmax_input_fp16[datain_lines][datain_bandwidth];

//================ main =====================//
int main(int argc, char *argv[]) {
    sim_init(argc, argv);

    init_monitor(argc, argv);

    // display_float_matrix(softmax_input_float, datain_lines, datain_bandwidth);

    // for (int j = 0; j < datain_lines; j++) {
    //     for (int k = 0; k < datain_bandwidth; k++) {
    //         softmax_input_fp16[j][k] = float_to_fp16(softmax_input_float[j][k]);
    //     }
    // }
    
    // display_fp16_matrix(softmax_input_fp16, datain_lines, datain_bandwidth);



    // read_fp16_array("src/main/scala/Softmax/test_data/input_1024x1024.hex", softmax_input_fp16);



    sdb_mainloop();

    sim_exit();

    // share_exp(softmax_input_fp16, softmax_input_fp16);

    // display_fp16_matrix(softmax_input_fp16, datain_lines, datain_bandwidth);


} 