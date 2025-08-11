#include "tet.h"
#include "utils/macro.h"
#include "utils/debug.h"


extern uint8_t softmax_input[datain_lines][datain_bandwidth];

extern "C" void softmax_read_matrix(svBit en, int line_num, short int *line_data) {
    // printf("en = %d, numline = %d\n", en, line_num);
    if (en) {
        for (int i = 0; i < datain_bandwidth; i++) {
            line_data[i] = softmax_input[line_num][i];
        }
    } else {
        for (int i = 0; i < datain_bandwidth; i++) {
            line_data[i] = 0;
        }
    }
}

// extern "C" void softmax_read_FP16_matrix(svBit en, int line_num, int line_size, svOpenArrayHandle line_data) {
//     // 获取开放数组的基地址
//     void *base_addr = svGetArrayPtr(line_data);
//     if (en) {
//         // 遍历数组并设置每个元素的值
//         for (int i = 0; i < line_size; i++) {
//             base_addr[i] = (void*)&softmax_input[line_num][i];
//         }
//     } else {
//         // 如果 en 为 0，将数组的所有元素设置为 0
//         for (int i = 0; i < line_size; i++) {
//             shortint zero_value = 0;
//             base_addr[i] = (void*)&zero_value;
//         }
//     }
// }

// extern "C" void softmax_read_FP16_matrix(svBit en, int line_num, int line_size, svBitVecVal* line_data) {
//     const void* base_addr = svGetArrayPtr(line_data);
//     if (en) {
//         for (int i = 0; i < line_size; i++) {
//             svBitVecVal* elem_ptr = (svBitVecVal*)(base_addr + i * sizeof(svBitVecVal));
//             *elem_ptr = softmax_input[line_num][i];
//         }
//     } else {
//         for (int i = 0; i < line_size; i++) {
//             shortint zero_value = 0;
//             svBitVecVal* elem_ptr = (svBitVecVal*)(base_addr + i * sizeof(svBitVecVal));
//             *elem_ptr = zero_value;
//         }
//     }
// }

extern fp16_t softmax_input_fp16[datain_lines][datain_bandwidth];

extern "C" void softmax_read_FP16_matrix(svBit en, int line_num, short int *line_data) {
    // printf("en = %d, numline = %d\n", en, line_num);
    if (en) {
        for (int i = 0; i < datain_bandwidth; i++) {
            line_data[i] = softmax_input_fp16[line_num][i].val;
        }
    } else {
        for (int i = 0; i < datain_bandwidth; i++) {
            line_data[i] = 0;
        }
    }
}



extern uint8_t softmax_output[16][16];

extern "C" void softmax_output_trace(svBit en, int line_num, const char *line_data) {
    // printf("en = %d, numline = %d\n", en, line_num);
    if (en) {
        for (int i = 0; i < 16; i++) {
            softmax_output[line_num][i] = line_data[i];
        }
    }
}

