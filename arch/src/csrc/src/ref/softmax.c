#include "tet.h"
#include <math.h>

extern uint8_t softmax_input[16][16];
double softmax_ref[16][16];

// void softmax(uint8_t softmax_matrix[16][16]) {
//     uint8_t max = 0;
//     for (int j = 0; j < 16; j++) {
//         for (int k = 0; k < 16; k++) {
//             if (softmax_matrix[j][k] > max) max = softmax_matrix[j][k];   
//         }
//     }
//     for (int j = 0; j < 16; j++) {
//         for (int k = 0; k < 16; k++) {
//             softmax_ref[j][k] = softmax_matrix[j][k] - max;   

//         }
//     }
//     double exp_sum = 0;
//     for (int j = 0; j < 16; j++) {
//         for (int k = 0; k < 16; k++) {
//             exp_sum += exp(softmax_ref[j][k]); 
//             if (k == 15) { 
//                 for (int i = 0; i < 16; i++) {
//                     softmax_ref[j][i] = exp(softmax_ref[j][i])/exp_sum;   
//                     // printf("exp(fenzi)=%lf exp(fenmu)=%lf\n", exp(softmax_ref[j][k]), exp_sum);
//                 }
//                 break;
//             } 
//         }
//         exp_sum = 0;
//     }
// }
