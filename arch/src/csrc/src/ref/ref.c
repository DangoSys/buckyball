#include "tet.h"
#include <stdio.h>


// void display_result(uint8_t (*dut_matrix)[16], double (*ref_matrix)[16]) {
//     printf("softmax hardware result:\n");
//     for (int j = 0; j < 16; j++) {
//         for (int k = 0; k < 16; k++) {
//             printf("%4d ", dut_matrix[j][k]);    
//         }
//         printf("\n");    
//     }

//     printf("softmax software result:\n");
//     for (int j = 0; j < 16; j++) {
//         for (int k = 0; k < 16; k++) {
//             printf("%.3lf ", ref_matrix[j][k]);    
//         }
//         printf("\n");    
//     }
// }

void share_exp(fp16_t (*matrix_in)[datain_bandwidth], fp16_t (*matrix_out)[datain_bandwidth]) {
    printf("C do share exp operation:\n");
    int max = 0;
    for (int j = 0; j < datain_lines; j++) {
        for (int k = 0; k < datain_bandwidth; k++) {
            if (matrix_in[j][k].exp > max) { max = matrix_in[j][k].exp;}
        } // 找最大值
        printf("max number in this group is %d\n", max);    
        
        int shift = 0;
        printf("share exp this line:");    
        for (int k = 0; k < datain_bandwidth; k++) {    
            shift = (max == matrix_in[j][k].exp) ? 0 : (max - matrix_in[j][k].exp);
            matrix_out[j][k].exp  = max;
            matrix_out[j][k].frac = (matrix_in[j][k].exp >= 1) ? ((matrix_out[j][k].frac | 0x400) >> shift) : ((matrix_out[j][k].frac) >> shift); 
            printf("0x%x(%1d, %4d, %4d) ", matrix_in[j][k].val, matrix_in[j][k].sign, matrix_in[j][k].exp, matrix_in[j][k].frac);    
        } // 对齐指数
        printf("\n");  
        max = 0; 
    }
}
