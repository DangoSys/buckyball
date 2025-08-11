// // #include "tet.h"
// #include "utils/config.h"
// #include "input/datatype.h"
// #include <stdio.h>
// #include <stdlib.h>

// int read_fp16_array(const char *filename, fp16_t array[datain_lines][datain_bandwidth]) {
//     FILE *fp;
//     fp16_t buffer[datain_lines * datain_bandwidth];

//     // 打开文件
//     fp = fopen(filename, "r");
//     if (fp == NULL) {
//         printf("Error opening file\n");
//         return 1;
//     }

//     // 读取数据到一维数组
//     int count = 0;
//     for (int i = 0; i < datain_lines * datain_bandwidth; i++) {
//         fscanf(fp, "%x", &buffer[i]);
//         count++;
//         if (count == datain_bandwidth) {
//             fscanf(fp, "\n"); // 跳过换行符
//             count = 0;
//         }
//     }

//     fclose(fp);

//     // 将一维数组重新组织到二维数组
//     for (int i = 0; i < datain_lines; i++) {
//         for (int j = 0; j < datain_bandwidth; j++) {
//             array[i][j] = buffer[i * 1024 + j];
//         }
//     }


//     return 0;
// }


