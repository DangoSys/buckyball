#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_data[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_data[DIM * DIM] __attribute__((aligned(64)));
static elem_t expected_output[DIM * DIM] __attribute__((aligned(64)));

// CPU版本的ReLU（用于生成预期结果）
void cpu_relu(elem_t *input, elem_t *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (input[i] < 0)
        {
            output[i] = 0;
        }
        else
        {
            output[i] = input[i];
        }
    }
}

// 硬件版本的ReLU（调用加速器）
void hw_relu(elem_t *input, elem_t *output, int size)
{
    uint32_t op1_addr = spad_addr(0, 0); // 输入数据在 scratchpad bank 0
    uint32_t wr_addr = spad_addr(1, 0);  // 输出数据在 scratchpad bank 1

    // 将输入数据移动到 scratchpad
    bb_mvin((uintptr_t)input, op1_addr, size);
    bb_fence();

    // 调用 ReLU 加速器指令
    bb_relu(op1_addr, wr_addr, size);
    bb_fence();

    // 从 scratchpad 读取结果
    bb_mvout((uintptr_t)output, wr_addr, size);
    bb_fence();
}

int test_relu()
{
    printf("Initializing test data...\n");

    // 初始化随机数据（包含正负值）
    srand(42);
    for (int i = 0; i < DIM * DIM; i++)
    {
        input_data[i] = (rand() % 256) - 128; // -128到127
    }

    // CPU计算预期结果
    printf("Running CPU ReLU...\n");
    cpu_relu(input_data, expected_output, DIM * DIM);

    // 硬件计算
    printf("Running Hardware ReLU...\n");
    hw_relu(input_data, output_data, DIM * DIM);

    // 比较结果
    printf("Comparing results...\n");
    int errors = 0;
    for (int i = 0; i < DIM * DIM; i++)
    {
        if (output_data[i] != expected_output[i])
        {
            printf("Mismatch at index %d: hw=%d, expected=%d\n",
                   i, output_data[i], expected_output[i]);
            errors++;
            if (errors >= 10)
                break;
        }
    }

    if (errors == 0)
    {
        printf("Test PASSED!\n");
        return 1;
    }
    else
    {
        printf("Test FAILED with %d errors\n", errors);
        return 0;
    }
}

int main()
{
#ifdef MULTICORE
    multicore(MULTICORE);
#endif
    printf("=== ReLU Accelerator Test ===\n");
    int passed = test_relu();
    if (passed)
    {
        printf("ReLU test PASSED\n");
        return 0;
    }
    else
    {
        printf("ReLU test FAILED\n");
        return 1;
    }
#ifdef MULTICORE
    exit(0);
#endif
}