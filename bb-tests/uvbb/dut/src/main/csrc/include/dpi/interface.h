#ifndef _DPI_INTERFACE_H_
#define _DPI_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

// DPI-C接口函数声明
void sram_read(int bank_id, int addr, unsigned char data[16]);
void sram_write(int bank_id, int addr, unsigned char data[16], int mask);
void acc_read(int bank_id, int addr, int data[4]);
void acc_write(int bank_id, int addr, int data[4], int mask);
void cmd_request(unsigned char valid, unsigned char ready, int bid, int iter,
                 long long special, int rob_id);
void cmd_response(unsigned char valid, unsigned char ready, int rob_id);

#ifdef __cplusplus
}
#endif

#endif
