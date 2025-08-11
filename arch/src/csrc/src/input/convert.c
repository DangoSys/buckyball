#include "tet.h"


fp16_t float_to_fp16(float val) {
    /*
        1、强制把float转为unsigned long（32bit），用来取对应bit的数
        2、截取后23位尾数，右移13位，剩余10位,得到了fp16的尾数位(取高10位)
        3、符号位直接右移16位,得到了fp16的符号位
        4、截取指数的8位先右移13位，这里得到8位，但是我们只用到5位，左边多出的3位不用管
        由于之前0~255表示-127~128，调整过后变成 0~31表示-15~16,所以fp16情况下指数位是+15,
        所以这里需要在移位后指数的基础上减去(127-15) = 112(左移10位，因为指数是在bit10开始),
    */
    fp16_t result;
    uint32_t raw = *(uint32_t *)&val; // 将float视为32位整数

    result.sign = (raw & 0x80000000) >> 31 ;
    result.exp  = ((raw >> 23) & 0xff) - 127 + 15;
    result.frac = (raw >> 13) & 0x3ff;
    // result.val  = (result.sign << 15) |  (result.exp << 10) | result.frac; 

    if(raw & 0x1000) {
        result.val += 1;
    } // 四舍五入(尾数被截掉部分的最高位为1, 则尾数剩余部分+1)

    fp16_t converted = *(fp16_t *)&result.val; 
    return converted;
}


// 从FP16数据中解包出两个8位无符号整数
// void fp16_unpack(fp16 fp16, uint8_t* sign, uint8_t* exponent, uint16_t* fraction) {
//     *sign     = fp16.sign;
//     *exponent = fp16.exp;
//     *fraction = fp16.frac;
// }