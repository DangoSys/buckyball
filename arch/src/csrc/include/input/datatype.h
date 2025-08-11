#ifndef __DATATYPE_H__
#define __DATATYPE_H__

// FP16数据类型
// typedef union {
//     uint16_t value;
//     struct {
//         uint16_t frac : 10;
//         uint16_t exp : 5;
//         uint16_t sign : 1;
//     };
// } fp16_t;

typedef union {
    uint16_t val;
    struct {
        uint16_t frac : 10;
        uint16_t exp : 5;
        uint16_t sign : 1;
    };
} fp16_t;

// typedef uint16_t fp16_t;

// typedef struct {
//     // uint16_t value;
//     uint16_t sign : 1;
//     uint16_t exp : 5;
//     uint16_t frac : 10;
// } fp16_t;

#endif