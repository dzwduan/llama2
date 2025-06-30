#ifndef __QUANT_H__
#define __QUANT_H__

#include "types.h"

void int8_quantize(QuantizedTensor *qx, float* x, int n);

void int8_dequantize(QuantizedTensor *qx, float* x, int n);


//TODO: Add AWQ quantization functions






#endif // __QUANT_H__