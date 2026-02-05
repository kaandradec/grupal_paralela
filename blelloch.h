#ifndef BLELLOCH_H
#define BLELLOCH_H
#pragma once

#include <vector>

void blellochSerial(std::vector<int> &v);
void blellochOpenMP_Scalar(std::vector<int> &v);
void blelloch_simd_8(int *data);

#endif // BLELLOCH_H