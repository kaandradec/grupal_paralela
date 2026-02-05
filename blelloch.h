#ifndef BLELLOCH_H
#define BLELLOCH_H
#pragma once

#include <vector>

void blelloch_serial(std::vector<int> &v);
void blelloch_openMP_regiones_paralelas(std::vector<int> &v);
void blelloch_simd(int *data);

#endif // BLELLOCH_H