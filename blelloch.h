#ifndef BLELLOCH_H
#define BLELLOCH_H
#pragma once

#include <vector>

void blelloch_serial(std::vector<int> &v);
void blelloch_openMP_regiones_paralelas(std::vector<int> &v);
void blelloch_simd(int *data, int n);
void blelloch_openMP_regiones_paralelas_simd(std::vector<int> &v);

#endif // BLELLOCH_H