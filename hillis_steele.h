#ifndef HILLIS_STEELE_H
#define HILLIS_STEELE_H

#pragma once

#include <vector>
#include <cstdint>

// 1. Serial: Inclusive Scan estándar O(W)
void hillis_steele_serial(std::vector<int> &v);

// 2. SIMD: Hillis & Steele paralelo vectorizado (AVX2)
void hillis_steele_simd(std::vector<int> &v);

// 3. OpenMP: Regiones Paralelas (Manual, sin 'for parallel')
void hillis_steele_omp_regiones(std::vector<int> &v);

// 4. OpenMP + SIMD: Mezcla de hilos manuales con vectorización AVX2
void hillis_steele_omp_simd(std::vector<int> &v);

#endif // HILLIS_STEELE_H