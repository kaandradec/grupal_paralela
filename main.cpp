#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vector>
#include <cstdio>
#include <chrono>

#include "blelloch.h"

// #define N 8
// #define N 64
// #define N 256
#define N 1024
int main()
{
  using clock = std::chrono::high_resolution_clock;

  fmt::println("Configuracion: N = {}", N);
  fflush(stdout);

  std::vector<int> datos_base(N);
  for (int i = 0; i < N; i++)
    datos_base[i] = i + 1; // {1,2,3,4,...,N}

  auto v_serial = datos_base;
  auto v_omp_scalar = datos_base;
  auto v_simd = datos_base;
  auto v_omp_simd = datos_base;

  const auto t0 = clock::now();
  blelloch_serial(v_serial);
  const double tiempo_serial = std::chrono::duration<double>(clock::now() - t0).count();
  fmt::println("Serial:     {}\n", v_serial);
  fflush(stdout);

  const auto t1 = clock::now();
  blelloch_openMP_regiones_paralelas(v_omp_scalar);
  const double tiempo_omp = std::chrono::duration<double>(clock::now() - t1).count();
  fmt::println("OMP RP:     {}\n", v_omp_scalar);
  fflush(stdout);

  const auto t2 = clock::now();
  blelloch_simd(v_simd.data(), static_cast<int>(v_simd.size()));
  const double tiempo_simd = std::chrono::duration<double>(clock::now() - t2).count();
  fmt::println("SIMD:       {}\n", v_simd);
  fflush(stdout);

  const auto t3 = clock::now();
  blelloch_openMP_regiones_paralelas_simd(v_omp_simd);
  const double tiempo_omp_simd = std::chrono::duration<double>(clock::now() - t3).count();
  fmt::println("OMP RP SIMD: {}\n", v_omp_simd);
  fflush(stdout);

  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo             Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial             {:.6f}", tiempo_serial * 1000);
  fmt::println("OMP RP             {:.6f}", tiempo_omp * 1000);
  fmt::println("SIMD               {:.6f}", tiempo_simd * 1000);
  fmt::println("OMP RP SIMD        {:.6f}", tiempo_omp_simd * 1000);

  return 0;
}