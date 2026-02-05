#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vector>
#include <cstdio> // fflush
#include <chrono>

#include "blelloch.h"

#define N 16

int main()
{
  using clock = std::chrono::high_resolution_clock;

  fmt::print("Configuracion: N = {}\n", N);
  fflush(stdout);

  std::vector<int> datos_base(N);
  for (int i = 0; i < N; i++)
    datos_base[i] = i + 1; // {1,2,3,4,...,N}

  auto v_serial = datos_base;
  auto v_omp_scalar = datos_base;
  auto v_simd = datos_base;

  const auto t0 = clock::now();
  blellochSerial(v_serial);
  const double tiempo_serial = std::chrono::duration<double>(clock::now() - t0).count();
  fmt::print("Serial:     {}\n", v_serial);
  fflush(stdout);

  const auto t1 = clock::now();
  blellochOpenMP_Scalar(v_omp_scalar);
  const double tiempo_omp = std::chrono::duration<double>(clock::now() - t1).count();
  fmt::print("OMP Scalar: {}\n", v_omp_scalar);
  fflush(stdout);

  const auto t2 = clock::now();
  blelloch_simd_8(v_simd.data());
  const double tiempo_simd = std::chrono::duration<double>(clock::now() - t2).count();
  fmt::print("SIMD:       {}\n", v_simd);
  fflush(stdout);

  fmt::println("\nComparativa de tiempos (s):");
  fmt::println("Metodo             Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial             {:.6f}", tiempo_serial);
  fmt::println("OMP Scalar         {:.6f}", tiempo_omp);
  fmt::println("SIMD               {:.6f}", tiempo_simd);

  return 0;
}