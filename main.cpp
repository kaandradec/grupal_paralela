#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vector>
#include <cstdio>
#include <chrono>
#include <random>
#include <omp.h>
#include <numeric>

#include "blelloch.h"
#include "histograma.h"

// [SCAN]
#define N 1024

// [HISTOGRAMA]
#define M 1000
#define NUM_BINS 10
#define MIN_VAL 0
#define MAX_VAL 99

// Generación de datos aleatorios
std::vector<int> generar_datos_histograma(int cantidad, int min_val, int max_val)
{
  std::vector<int> datos(cantidad);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(min_val, max_val);
  for (int &valor : datos)
    valor = dist(gen);
  return datos;
}

void ejercicio_scan_blelloch()
{
  std::vector<int> datos_base(N);
  std::iota(datos_base.begin(), datos_base.end(), 1);

  // Validaciones simples
  if ((N & (N - 1)) != 0)
  {
    fmt::println("[SCAN] N debe ser potencia de dos.");
    return;
  }

  auto v_serial = datos_base;
  auto v_omp = datos_base;
  auto v_simd = datos_base;
  auto v_omp_simd = datos_base;

  using clock = std::chrono::high_resolution_clock;

  // Ejecución
  const auto t0 = clock::now();
  blelloch_serial(v_serial);
  const double tiempo_serial = std::chrono::duration<double>(clock::now() - t0).count();

  const auto t1 = clock::now();
  blelloch_openMP_regiones_paralelas(v_omp);
  const double tiempo_omp = std::chrono::duration<double>(clock::now() - t1).count();

  const auto t2 = clock::now();
  blelloch_simd(v_simd.data(), N);
  const double tiempo_simd = std::chrono::duration<double>(clock::now() - t2).count();

  const auto t3 = clock::now();
  blelloch_openMP_regiones_paralelas_simd(v_omp_simd);
  const double tiempo_omp_simd = std::chrono::duration<double>(clock::now() - t3).count();

  fmt::println("[SCAN]");
  fmt::println("Resultado Serial:    {}", v_serial);
  fmt::println("Resultado OMP RP:    {}", v_omp);
  fmt::println("Resultado SIMD:      {}", v_simd);
  fmt::println("Resultado OMP SIMD:  {}", v_omp_simd);

  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo              Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial              {:.6f}", tiempo_serial * 1000);
  fmt::println("OMP RP              {:.6f}", tiempo_omp * 1000);
  fmt::println("SIMD                {:.6f}", tiempo_simd * 1000);
  fmt::println("OMP RP SIMD         {:.6f}", tiempo_omp_simd * 1000);
}

void ejercicio_histograma()
{
  std::vector<int> datos = generar_datos_histograma(M, MIN_VAL, MAX_VAL);

  // Validaciones simples
  if (MAX_VAL < MIN_VAL || NUM_BINS <= 0)
  {
    fmt::println("[HISTOGRAMA] Configuracion invalida.");
    return;
  }

  std::vector<int> bins_serial(NUM_BINS, 0);
  std::vector<int> bins_omp(NUM_BINS, 0);

  using clock = std::chrono::high_resolution_clock;

  const auto t0 = clock::now();
  histograma_serial(datos, bins_serial, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_serial = std::chrono::duration<double>(clock::now() - t0).count();

  const auto t1 = clock::now();
  histograma_openMP_regiones_paralelas(datos, bins_omp, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_omp = std::chrono::duration<double>(clock::now() - t1).count();

  fmt::println("\n[HISTOGRAMA]");
  fmt::println("Resultado Serial:    {}", bins_serial);
  fmt::println("Resultado OMP RP:    {}", bins_omp);

  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo              Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial              {:.6f}", tiempo_serial * 1000);
  fmt::println("OMP RP              {:.6f}", tiempo_omp * 1000);
}

int main()
{
  fmt::println("PROYECTO: SCAN E HISTOGRAMA");

  ejercicio_scan_blelloch();
  ejercicio_histograma();

  return 0;
}