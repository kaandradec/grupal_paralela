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
#include "hillis_steele.h"
#include <string>
#include <algorithm>

// [SCAN]
// BLELLOCH
#define N 1024
// #define N (1 << 20) // 1048576 elementos

// HILLIS & STEELE
// #define W 1000000
#define W 10000000
#define NUM_THREADS 4

// [HISTOGRAMA]
// #define M 100000
#define M 8000000
#define NUM_BINS 10
#define MIN_VAL 0
#define MAX_VAL 100

// Función auxiliar para imprimir una parte del vector
template <typename T>
void imprimir_vector(const std::string &etiqueta, const std::vector<T> &v, size_t n = 100)
{
  fmt::print("{}[", etiqueta);
  size_t limite = std::min(v.size(), n);
  for (size_t i = 0; i < limite; ++i)
  {
    fmt::print("{}", v[i]);
    if (i < limite - 1)
      fmt::print(", ");
  }
  if (v.size() > n)
    fmt::print(", ...");
  fmt::println("]");
}

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
  fmt::println("Número de elementos: {}", N);
  imprimir_vector("Resultado Serial:    ", v_serial);
  imprimir_vector("Resultado OMP RP:    ", v_omp);
  imprimir_vector("Resultado SIMD:      ", v_simd);
  imprimir_vector("Resultado OMP SIMD:  ", v_omp_simd);

  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo              Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial              {:.6f}", tiempo_serial * 1000);
  fmt::println("OMP RP              {:.6f}", tiempo_omp * 1000);
  fmt::println("SIMD                {:.6f}", tiempo_simd * 1000);
  fmt::println("OMP RP SIMD         {:.6f}\n", tiempo_omp_simd * 1000);
}

void ejercicio_scan_hillis_steele()
{
  using clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::duration<double, std::milli>;

  omp_set_num_threads(NUM_THREADS);

  fmt::print("=== Benchmark Hillis & Steele Scan ===\n");
  fmt::print("Vector Size (N): {}\n", W);
  fmt::print("Threads OpenMP : {}\n\n", NUM_THREADS);

  // Generar Datos
  std::vector<int> datos_base(W);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 1);

  for (int i = 0; i < W; i++)
    datos_base[i] = dis(gen);

  auto v_serial = datos_base;
  auto v_simd = datos_base;
  auto v_omp_reg = datos_base;
  auto v_omp_simd = datos_base;

  // --- SERIAL ---
  auto t0 = clock::now();
  hillis_steele_serial(v_serial);
  auto t1 = clock::now();
  double time_serial_ms = std::chrono::duration_cast<ms>(t1 - t0).count();

  // --- SIMD (AVX2) ---
  auto t2 = clock::now();
  hillis_steele_simd(v_simd);
  auto t3 = clock::now();
  double time_simd_ms = std::chrono::duration_cast<ms>(t3 - t2).count();

  // --- OpenMP Regiones ---
  auto t4 = clock::now();
  hillis_steele_omp_regiones(v_omp_reg);
  auto t5 = clock::now();
  double time_omp_ms = std::chrono::duration_cast<ms>(t5 - t4).count();

  // --- OpenMP + SIMD ---
  auto t6 = clock::now();
  hillis_steele_omp_simd(v_omp_simd);
  auto t7 = clock::now();
  double time_omp_simd_ms = std::chrono::duration_cast<ms>(t7 - t6).count();

  imprimir_vector("Serial       (First 10): ", v_serial, 10);
  imprimir_vector("SIMD         (First 10): ", v_simd, 10);
  imprimir_vector("OMP Regions  (First 10): ", v_omp_reg, 10);
  imprimir_vector("OMP + SIMD   (First 10): ", v_omp_simd, 10);

  // Tabla Comparativa
  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo             Tiempo (ms)   Speedup");
  fmt::println("----------------------------------------");

  fmt::println("Serial           : {:.4f} ms", time_serial_ms);
  fmt::println("SIMD (AVX2)      : {:.4f} ms", time_simd_ms, time_serial_ms / time_simd_ms);
  fmt::println("OMP Regions      : {:.4f} ms", time_omp_ms, time_serial_ms / time_omp_ms);
  fmt::println("OMP + SIMD       : {:.4f} ms", time_omp_simd_ms, time_serial_ms / time_omp_simd_ms);
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
  auto v_simd = std::vector<int>(NUM_BINS, 0);
  auto v_omp_simd = std::vector<int>(NUM_BINS, 0);

  using clock = std::chrono::high_resolution_clock;

  const auto t0 = clock::now();
  histograma_serial(datos, bins_serial, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_serial = std::chrono::duration<double>(clock::now() - t0).count();

  const auto t1 = clock::now();
  histograma_openMP_regiones_paralelas(datos, bins_omp, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_omp = std::chrono::duration<double>(clock::now() - t1).count();

  const auto t2 = std::chrono::high_resolution_clock::now();
  histograma_simd(datos.data(), M, v_simd, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_simd = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t2).count();

  const auto t3 = clock::now();
  histograma_openMP_regiones_paralelas_simd(datos, v_omp_simd, MIN_VAL, MAX_VAL, NUM_BINS);
  const double tiempo_omp_simd = std::chrono::duration<double>(clock::now() - t3).count();

  fmt::println("\n[HISTOGRAMA]");
  imprimir_vector("Resultado Serial:    ", bins_serial);
  imprimir_vector("Resultado OMP RP:    ", bins_omp);
  imprimir_vector("Resultado SIMD:      ", v_simd);
  imprimir_vector("Resultado OMP SIMD:  ", v_omp_simd);

  fmt::println("\nComparativa de tiempos (ms):");
  fmt::println("Metodo              Tiempo");
  fmt::println("----------------------------");
  fmt::println("Serial              {:.6f}", tiempo_serial * 1000);
  fmt::println("OMP RP              {:.6f}", tiempo_omp * 1000);
  fmt::println("SIMD                {:.6f}", tiempo_simd * 1000);
  fmt::println("OMP RP SIMD         {:.6f}\n", tiempo_omp_simd * 1000);
}

int main()
{
  fmt::println("PROYECTO: SCAN E HISTOGRAMA");

  ejercicio_scan_blelloch();
  ejercicio_scan_hillis_steele();
  ejercicio_histograma();

  return 0;
}