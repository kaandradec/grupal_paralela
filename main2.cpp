#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

#include "hillis_steele.h"

// Ajusta N a algo grande (ej. 1M o 10M) para ver diferencias reales
#define N 1000000 
#define NUM_THREADS 4

int main()
{
    using clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    omp_set_num_threads(NUM_THREADS);
    
    fmt::print("=== Benchmark Hillis & Steele Scan ===\n");
    fmt::print("Vector Size (N): {}\n", N);
    fmt::print("Threads OpenMP : {}\n\n", NUM_THREADS);
    
    // Generar Datos
    std::vector<int> datos_base(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1); 
    
    for(int i=0; i<N; i++) datos_base[i] = dis(gen);

    auto v_serial   = datos_base;
    auto v_simd     = datos_base;
    auto v_omp_reg  = datos_base;
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

    fmt::print("Serial       (First 10): {}\n", std::vector<int>(v_serial.begin(), v_serial.begin() + 10));
    fmt::print("SIMD         (First 10): {}\n", std::vector<int>(v_simd.begin(), v_simd.begin() + 10));
    fmt::print("OMP Regions  (First 10): {}\n", std::vector<int>(v_omp_reg.begin(), v_omp_reg.begin() + 10));
    fmt::print("OMP + SIMD   (First 10): {}\n", std::vector<int>(v_omp_simd.begin(), v_omp_simd.begin() + 10));

    // Tabla Comparativa
    fmt::println("\nComparativa de tiempos (ms):");
    fmt::println("Metodo             Tiempo (ms)   Speedup");
    fmt::println("----------------------------------------");
    
    fmt::println("Serial           : {:.4f} ms", time_serial_ms);
    fmt::println("SIMD (AVX2)      : {:.4f} ms", time_simd_ms, time_serial_ms / time_simd_ms);
    fmt::println("OMP Regions      : {:.4f} ms", time_omp_ms, time_serial_ms / time_omp_ms);
    fmt::println("OMP + SIMD       : {:.4f} ms", time_omp_simd_ms, time_serial_ms / time_omp_simd_ms);

    return 0;
}