#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

// AUXILIAR: Calcular el índice del bin
inline int get_bin_index(int value, int min_val, int bin_width, int num_bins)
{
  int idx = (value - min_val) / bin_width;
  if (idx >= num_bins)
    idx = num_bins - 1;
  if (idx < 0)
    idx = 0;
  return idx;
}

// 1. VERSIÓN SERIAL
void histograma_serial(const std::vector<int> &data, std::vector<int> &bins,
                       int min_val, int max_val, int num_bins)
{
  // Limpieza inicial
  std::fill(bins.begin(), bins.end(), 0);

  int range = max_val - min_val + 1;
  int bin_width = std::ceil((double)range / num_bins);

  for (const int &val : data)
  {
    int idx = get_bin_index(val, min_val, bin_width, num_bins);
    bins[idx]++;
  }
}

// 2. VERSIÓN OPENMP REGIONES PARALELAS
void histograma_openMP_regiones_paralelas(const std::vector<int> &data, std::vector<int> &bins,
                                          int min_val, int max_val, int num_bins)
{
  // Limpieza inicial del global
  std::fill(bins.begin(), bins.end(), 0);

  int range = max_val - min_val + 1;
  int bin_width = std::ceil((double)range / num_bins);
  int n = static_cast<int>(data.size());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    const int delta = (n + num_threads - 1) / num_threads;
    int start = tid * delta;
    int end = std::min(start + delta, n);

    // Histograma local por hilo
    std::vector<int> local_bins(num_bins, 0);

    if (start < end)
    {
      for (int i = start; i < end; ++i)
      {
        int val = data[i];
        int idx = get_bin_index(val, min_val, bin_width, num_bins);
        local_bins[idx]++;
      }
    }

#pragma omp critical
    {
      for (int b = 0; b < num_bins; ++b)
      {
        bins[b] += local_bins[b];
      }
    }
  }
}

// 3. VERSIÓN SIMD (AVX2)
void histograma_simd(const int *data, int n, std::vector<int> &bins,
                     int min_val, int max_val, int num_bins)
{
  std::fill(bins.begin(), bins.end(), 0);

  int range = max_val - min_val + 1;
  float bin_width = std::ceil((double)range / num_bins);
  float inv_bin_width = 1.0f / bin_width;

  __m256 v_min = _mm256_set1_ps((float)min_val);
  __m256 v_inv_width = _mm256_set1_ps(inv_bin_width);
  __m256 v_max_bin = _mm256_set1_ps((float)(num_bins - 1));
  __m256 v_zero = _mm256_setzero_ps();

  int i = 0;
  // Procesar de 8 en 8 elementos
  for (; i + 8 <= n; i += 8)
  {
    // 1. Cargar 8 enteros y convertirlos a float
    __m256i v_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data[i]));
    __m256 v_floats = _mm256_cvtepi32_ps(v_data);

    // 2. Calcular índices: (val - min) / width
    __m256 v_sub = _mm256_sub_ps(v_floats, v_min);
    __m256 v_idx = _mm256_mul_ps(v_sub, v_inv_width);

    // 3. Truncar a entero y aplicar límites (Clamping)
    v_idx = _mm256_floor_ps(v_idx);
    v_idx = _mm256_min_ps(_mm256_max_ps(v_idx, v_zero), v_max_bin);

    // 4. Convertir de vuelta a int para usar como índices
    __m256i v_final_idx = _mm256_cvtps_epi32(v_idx);

    // extraemos los índices para actualizar los bins.
    int indices[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(indices), v_final_idx);

    for (int k = 0; k < 8; ++k)
    {
      bins[indices[k]]++;
    }
  }

  // Procesar remanentes
  for (; i < n; ++i)
  {
    int idx = (data[i] - min_val) / (int)bin_width;
    if (idx < 0)
      idx = 0;
    if (idx >= num_bins)
      idx = num_bins - 1;
    bins[idx]++;
  }
}

void histograma_openMP_regiones_paralelas_simd(const std::vector<int> &data, std::vector<int> &bins,
                                               int min_val, int max_val, int num_bins)
{
  // Limpieza inicial del global
  std::fill(bins.begin(), bins.end(), 0);

  const int n = static_cast<int>(data.size());
  const int range = max_val - min_val + 1;
  const float bin_width = std::ceil((double)range / num_bins);
  const float inv_bin_width = 1.0f / bin_width;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    // Reparto manual de carga
    const int delta = (n + num_threads - 1) / num_threads;
    int start = tid * delta;
    int end = std::min(start + delta, n);

    // Histograma local para evitar colisiones entre hilos
    std::vector<int> local_bins(num_bins, 0);

    // Constantes SIMD (viven en registros de cada hilo)
    __m256 v_min = _mm256_set1_ps((float)min_val);
    __m256 v_inv_width = _mm256_set1_ps(inv_bin_width);
    __m256 v_max_bin = _mm256_set1_ps((float)(num_bins - 1));
    __m256 v_zero = _mm256_setzero_ps();

    int i = start;
    // Bucle SIMD dentro del rango del hilo
    for (; i + 8 <= end; i += 8)
    {
      // Cargar y convertir a float
      __m256i v_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data[i]));
      __m256 v_floats = _mm256_cvtepi32_ps(v_data);

      // Calcular índices vectorialmente
      __m256 v_sub = _mm256_sub_ps(v_floats, v_min);
      __m256 v_idx = _mm256_mul_ps(v_sub, v_inv_width);

      // Truncar y Clamping
      v_idx = _mm256_floor_ps(v_idx);
      v_idx = _mm256_min_ps(_mm256_max_ps(v_idx, v_zero), v_max_bin);

      // Convertir a enteros
      __m256i v_final_idx = _mm256_cvtps_epi32(v_idx);

      // Extraer a memoria temporal para actualizar el histograma LOCAL
      alignas(32) int indices[8];
      _mm256_store_si256(reinterpret_cast<__m256i *>(indices), v_final_idx);

      for (int k = 0; k < 8; ++k)
      {
        local_bins[indices[k]]++;
      }
    }

    // Remanentes del hilo (escalar)
    for (; i < end; ++i)
    {
      int idx = (data[i] - min_val) / (int)bin_width;
      if (idx < 0)
        idx = 0;
      if (idx >= num_bins)
        idx = num_bins - 1;
      local_bins[idx]++;
    }

// Reducción al histograma GLOBAL
#pragma omp critical
    {
      for (int b = 0; b < num_bins; ++b)
      {
        bins[b] += local_bins[b];
      }
    }
  }
}