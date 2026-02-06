#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

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