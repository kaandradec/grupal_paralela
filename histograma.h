#ifndef HISTOGRAMA_H
#define HISTOGRAMA_H
#include <vector>

inline int get_bin_index(int value, int min_val, int bin_width, int num_bins);
void histograma_serial(const std::vector<int> &data, std::vector<int> &bins,
                       int min_val, int max_val, int num_bins);
void histograma_openMP_regiones_paralelas(const std::vector<int> &data, std::vector<int> &bins,
                                          int min_val, int max_val, int num_bins);
void histograma_simd(const int *data, int n, std::vector<int> &bins,
                     int min_val, int max_val, int num_bins);
void histograma_openMP_regiones_paralelas_simd(const std::vector<int> &data, std::vector<int> &bins,
                                               int min_val, int max_val, int num_bins);

#endif // HISTOGRAMA_H