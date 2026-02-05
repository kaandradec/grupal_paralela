#include "blelloch.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <vector>

void blellochSerial(std::vector<int> &v)
{
  const int n = static_cast<int>(v.size());
  const int levels = static_cast<int>(std::log2(n));

  for (int d = 0; d < levels; ++d)
  {
    const int stride = 1 << (d + 1);
    for (int i = 0; i < n; i += stride)
    {
      v[i + stride - 1] += v[i + (stride / 2) - 1];
    }
  }

  v[n - 1] = 0;

  for (int d = levels - 1; d >= 0; --d)
  {
    const int stride = 1 << (d + 1);
    for (int i = 0; i < n; i += stride)
    {
      const int left = i + (stride / 2) - 1;
      const int right = i + stride - 1;
      const int temp = v[left];
      v[left] = v[right];
      v[right] += temp;
    }
  }
}

void blellochOpenMP_Scalar(std::vector<int> &v)
{
  const int n = static_cast<int>(v.size());
  const int max_threads = omp_get_max_threads();
  std::vector<int> thread_sums(max_threads, 0);
  std::vector<int> thread_offsets(max_threads, 0);

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    const int delta = (n + num_threads - 1) / num_threads;
    int start = tid * delta;
    int end = std::min(start + delta, n);

    int local_sum = 0;
    if (start < end)
    {
      for (int i = start; i < end; ++i)
        local_sum += v[i];
    }
    thread_sums[tid] = local_sum;

#pragma omp barrier

#pragma omp single
    {
      int acum = 0;
      for (int i = 0; i < num_threads; ++i)
      {
        thread_offsets[i] = acum;
        acum += thread_sums[i];
      }
    }

    if (start < end)
    {
      int current_val = thread_offsets[tid];
      for (int i = start; i < end; ++i)
      {
        const int temp = v[i];
        v[i] = current_val;
        current_val += temp;
      }
    }
  }
}

void blelloch_simd_8(int *data)
{
  __m256i v = _mm256_loadu_si256(reinterpret_cast<__m256i *>(data));

  // Up-sweep phase (in-register prefix)
  const __m256i m1 = _mm256_set_epi32(6, 6, 4, 4, 2, 2, 0, 0);
  const __m256i s1 = _mm256_add_epi32(v, _mm256_permutevar8x32_epi32(v, m1));
  v = _mm256_blend_epi32(v, s1, 0xAA);

  const __m256i m2 = _mm256_set_epi32(5, 5, 5, 5, 1, 1, 1, 1);
  const __m256i s2 = _mm256_add_epi32(v, _mm256_permutevar8x32_epi32(v, m2));
  v = _mm256_blend_epi32(v, s2, 0x88);

  const __m256i m3 = _mm256_set_epi32(3, 3, 3, 3, 3, 3, 3, 3);
  const __m256i s3 = _mm256_add_epi32(v, _mm256_permutevar8x32_epi32(v, m3));
  v = _mm256_blend_epi32(v, s3, 0x80);

  const __m256i zero_last = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
  v = _mm256_and_si256(v, zero_last);

  int tmp[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), v);

  int t;
  t = tmp[3];
  tmp[3] = tmp[7];
  tmp[7] += t;
  t = tmp[1];
  tmp[1] = tmp[3];
  tmp[3] += t;
  t = tmp[5];
  tmp[5] = tmp[7];
  tmp[7] += t;
  t = tmp[0];
  tmp[0] = tmp[1];
  tmp[1] += t;
  t = tmp[2];
  tmp[2] = tmp[3];
  tmp[3] += t;
  t = tmp[4];
  tmp[4] = tmp[5];
  tmp[5] += t;
  t = tmp[6];
  tmp[6] = tmp[7];
  tmp[7] += t;

  for (int i = 0; i < 8; ++i)
    data[i] = tmp[i];
}
