#include "hillis_steele.h"
#include <immintrin.h> // AVX2
#include <omp.h>       
#include <algorithm>   
#include <cstring>     
#include <cmath>       

//  SERIAL
void hillis_steele_serial(std::vector<int> &v)
{
    const size_t n = v.size();
    if (n == 0) return;

    for (size_t i = 1; i < n; ++i)
    {
        v[i] += v[i - 1];
    }
}

// SIMD (AVX2)
void hillis_steele_simd(std::vector<int> &v)
{
    int n = static_cast<int>(v.size());
    if (n == 0) return;

    int remainder = n % 8;
    int padding = (remainder == 0) ? 0 : (8 - remainder);
    if (padding > 0) v.resize(n + padding, 0);
    
    int n_padded = n + padding;
    std::vector<int> buffer(n_padded);
    
    int* curr = v.data();
    int* next = buffer.data();

    for (int stride = 1; stride < n_padded; stride *= 2)
    {
        int i = 0;
        // Fase Copia (bloques de 8)
        for (; i <= stride - 8; i += 8)
        {
            __m256i v_curr = _mm256_loadu_si256((__m256i*)&curr[i]);
            _mm256_storeu_si256((__m256i*)&next[i], v_curr);
        }
        for (; i < stride; ++i) next[i] = curr[i];

        int start_simd = i; 
        if(start_simd % 8 != 0) start_simd = (start_simd / 8 + 1) * 8;
        
        for(; i < start_simd && i < n_padded; ++i) next[i] = curr[i] + curr[i - stride];

        for (; i <= n_padded - 8; i += 8)
        {
            __m256i v_val = _mm256_loadu_si256((__m256i*)&curr[i]);
            __m256i v_prev = _mm256_loadu_si256((__m256i*)&curr[i - stride]);
            __m256i v_res = _mm256_add_epi32(v_val, v_prev);
            _mm256_storeu_si256((__m256i*)&next[i], v_res);
        }

        for (; i < n_padded; ++i) next[i] = curr[i] + curr[i - stride];

        std::swap(curr, next);
    }

    if (curr != v.data()) std::memcpy(v.data(), curr, n_padded * sizeof(int));
    if (padding > 0) v.resize(n);
}

// OpenMP Regiones
void hillis_steele_omp_regiones(std::vector<int> &v)
{
    int n = static_cast<int>(v.size());
    if (n == 0) return;

    std::vector<int> buffer(n);
    int* curr = v.data();
    int* next = buffer.data();

    #pragma omp parallel
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        int delta = (n + thread_count - 1) / thread_count;
        int start = thread_id * delta;
        int end = std::min(start + delta, n);

        for (int stride = 1; stride < n; stride *= 2)
        {
            if (start < end)
            {
                for (int i = start; i < end; i++)
                {
                    if (i >= stride) next[i] = curr[i] + curr[i - stride];
                    else             next[i] = curr[i];
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                std::swap(curr, next);
            }
        }
    }

    if (curr != v.data()) std::memcpy(v.data(), curr, n * sizeof(int));
}

// OpenMP + SIMD
void hillis_steele_omp_simd(std::vector<int> &v)
{
    int n = static_cast<int>(v.size());
    if (n == 0) return;

    int remainder = n % 8;
    int padding = (remainder == 0) ? 0 : (8 - remainder);
    if (padding > 0) v.resize(n + padding, 0);
    int n_padded = n + padding;

    std::vector<int> buffer(n_padded);
    int* curr = v.data();
    int* next = buffer.data();

    #pragma omp parallel
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        int delta = (n_padded + thread_count - 1) / thread_count;
        int start = thread_id * delta;
        int end = std::min(start + delta, n_padded);

        for (int stride = 1; stride < n_padded; stride *= 2)
        {
            if (start < end)
            {
                int copy_limit = std::min(end, stride);
                int i = start;
                
                for (; i < copy_limit; i++) next[i] = curr[i];

                i = std::max(start, stride);
                for (; i <= end - 8; i += 8)
                {
                    __m256i v_curr = _mm256_loadu_si256((__m256i*)&curr[i]);
                    __m256i v_prev = _mm256_loadu_si256((__m256i*)&curr[i - stride]);
                    __m256i v_res = _mm256_add_epi32(v_curr, v_prev);
                    _mm256_storeu_si256((__m256i*)&next[i], v_res);
                }
                // Cleanup del hilo
                for (; i < end; i++) next[i] = curr[i] + curr[i - stride];
            }

            #pragma omp barrier
            #pragma omp single
            {
                std::swap(curr, next);
            }
        }
    }

    if (curr != v.data()) std::memcpy(v.data(), curr, n_padded * sizeof(int));
    if (padding > 0) v.resize(n);
}