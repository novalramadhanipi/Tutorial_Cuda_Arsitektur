#include "cuda_check.cuh"
#include <vector>
#include <cstdio>

// Kernel menggunakan Grid-Stride Loop
__global__ void vector_add(const float *a, const float *b,
                           float *c, int n)
{
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = first; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int n = 10000003;
    const int threads = 256;
    const int blocks = 256;

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97) * 0.25f;
        b[i] = (i % 31) * 0.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Inisialisasi CUDA Events untuk mengukur waktu kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Rekam waktu eksekusi kernel
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Menghitung Bandwidth Efektif Nominal (GB/s)
    double tk_sec = static_cast<double>(kernel_ms) / 1000.0;
    double bytes_transferred = 12.0 * static_cast<double>(n); // 8 bytes baca + 4 bytes tulis
    double b_effective = bytes_transferred / (tk_sec * 1e9);

    std::printf("n=%d | Blocks=%d Threads=%d | Tk=%.6f ms | B_effective=%.2f GB/s\n",
                n, blocks, threads, kernel_ms, b_effective);

    // Validasi Hasil Pembacaan
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = static_cast<double>(a[i]) + b[i];
        if (!close_enough(c[i], expected)) ++errors;
    }

    std::printf("errors=%d | first=%.2f last=%.2f\n", errors, c.front(), c.back());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return errors != 0;
}
