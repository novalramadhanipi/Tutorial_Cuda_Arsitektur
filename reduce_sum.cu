#include "cuda_check.cuh"
#include <vector>
#include <cstdio>

constexpr int BLOCK = 256; // Must be a power of two for this algorithm.

__global__ void block_sum(const int *a, int *partial, int n)
{
    __shared__ int values[BLOCK];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    values[t] = (i < n) ? a[i] : 0;
    __syncthreads();

    for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
        if (t < offset) values[t] += values[t + offset];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = values[0];
}

int main()
{
    const int n = 1000003;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<int> a(n);
    long long expected = 0;
    
    for (int i = 0; i < n; ++i) {
        a[i] = i % 7;
        expected += a[i];
    }

    int *d_a = nullptr, *d_partial_1 = nullptr, *d_partial_2 = nullptr;
    std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);
    std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(int);
    
    // Alokasi 3 buffer terpisah di VRAM
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_partial_1), partial_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_partial_2), partial_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    
    // --- TAHAP 1: Reduksi dari d_a ke d_partial_1 ---
    int current_n = n;
    int current_blocks = (current_n + BLOCK - 1) / BLOCK;
    block_sum<<<current_blocks, BLOCK>>>(d_a, d_partial_1, current_n);
    CUDA_CHECK(cudaGetLastError());

// --- TAHAP 2: Reduksi bertingkat (Recursive Reduction) di GPU ---
    int *d_in = d_partial_1;
    int *d_out = d_partial_2;
    current_n = current_blocks; // 3907 elemen
    
    int pass = 1;
    std::printf("Tahap Awal GPU: %d elemen direduksi ke %d blok\n", n, current_n);

    while (current_n > 1) {
        current_blocks = (current_n + BLOCK - 1) / BLOCK;
        
        std::printf("Pass GPU %d: Memproses %d elemen menggunakan %d blok...\n", 
                    pass++, current_n, current_blocks);
                    
        block_sum<<<current_blocks, BLOCK>>>(d_in, d_out, current_n);
        CUDA_CHECK(cudaGetLastError());
        
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
        
        current_n = current_blocks;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Reduksi GPU selesai! Hanya memindahkan 1 elemen ke CPU.\n");

    // Hasil akhir selalu berada di elemen pertama d_in (karena pointer ditukar di akhir loop)
    long long result = 0;
    int final_val = 0;
    CUDA_CHECK(cudaMemcpy(&final_val, d_in, sizeof(int), cudaMemcpyDeviceToHost));
    result = final_val;
    
    std::printf("expected=%lld result=%lld match=%s\n",
                expected, result, expected == result ? "yes" : "no");
                
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_partial_1));
    CUDA_CHECK(cudaFree(d_partial_2));
    
    return result != expected;
}
