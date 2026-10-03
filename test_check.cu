#include "cuda_check.cuh"
#include <cstdio>

__global__ void dummy_kernel() {
    // Kernel kosong sederhana
}

int main() {
    // Memanggil kernel dengan 1 block dan 64 threads
    dummy_kernel<<<1, 64>>>();

    // Penerapan poin 3.1
    CUDA_CHECK(cudaGetLastError());       // Cek kesalahan launch
    CUDA_CHECK(cudaDeviceSynchronize());  // Tunggu GPU & cek kesalahan eksekusi

    std::printf("Kernel berhasil dieksekusi tanpa error!\n");
    return 0;
}
