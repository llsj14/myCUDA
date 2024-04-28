#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>

template <typename T>
void allocate_matrix_memory(int row, int col, T* matrix) {
    cudaError_t err = cudaMalloc(&matrix, row * col * sizeof(T));
    if(cudaSuccess != err) {
        printf("CUDA:ERROR: cudaMalloc failure %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

template <typename T>
void free_matrix_memory(T* matrix) {
    cudaError_t err = cudaFree(matrix);
    if(cudaSuccess != err) {
        printf("CUDA:ERROR: cudaFree failure %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void test_cublas_sgemm_ex(int m, int n, int k) {
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    allocate_matrix_memory(m, k, A);
    allocate_matrix_memory(n, k, B);
    allocate_matrix_memory(m, n, C);

    const float alpha = 1.0;
    const float beta  = 0.0;
    static cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    
    cudaDataType_t Atype, Btype, Ctype;
    Atype = Btype = Ctype= CUDA_R_32F;
    
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, Atype, m, B, Btype, n, &beta, C, Ctype, k);
    
    free_matrix_memory(A);
    free_matrix_memory(B);
    free_matrix_memory(C);
}

int main() {
    test_cublas_sgemm_ex(16, 16, 16);

    return 0;
}
