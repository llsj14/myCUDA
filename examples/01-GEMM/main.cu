#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>

template <typename T>
void allocate_matrix_memory(int row, int col, T** matrix) {
    cudaError_t err = cudaMalloc((void**)matrix, row * col * sizeof(T));
    if(cudaSuccess != err) {
        printf("CUDA:ERROR: cudaMalloc failure %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

template <typename T>
void set_matrix_random_number(int row, int col, T* matrix, curandGenerator_t generator) {
	if (sizeof(T) == sizeof(float)) {
		curandStatus_t status = curandGenerateUniform(generator, matrix, row * col);
		if (CURAND_STATUS_SUCCESS != status) {
			printf("CUDA:ERROR: curandGenerationUniform failure %d\n", status);
			exit(1);
		}
	}
	else {
		printf("set matrix random number only supports float now");
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
    allocate_matrix_memory(m, k, &A);
    allocate_matrix_memory(n, k, &B);
    allocate_matrix_memory(m, n, &C);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 123456789ULL);
    set_matrix_random_number(m, k, A, generator);
    set_matrix_random_number(n, k, B, generator);

    const float alpha = 1.0;
    const float beta  = 0.0;
    static cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    
    cudaDataType_t Atype, Btype, Ctype;
    Atype = Btype = Ctype= CUDA_R_32F;
    
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, Atype, m, B, Btype, n, &beta, C, Ctype, k);

    float *C_cpu = (float*)malloc(n * k);
    cudaMemcpy(C_cpu, C, n * k, cudaMemcpyDeviceToHost);
    for (int i=0; i<10; i++) {
        printf("%f\n", C_cpu[i]);
    }
    free(C_cpu);

    curandDestroyGenerator(generator);
    free_matrix_memory(A);
    free_matrix_memory(B);
    free_matrix_memory(C);
}

int main() {
    test_cublas_sgemm_ex(16, 16, 16);

    return 0;
}
