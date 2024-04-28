#include <iostream>

__host__ void helloWorldCPU(void)
{
    printf("Hello World CPU\n");
}

__global__ void helloWorldGPU(void)
{
    size_t row = blockDim.x * blockIdx.x + threadIdx.x;
    size_t col = blockDim.y * blockIdx.y + threadIdx.y;
    size_t ch  = blockDim.z * blockIdx.z + threadIdx.z;

    printf("Hello Wolrd GPU: row: %ld, col: %ld, ch: %ld\n", row, col, ch);
}

int main(void)
{
    helloWorldCPU();

    helloWorldGPU<<<5, 5>>>();
    cudaDeviceSynchronize();

    dim3 grid_dim(1,1,1);
    dim3 block_dim(5,5,5);
    helloWorldGPU<<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();

    return 0;
}
