#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void add_basic(int *x, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        x[i] = x[i] * 2;
    }
}

int main()
{
    // COMPLETE THIS
    int N = 1000000;

    int *x;

    cudaMallocManaged(&x, N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
    }

    add_basic<<<1, 32>>>(x, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 2.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(x[i] - 2.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);

    return 0;
}