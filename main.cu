#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>
#include <string>

//add values of array x to array y
__global__ void add(int n, float *x, float *y)
{

    for (int i = 0; i < n; i++)
    {
            y[i] += x[i];
    }
}

__global__ void block_add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main() {
    int N = 1048576;

    float *x;
    float *y;
    cudaMallocManaged(&x, N* sizeof(float));
    cudaMallocManaged(&y, N* sizeof(float));

    //init array x and y on host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    int blockSize = 512*2048;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << numBlocks << std::endl;
    auto start = std::chrono::system_clock::now();
    //add<<<1, 1>>>(N, x, y);
    block_add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();

    //Check for errors
    float maxError = 0.0f;
    for(int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "MaxError: " << maxError << std::endl;
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Time elapsed: " << elapsed_seconds.count() << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}
