// vector_add.cu
#include <stdio.h>

__global__ void vecAdd(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N = 5;
    size_t size = N * sizeof(float);
    float A[] = {1, 2, 3, 4, 5};
    float B[] = {10, 20, 30, 40, 50};
    float C[5];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecAdd<<<1, N>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Vector C:\n");
    for (int i = 0; i < N; i++)
        printf("%f ", C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}



/*
 * ## Theoretical Concepts for CUDA Vector Addition ##
 *
 * This program implements vector addition (C = A + B) using NVIDIA's CUDA framework.
 * The core idea is to leverage the parallel processing capabilities of a GPU.
 *
 * 1. Vector Addition:
 *    - Given two vectors A and B of the same size n, their sum C is a vector where each
 *      element C[i] = A[i] + B[i].
 *    - This operation is inherently parallel because each element C[i] can be computed
 *      independently of the others.
 *
 * 2. CUDA Parallelism Model:
 *    - Kernel (`__global__` function): A function written in CUDA C/C++ that runs on the GPU.
 *      In this program, `add` is the kernel.
 *    - Threads: The basic unit of execution on the GPU. Many threads run the same kernel code
 *      in parallel.
 *    - Thread Blocks: Threads are grouped into blocks. Threads within a block can cooperate by
 *      sharing data through shared memory and synchronizing their execution.
 *    - Grid: Blocks are organized into a grid. All threads in a grid execute the same kernel.
 *    - Thread Hierarchy (Built-in Variables):
 *        - `threadIdx.x`: The index of a thread within its block (1D in this case).
 *        - `blockDim.x`: The number of threads in a block (1D).
 *        - `blockIdx.x`: The index of a block within the grid (1D).
 *    - Global Thread ID: A unique ID for each thread across the entire grid can be calculated.
 *      For a 1D grid of 1D blocks (as used here):
 *      `int tid = blockIdx.x * blockDim.x + threadIdx.x;`
 *      This `tid` is then used to map each thread to a specific element of the vectors.
 *
 * 3. Memory Spaces:
 *    - Host Memory: CPU's RAM. Vectors A, B, and C are initially created here.
 *    - Device Memory: GPU's RAM. Vectors X, Y, and Z (corresponding to A, B, C) are stored here
 *      during the GPU computation.
 *    - `cudaMalloc()`: Allocates memory on the device.
 *    - `cudaMemcpy()`: Transfers data between host and device memory.
 *        - `cudaMemcpyHostToDevice`: Copies data from CPU to GPU.
 *        - `cudaMemcpyDeviceToHost`: Copies data from GPU to CPU.
 *    - `cudaFree()`: Deallocates memory on the device.
 *
 * 4. Program Workflow:
 *    a. Initialization (Host): Vectors A and B are created and filled with values on the CPU.
 *       Memory for C is also allocated on the host.
 *    b. Device Memory Allocation (Host calls CUDA API): Memory is allocated on the GPU for X, Y, Z.
 *    c. Data Transfer (Host to Device): Contents of A and B are copied to X and Y on the GPU.
 *    d. Kernel Launch (Host calls Kernel): The `add` kernel is launched on the GPU.
 *       - `add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);`
 *       - `blocksPerGrid`: Specifies the number of thread blocks in the grid.
 *       - `threadsPerBlock`: Specifies the number of threads in each block.
 *       - The host code calculates these values to ensure one thread per vector element.
 *    e. Kernel Execution (Device): Each GPU thread executes the `add` kernel. Using its unique
 *       `tid`, it computes `Z[tid] = X[tid] + Y[tid]`.
 *    f. Data Transfer (Device to Host): The resulting vector Z is copied from the GPU to vector C on the CPU.
 *    g. Cleanup (Host calls CUDA API & C++): Memory allocated on the device (`cudaFree`) and on the
 *       host (`delete[]`) is freed.
 *
 * 5. Scalability:
 *    - The performance benefit of CUDA comes from executing thousands of threads in parallel.
 *    - For vector addition, as the vector size `N` increases, the GPU can often perform the
 *      additions much faster than a CPU executing a sequential loop, provided `N` is large
 *      enough to overcome the overhead of data transfers and kernel launch.
 *
 * 6. Error Handling (Important Note):
 *    - This program omits explicit CUDA error checking (e.g., checking the return values of
 *      `cudaMalloc`, `cudaMemcpy`, and using `cudaGetLastError()` after kernel launches).
 *    - In production code, robust error handling is essential for diagnosing issues related
 *      to GPU operations or memory.
 */
