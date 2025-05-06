// matrix_mul.cu
#include <stdio.h>

__global__ void matMul(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 2; // 2x2 matrix for simplicity
    size_t size = N * N * sizeof(float);
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix C:\n");
    for (int i = 0; i < N * N; i++)
    {
        printf("%f ", C[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}




/*
 * ## Conceptual Overview of the CUDA Matrix Multiplication Program ##
 *
 * This program demonstrates how to perform matrix multiplication (C = A * B)
 * using NVIDIA's CUDA platform, leveraging the parallel processing power of a GPU.
 *
 * 1. Matrix Multiplication Basics:
 *    - If A is an (m x p) matrix and B is a (p x n) matrix, their product C will be an (m x n) matrix.
 *    - Each element C[row][col] is calculated as the dot product of the 'row'-th row of A
 *      and the 'col'-th column of B.
 *    - C[row][col] = sum(A[row][k] * B[k][col]) for k from 0 to p-1.
 *    - In this program, all matrices are N x N (square).
 *
 * 2. CUDA Parallelization Strategy:
 *    - The core idea is to assign the computation of each element of the result matrix C
 *      to a separate CUDA thread.
 *    - GPU threads are organized into a grid of thread blocks. For a 2D matrix, it's
 *      natural to use a 2D grid of 2D thread blocks.
 *    - `threadIdx.x`, `threadIdx.y`: Give the coordinates of a thread within its block.
 *    - `blockIdx.x`, `blockIdx.y`: Give the coordinates of a thread block within the grid.
 *    - `blockDim.x`, `blockDim.y`: Give the dimensions of a thread block.
 *    - The global row and column in the matrix that a thread is responsible for is calculated as:
 *        `int row = blockIdx.y * blockDim.y + threadIdx.y;`
 *        `int col = blockIdx.x * blockDim.x + threadIdx.x;`
 *    - `dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);` defines each block to have BLOCK_SIZE x BLOCK_SIZE threads.
 *    - `dim3 grid(K, K);` defines the grid to have K x K blocks. Since N = K * BLOCK_SIZE, the total
 *      number of threads matches the N x N elements of the matrix.
 *
 * 3. Memory Management:
 *    - Host (CPU) Memory: Matrices `hA`, `hB` are initially created and populated on the CPU's RAM
 *      using `new float[N*N]`.
 *    - Device (GPU) Memory: Memory for matrices `dA`, `dB`, `dC` on the GPU's RAM is allocated
 *      using `cudaMalloc()`.
 *    - Data Transfers:
 *        - `cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);` copies matrix A from host to device.
 *        - `cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);` copies the result matrix C from device back to host.
 *    - Deallocation: It's crucial to free allocated memory to prevent leaks.
 *        - `delete[]` for host memory allocated with `new[]`.
 *        - `cudaFree()` for device memory allocated with `cudaMalloc()`.
 *
 * 4. The Kernel (`gpuMM` function):
 *    - `__global__ void gpuMM(...)`: This CUDA C/C++ keyword signifies that `gpuMM` is a kernel function.
 *      It's executed on the GPU and can be called from the host (CPU) code.
 *    - Each thread executing this kernel first determines the `row` and `col` of the element
 *      of matrix C it is responsible for.
 *    - It then iterates (the `for (int n = 0; n < N; ++n)` loop) to compute the sum of products
 *      `A[row*N+n] * B[n*N+col]`, which is the value for `C[row*N+col]`.
 *    - Matrices are stored linearly in memory (row-major order), so `A[row][n]` is accessed as `A[row*N+n]`.
 *
 * 5. Kernel Launch:
 *    - `gpuMM<<<grid, threadBlock>>>(dA, dB, dC, N);`
 *    - This is the syntax for launching the kernel. `grid` specifies the number of blocks in the grid,
 *      and `threadBlock` specifies the number of threads in each block.
 *
 * 6. Limitations and Considerations in this Program:
 *    - Matrix Size Constraint: The current setup assumes N is a multiple of BLOCK_SIZE.
 *      More robust code would handle arbitrary matrix sizes, possibly by adding checks within
 *      the kernel to ensure threads don't access out-of-bounds memory.
 *    - Input `K`: The user is prompted for `K`, but it's then hardcoded to `K=1`. This should be made consistent.
 *    - Error Checking: Production CUDA code should always check the return status of CUDA API calls
 *      (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launches) for errors.
 *    - Shared Memory Optimization: For more complex matrix multiplications, especially larger ones,
 *      using shared memory (a fast, on-chip memory accessible by threads within the same block)
 *      can significantly improve performance by reducing global memory accesses. This involves loading
 *      tiles of the input matrices into shared memory.
 *
 * This example provides a fundamental understanding of GPU-accelerated matrix multiplication.
 * More advanced implementations would involve optimizations like tiling (using shared memory),
 * handling non-square matrices, and more sophisticated error handling.
 */



/*
 * Matrix Multiplication Program using CUDA
 * ========================================
 *
 * This program performs matrix multiplication (C = A * B) using CUDA.
 * It initializes two matrices A and B on the host (CPU), defines their dimensions,
 * copies them to the device (GPU), performs the multiplication on the GPU using a
 * custom CUDA kernel (`gpuMM`), and then copies the result matrix C back to the host to print.
 *
 * The kernel `gpuMM` assigns each thread to compute one element of the output matrix C.
 * The matrices are square (N x N), and N must be a multiple of BLOCK_SIZE (defined as 2).
 *
 * COMPILATION & EXECUTION (Linux/macOS with NVIDIA CUDA Toolkit):
 * ---------------------------------------------------------------
 * 1. Ensure NVIDIA CUDA Toolkit is installed.
 * 2. Save the code as metrixmul.cu (or metrixmul.cpp if nvcc is configured).
 * 3. Compile:
 *    nvcc metrixmul.cu -o metrixmul
 * 4. Execute:
 *    ./metrixmul
 *    The program will prompt for a value K, where matrix size N = K * BLOCK_SIZE.
 *    (Note: The code currently hardcodes K=1, so N=2).
 *
 * THEORETICAL CONCEPTS:
 *
 * CUDA (Compute Unified Device Architecture):
 * ------------------------------------------
 * - A parallel computing platform and programming model by NVIDIA for GPGPU.
 *
 * Key CUDA Concepts Used:
 * -----------------------
 * - `__global__` function (Kernel): `gpuMM` runs on the GPU.
 * - `blockIdx`, `blockDim`, `threadIdx`: Built-in variables for thread identification
 *   within a 2D grid of 2D thread blocks.
 *   - `blockIdx.y`, `blockIdx.x`: Y and X indices of the current thread block.
 *   - `blockDim.y`, `blockDim.x`: Dimensions of a thread block.
 *   - `threadIdx.y`, `threadIdx.x`: Y and X indices of the current thread within its block.
 * - `cudaMalloc()`: Allocates GPU memory.
 * - `cudaMemcpy()`: Copies data between host and device memory.
 *   - `cudaMemcpyHostToDevice`: Host to Device.
 *   - `cudaMemcpyDeviceToHost`: Device to Host.
 * - Kernel Launch (`gpuMM<<<grid,threadBlock>>>`): Executes the kernel.
 *   - `grid`: Defines the dimensions of the grid of thread blocks (KxK blocks).
 *   - `threadBlock`: Defines the dimensions of each thread block (BLOCK_SIZExBLOCK_SIZE threads).
 * - `dim3`: A CUDA data type for specifying dimensions (e.g., for grids and blocks).
 * - `cudaFree()`: Frees GPU memory (implicitly called at program termination for allocated memory if not explicitly called, but good practice to include).
 *
 * Matrix Indexing:
 * ----------------
 * - Matrices are stored in row-major order in a 1D array. Element (row, col) of an N x N
 *   matrix `M` is accessed as `M[row * N + col]`.
 *
 * Workflow:
 * ---------
 * 1. Get matrix dimension factor K from user (currently overridden to K=1, N=2).
 * 2. Initialize matrices hA, hB on the host.
 * 3. Allocate memory dA, dB, dC on the device.
 * 4. Copy hA to dA, hB to dB.
 * 5. Define grid and thread block dimensions.
 * 6. Launch `gpuMM` kernel on the device.
 * 7. Copy result dC from device to host matrix C.
 * 8. Print input and result matrices.
 * 9. Free host memory (device memory is also implicitly freed, but explicit `cudaFree` is better).
 *
 * Error Handling (Simplified):
 * ----------------------------
 * - Basic example without explicit CUDA error checking. Real applications need robust error handling.
 *
 * Printing Bug Note:
 * ------------------
 * - The printing loops for input matrices `hA` and `hB`, and the result matrix `C`,
 *   use `hA[row*col]`, `hB[row*col]`, and `C[row*col]` for indexing. This is incorrect
 *   for standard row-major or column-major matrix element access. It should be
 *   `hA[row*N + col]`, `hB[row*N + col]`, and `C[row*N + col]` respectively to print correctly.
 *   The GPU computation uses the correct `A[row*N+n]` and `B[n*N+col]` indexing.
 *
 * Sample Output (for K=1, N=2, with printing bug fixed, hA elements = 2, hB elements = 4):
 * -----------------------------------------------------------------------------------------
 * Enter a Value for Size/2 of matrix (user input, but K is set to 1)
 *
 *  Executing Matrix Multiplcation
 *
 *  Matrix size: 2x2
 *
 *  Input Matrix 1
 * 2 2
 * 2 2
 *
 *  Input Matrix 2
 * 4 4
 * 4 4
 *
 *
 *
 *
 *
 *
 *  Resultant matrix
 *
 * 16 16
 * 16 16
 * Finished.
 */
