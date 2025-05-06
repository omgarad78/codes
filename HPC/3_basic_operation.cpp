#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

// Display Array
void displayArray(int nums[], int length)
{
    cout << "Nums: [";
    for (int i = 0; i < length; i++)
    {
        cout << nums[i];
        if (i != length - 1)
            cout << ", ";
    }
    cout << "]" << endl;
}

// Parallel Min Operation
void minOperation(int nums[], int length)
{
    int minValue = nums[0];
#pragma omp parallel for reduction(min : minValue)
    for (int i = 0; i < length; i++)
    {
        minValue = (nums[i] < minValue) ? nums[i] : minValue;
    }
    cout << "Min value: " << minValue << endl;
}

// Parallel Max Operation
void maxOperation(int nums[], int length)
{
    int maxValue = nums[0];
#pragma omp parallel for reduction(max : maxValue)
    for (int i = 0; i < length; i++)
    {
        maxValue = (nums[i] > maxValue) ? nums[i] : maxValue;
    }
    cout << "Max value: " << maxValue << endl;
}

// Parallel Sum Operation
void sumOperation(int nums[], int length)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++)
    {
        sum += nums[i];
    }
    cout << "Sum: " << sum << endl;
}

// Parallel Average Operation
void avgOperation(int nums[], int length)
{
    float sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++)
    {
        sum += nums[i];
    }
    cout << "Average: " << (sum / length) << endl;
}

// Main Function
int main()
{
    int nums[] = {4, 6, 3, 2, 6, 7, 9, 2, 1, 6, 5};
    int length = sizeof(nums) / sizeof(int);

    auto start = high_resolution_clock::now();

    displayArray(nums, length);
    minOperation(nums, length);
    maxOperation(nums, length);
    sumOperation(nums, length);
    avgOperation(nums, length);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\nExecution time: " << duration.count() << " microseconds" << endl;

    return 0;
}


/*
Theoretical Concepts:

1. Parallel Reduction Fundamentals:
   - Reduction is a fundamental parallel programming pattern
   - Combines multiple values into a single result using an associative operator
   - Follows the principle of divide-and-conquer
   - Time complexity: O(log n) with n processors, O(n/p + log p) with p processors

2. OpenMP Architecture:
   - Fork-Join Model:
     * Master thread creates team of worker threads (fork)
     * Threads execute parallel region
     * Threads synchronize and merge results (join)
   - Thread Management:
     * Dynamic thread creation and destruction
     * Thread pool for better performance
     * Automatic load balancing

3. Memory Model:
   - Shared Memory Architecture:
     * All threads share the same address space
     * Direct access to shared variables
     * Need for synchronization mechanisms
   - Memory Consistency:
     * Flush operations for memory synchronization
     * Atomic operations for thread safety
     * Memory barriers for ordering

4. Parallel Reduction Algorithm:
   - Step 1: Data Partitioning
     * Divide input data among threads
     * Each thread processes its portion independently
   - Step 2: Local Computation
     * Each thread performs reduction on its portion
     * Creates partial results
   - Step 3: Global Reduction
     * Combine partial results using reduction operator
     * Tree-based combination for efficiency
     * Final result in shared variable

5. Performance Considerations:
   - Scalability:
     * Strong scaling: fixed problem size, increasing processors
     * Weak scaling: problem size grows with processors
   - Overhead Factors:
     * Thread creation and management
     * Synchronization costs
     * Memory access patterns
   - Optimization Techniques:
     * Cache utilization
     * False sharing avoidance
     * Load balancing
     * Memory alignment

6. Mathematical Properties:
   - Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
   - Commutativity: a ⊕ b = b ⊕ a
   - Identity Element: a ⊕ e = a
   - Required for parallel reduction to work correctly

7. Common Reduction Operations:
   - Arithmetic: sum, product, average
   - Logical: AND, OR, XOR
   - Statistical: min, max, variance
   - Custom: user-defined reduction operators

8. Error Handling:
   - Race Conditions:
     * Multiple threads accessing shared data
     * Need for proper synchronization
   - Deadlocks:
     * Circular dependencies in synchronization
     * Proper lock ordering
   - Load Imbalance:
     * Uneven work distribution
     * Dynamic scheduling solutions
*/
