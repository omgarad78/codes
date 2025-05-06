#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Function to generate a random array
void generateArray(vector<int> &arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 10000; // Random numbers between 0-9999
    }
}

// Sequential Bubble Sort
void bubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int> &arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

// Sequential Merge Sort
void mergeSort(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort using OpenMP
void parallelMergeSort(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, left, mid);

#pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

// Function to measure execution time
template <typename Func>
double measureTime(Func func, vector<int> &arr)
{
    auto start = high_resolution_clock::now();
    func(arr);
    auto stop = high_resolution_clock::now();
    return duration<double, milli>(stop - start).count();
}

int main()
{
    srand(time(0));

    int size = 1000; // Change size for testing
    vector<int> arr(size);

    // Bubble Sort Comparison
    generateArray(arr, size);
    vector<int> arrCopy = arr;
    cout << "Sequential Bubble Sort Time: "
         << measureTime(bubbleSort, arrCopy) << " ms\n";

    generateArray(arr, size);
    arrCopy = arr;
    cout << "Parallel Bubble Sort Time: "
         << measureTime(parallelBubbleSort, arrCopy) << " ms\n";

    // Merge Sort Comparison
    generateArray(arr, size);
    arrCopy = arr;
    cout << "Sequential Merge Sort Time: "
         << measureTime([&](vector<int> &a)
                        { mergeSort(a, 0, size - 1); }, arrCopy)
         << " ms\n";

    generateArray(arr, size);
    arrCopy = arr;
    cout << "Parallel Merge Sort Time: "
         << measureTime([&](vector<int> &a)
                        { parallelMergeSort(a, 0, size - 1); }, arrCopy)
         << " ms\n";

    return 0;
}


// 📘 Sorting Algorithms: Sequential vs Parallel (Using OpenMP)
// 🔹 Objective
// The program compares the execution time of sequential and parallel versions of:

// Bubble Sort

// Merge Sort

// It uses OpenMP for parallelism and measures performance using the chrono library.

// 🔹 Key Concepts
// ✅ Bubble Sort
// A simple sorting algorithm that repeatedly compares adjacent elements and swaps them if they are in the wrong order.

// Time Complexity: O(n²)

// Best for small datasets due to poor scalability.

// 🔸 Parallel Bubble Sort
// Each pass of the inner loop is parallelized using OpenMP.

// However, not very efficient in parallel form due to data dependency — swapping elements affects adjacent comparisons.

// ✅ Merge Sort
// A divide-and-conquer algorithm that splits the array, recursively sorts both halves, and then merges them.

// Time Complexity: O(n log n)

// Stable and suitable for large datasets.

// 🔸 Parallel Merge Sort
// Uses OpenMP sections to sort left and right halves in parallel.

// Greatly benefits from parallel execution as each subarray can be processed independently.

// 🔹 Execution Time Measurement
// The chrono library is used to capture start and end times.

// Time difference is calculated in milliseconds.

// 🔹 Parallelism with OpenMP
// #pragma omp parallel for – splits loop iterations across threads.

// #pragma omp parallel sections – allows multiple independent sections to run concurrently.

// Useful for multicore processors to reduce computation time.

// 🧪 Performance Testing
// Random arrays are generated using rand().

// Sorting is performed multiple times on identical copies for fair comparison.

// Performance is printed for:

// Sequential Bubble Sort

// Parallel Bubble Sort

// Sequential Merge Sort

// Parallel Merge Sort

// -----------------------------------------------------------------------------------------


// 🧮 1. Bubble Sort (Sequential)
// 🔸 Concept:
// Repeatedly compares adjacent elements and swaps them if they are in the wrong order.

// After each pass, the largest unsorted element moves to its correct position — like bubbles rising in water.

// 🧠 Example:
// Initial array: [5, 3, 8, 4]

// Step-by-step:
// Pass 1:

// Compare 5 & 3 → swap → [3, 5, 8, 4]

// Compare 5 & 8 → no swap

// Compare 8 & 4 → swap → [3, 5, 4, 8]

// Pass 2:

// Compare 3 & 5 → no swap

// Compare 5 & 4 → swap → [3, 4, 5, 8]

// Pass 3:

// Compare 3 & 4 → no swap

// ✅ Sorted output: [3, 4, 5, 8]

// 🧮 2. Bubble Sort (Parallel using OpenMP)
// 🔸 Concept:
// Tries to run the inner loop (comparisons) in parallel using threads.

// But: Each swap may affect future comparisons, so true parallel efficiency is limited.

// ⚠️ Example:
// Consider [5, 3, 8, 4], and suppose 2 threads run:

// Thread 1 compares (5, 3)

// Thread 2 compares (8, 4)

// But in the next pass, order depends on earlier swaps, so results may become unpredictable if not synchronized properly.

// 📌 So, it works, but performance gains are minimal and may not scale well.

// 🧮 3. Merge Sort (Sequential)
// 🔸 Concept:
// Recursively splits the array in half.

// Sorts each half.

// Then merges the two sorted halves.

// 🧠 Example:
// Initial array: [6, 3, 8, 2]

// Step-by-step:
// Split into [6, 3] and [8, 2]

// Sort [6, 3] → [3, 6]

// Sort [8, 2] → [2, 8]

// Merge [3, 6] and [2, 8] → [2, 3, 6, 8]

// ✅ Sorted output: [2, 3, 6, 8]

// 🧮 4. Merge Sort (Parallel using OpenMP)
// 🔸 Concept:
// Same divide-and-conquer logic, but:

// Left and right halves are sorted concurrently in separate threads.

// 🧠 Example (continued):
// Initial: [6, 3, 8, 2]

// Thread 1 sorts [6, 3] → [3, 6]

// Thread 2 sorts [8, 2] → [2, 8]

// Main thread merges them → [2, 3, 6, 8]

// ✅ Much faster for large arrays, because recursion tree grows and can be parallelized deeply.
