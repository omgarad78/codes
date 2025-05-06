#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <atomic>

using namespace std;

class Graph
{
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int v, int w)
    {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }

    void parallelBFS(int start)
    {
        vector<atomic<bool>> visited(V);
        for (int i = 0; i < V; ++i)
            visited[i] = false;

        vector<int> currentLevel = {start};
        visited[start] = true;

        cout << "Parallel BFS Traversal: ";

        while (!currentLevel.empty())
        {
            vector<int> nextLevel;

#pragma omp parallel
            {
                vector<int> localNext;

#pragma omp for nowait
                for (int i = 0; i < currentLevel.size(); i++)
                {
                    int node = currentLevel[i];

#pragma omp critical
                    cout << node << " ";

                    for (int neighbor : adj[node])
                    {
                        bool expected = false;
                        if (visited[neighbor].compare_exchange_strong(expected, true))
                        {
                            localNext.push_back(neighbor);
                        }
                    }
                }

#pragma omp critical
                nextLevel.insert(nextLevel.end(), localNext.begin(), localNext.end());
            }

            currentLevel = nextLevel;
        }

        cout << endl;
    }

    void sequentialDFSUtil(int node, vector<bool> &visited)
    {
        visited[node] = true;
        cout << node << " ";
        for (int neighbor : adj[node])
        {
            if (!visited[neighbor])
                sequentialDFSUtil(neighbor, visited);
        }
    }

    void sequentialDFS(int start)
    {
        vector<bool> visited(V, false);
        cout << "Sequential DFS Traversal: ";
        sequentialDFSUtil(start, visited);
        cout << endl;
    }
};

int main()
{
    Graph g(6);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    cout << "Starting Parallel BFS from node 0:" << endl;
    g.parallelBFS(0);

    cout << "Starting Sequential DFS from node 0:" << endl;
    g.sequentialDFS(0);

    return 0;
}



// 📘 Graph Traversal Using Parallel BFS and Sequential DFS
// 🔹 Graph Basics
// A graph consists of nodes (vertices) and edges connecting them.

// Can be directed or undirected, and weighted or unweighted.

// Two common traversal algorithms are Breadth-First Search (BFS) and Depth-First Search (DFS).

// 🔹 Breadth-First Search (BFS)
// Explores the graph level by level starting from a given node.

// Uses a queue-like structure to track the "current level" of nodes.

// Visits all neighbors of a node before moving deeper.

// ✅ Parallel BFS
// BFS can be parallelized to speed up the exploration.

// Each level of nodes is processed simultaneously by multiple threads.

// Uses OpenMP for parallel execution and atomic operations to prevent race conditions.

// Efficient for large graphs where each level has many nodes.

// 🔹 Depth-First Search (DFS)
// Explores the graph by going as deep as possible along each branch before backtracking.

// Typically implemented using recursion or a stack.

// Best for tasks like:

// Checking connectivity

// Detecting cycles

// Topological sorting

// ✅ Sequential DFS
// Runs in a single thread.

// Simple and memory-efficient.

// Traverses nodes in a depth-first manner.

// 🔹 Parallel vs Sequential
// Feature	Parallel BFS	Sequential DFS
// Traversal Type	Level-order (BFS)	Depth-order (DFS)
// Speed	Faster on large graphs	Slower, but simpler
// Threads Used	Multiple (OpenMP)	Single-threaded
// Use Case	Large real-time applications	Graph analysis, tree problems

// 🧠 Concepts Used
// Adjacency List: Efficient structure for storing sparse graphs.

// Atomic Variables: Prevent multiple threads from modifying the same variable simultaneously.

// OpenMP: Parallel programming library for C++ to utilize multi-core processors.

// Critical Sections: Ensures certain code blocks are accessed by only one thread at a time.
