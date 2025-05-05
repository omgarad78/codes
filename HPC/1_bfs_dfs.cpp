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
