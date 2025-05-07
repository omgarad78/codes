#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int MAXN = 1e5;
vector<int> adj[MAXN + 5]; // Adjacency list
bool visited[MAXN + 5];    // Mark visited nodes

// DFS function (with OpenMP parallel for — note: this may lead to race conditions!)
void dfs(int node)
{
    visited[node] = true;

#pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++)
    {
        int next_node = adj[node][i];

        // This causes race condition: multiple threads may try to read/write visited[]
        if (!visited[next_node])
        {
            dfs(next_node); // Recursive call (not parallel-safe)
        }
    }
}

int main()
{
    cout << "Enter number of nodes and edges: ";
    int n, m; // number of nodes and edges
    cin >> n >> m;

    cout << "Enter edges (u v):\n";
    for (int i = 1; i <= m; i++)
    {
        int u, v; // edge between u and v
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    int start_node; // start node of DFS
    cout << "Enter starting node for DFS: ";
    cin >> start_node;

    dfs(start_node);

    cout << "Visited nodes: ";
    for (int i = 1; i <= n; i++)
    {
        if (visited[i])
        {
            cout << i << " ";
        }
    }
    cout << endl;

    return 0;
}

// 5 4
// 1 2
// 1 3
// 2 4
// 3 5
// 1

// 🧠 What this means:
// 5 nodes, 4 edges

// Edges:
// 1–2,
// 1–3,
// 2–4,
// 3–5

// Start DFS from node 1
