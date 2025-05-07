#include <iostream>
#include <queue>
#include <vector>
#include <omp.h>
#include <mutex>
using namespace std;

int main()
{
    int num_vertices, num_edges, source;
    cin >> num_vertices >> num_edges >> source;

    vector<vector<int>> adj_list(num_vertices + 1);
    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        cin >> u >> v;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }

    queue<int> q;
    vector<bool> visited(num_vertices + 1, false);
    mutex mtx; // For thread-safe access to queue and visited

    q.push(source);
    visited[source] = true;

    while (!q.empty())
    {
        int curr_vertex;
        {
            curr_vertex = q.front();
            q.pop();
        }

        cout << curr_vertex << " ";

#pragma omp parallel for
        for (int i = 0; i < adj_list[curr_vertex].size(); i++)
        {
            int neighbour = adj_list[curr_vertex][i];

            // Thread-safe check and update
            mtx.lock();
            if (!visited[neighbour])
            {
                visited[neighbour] = true;
                q.push(neighbour);
            }
            mtx.unlock();
        }
    }

    return 0;
}

// Enter number of vertices, edges, and the source vertex:
// 6 7 1
// Enter 7 edges (u v):
// 1 2
// 1 3
// 2 4
// 2 5
// 3 5
// 4 6
// 5 6
