#ifndef GRAPHAL_HPP
#define GRAPHAL_HPP

#include <vector>
#include <queue>
#include <stack>
#include <iostream>
using namespace std;

class GraphAL {
private:
    int n;                      // 顶点数
    vector<vector<int>> adj;    // 邻接表

public:
    explicit GraphAL(int verNum) : n(verNum), adj(verNum) {}

    /* 加无向边 */
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    /* DFS求连通分量 */
    void DFS_Components() const {
        vector<bool> vis(n, false);
        int compId = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                cout << "Component " << ++compId << " : ";
                dfs(i, vis);
                cout << '\n';
            }
        }
    }

    /* BFS求连通分量 */
    void BFS_Components() const {
        vector<bool> vis(n, false);
        int compId = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                cout << "Component " << ++compId << " : ";
                bfs(i, vis);
                cout << '\n';
            }
        }
    }

private:
    /* 单趟 DFS */
    void dfs(int u, vector<bool>& vis) const {
        vis[u] = true;
        cout << u << ' ';
        for (int v : adj[u])
            if (!vis[v]) dfs(v, vis);
    }

    /* 单趟 BFS */
    void bfs(int s, vector<bool>& vis) const {
        queue<int> q;
        vis[s] = true;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cout << u << ' ';
            for (int v : adj[u]) {
                if (!vis[v]) {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
};

#endif // GRAPHAL_HPP
