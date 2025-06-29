#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <queue>
#include <limits>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath> 
#define _USE_MATH_DEFINES
#include <math.h> 
#include <chrono> 
using namespace std;

// 边结构体
struct Edge {
    int u, v, weight; // 起始点、终止点、权重
    // 重载：边权比较
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// 边列表->邻接表
vector<vector<pair<int, int>>> edgesToAdjList(const vector<Edge>& edges, int V) {
    vector<vector<pair<int, int>>> adjList(V);
    for (const auto& edge : edges) {
        adjList[edge.u].push_back({ edge.v, edge.weight });
        adjList[edge.v].push_back({ edge.u, edge.weight });
    }
    return adjList;
}

// 并查集
struct DisjointSet {
    vector<int> parent; // 父节点
    vector<int> rank; // 秩
    // 初始化为父节点是自己，秩为0
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i; 
        }
    }
     // 递归查找根节点
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    // 合并
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        // 同根不需要合并
        if (rootX == rootY) 
            return;  
        // 将秩小的树连接到秩大的树下
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        }
        else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        }
        // 两秩相等，任选一个作为父节点，并将其秩加1
        else { 
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
};

// Kruskal：按权重从小到大考虑每条边，如果加入该边不会形成环，则将其加入MST
vector<Edge> kruskalMST(const vector<Edge>& edges, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    vector<Edge> sortedEdges = edges;
    sort(sortedEdges.begin(), sortedEdges.end()); // 按边权升序排序
    DisjointSet ds(V); // 检测环路
    // 按权重升序考虑每条边
    for (const auto& edge : sortedEdges) {
        int rootU = ds.find(edge.u);
        int rootV = ds.find(edge.v);
        // 添加边不会形成环（不在同一树下），将边加入MST，合并树
        if (rootU != rootV) { 
            result.push_back(edge);
            ds.unite(rootU, rootV);
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// Prim：从起始顶点开始，每次选择权重最小且连接MST与非MST顶点的边
vector<Edge> primMST(const vector<vector<pair<int, int>>>& adjList, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    vector<bool> visited(V, false); // 是否已加入MST
    vector<int> key(V, numeric_limits<int>::max()); // 各顶点到MST的最小边权
    vector<int> parent(V, -1); // MST中各顶点的父节点
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // 优先队列，选择权重最小的边
    // 起始顶点加入优先队列
    key[0] = 0;  
    pq.push({ 0, 0 });
    while (!pq.empty()) {
        int u = pq.top().second; // 获取key值最小的顶点
        pq.pop();
        if (visited[u]) // 跳过已访问
            continue; 
        visited[u] = true; // 标记已访问
        // 不是起始节点，添加边到结果
        if (parent[u] != -1) {
            result.push_back({ parent[u], u, key[u] });
        }
        // 更新key值
        for (const auto& neighbor : adjList[u]) {
            int v = neighbor.first; 
            int weight = neighbor.second;  
            // 未访问顶点权重变小
            if (!visited[v] && weight < key[v]) {
                parent[v] = u; // 更新父节点
                key[v] = weight; // 更新key值
                pq.push({ key[v], v }); // 加入优先队列
            }
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// 深度优先搜索
bool dfs(int u, vector<bool>& visited, const vector<vector<pair<int, int>>>& graph) {
    visited[u] = true;
    for (const auto& neighbor : graph[u]) {
        int v = neighbor.first;
        if (!visited[v]) {
            dfs(v, visited, graph);
        }
    }
    return true;
}

// 检查连通性：DFS后检查所有顶点是否被访问
bool isConnected(const vector<vector<pair<int, int>>>& graph, int V) {
    vector<bool> visited(V, false);
    dfs(0, visited, graph);
    for (int i = 0; i < V; ++i) {
        if (!visited[i]) {
            return false; 
        }
    }
    return true;
}

// 移除一条边（双向）
vector<vector<pair<int, int>>> removeEdge(const vector<vector<pair<int, int>>>& graph, const Edge& edge) {
    vector<vector<pair<int, int>>> newGraph = graph;
    for (auto it = newGraph[edge.u].begin(); it != newGraph[edge.u].end(); ++it) {
        if (it->first == edge.v && it->second == edge.weight) {
            newGraph[edge.u].erase(it);
            break;
        }
    }
    for (auto it = newGraph[edge.v].begin(); it != newGraph[edge.v].end(); ++it) {
        if (it->first == edge.u && it->second == edge.weight) {
            newGraph[edge.v].erase(it);
            break;
        }
    }
    return newGraph;
}

// 反向删除：按权重从大到小考虑每条边，如果删除该边不会导致图不连通，则将其从图中删除
vector<Edge> reverseDeleteMST(const vector<Edge>& edges, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    // 初始图包含所有边
    vector<vector<pair<int, int>>> graph(V);
    for (const auto& edge : edges) {
        graph[edge.u].push_back({ edge.v, edge.weight });
        graph[edge.v].push_back({ edge.u, edge.weight });
    }
    // 按边权降序排序
    vector<Edge> sortedEdges = edges;
    sort(sortedEdges.begin(), sortedEdges.end(), [](const Edge& a, const Edge& b) {
        return a.weight > b.weight;
        });
    vector<Edge> result = edges; 
    // 按边权降序考虑每条边：尝试移除后检查是否连通
    for (const auto& edge : sortedEdges) {
        auto tempGraph = removeEdge(graph, edge);
        if (isConnected(tempGraph, V)) {
            result.erase(remove_if(result.begin(), result.end(),
                [&edge](const Edge& e) {
                    return (e.u == edge.u && e.v == edge.v && e.weight == edge.weight) ||
                        (e.u == edge.v && e.v == edge.u && e.weight == edge.weight);
                }), result.end());
            graph = tempGraph;
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// Boruvka：每一步同时考虑所有树，为每个树寻找最小权重的外部连接边
vector<Edge> boruvkaMST(const vector<vector<pair<int, int>>>& adjList, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    DisjointSet ds(V); // 合并树
    int numComponents = V; // 树数
    while (numComponents > 1) {
        // 每个树的最小边（双向只计算一个）
        vector<Edge> cheapest(V, { -1, -1, numeric_limits<int>::max() });
        for (int u = 0; u < V; u++) {
            for (const auto& neighbor : adjList[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;
                if (u < v) {
                    int set1 = ds.find(u);
                    int set2 = ds.find(v);
                    // 如果边连接不同的树，检查并更新二者最小边
                    if (set1 != set2) {
                        if (weight < cheapest[set1].weight) {
                            cheapest[set1] = { u, v, weight };
                        }
                        if (weight < cheapest[set2].weight) {
                            cheapest[set2] = { u, v, weight };
                        }
                    }
                }
            }
        }
        // 添加所有找到的最小边到MST
        for (int i = 0; i < V; i++) {
            Edge edge = cheapest[i];
            if (edge.weight != numeric_limits<int>::max()) {
                int set1 = ds.find(edge.u);
                int set2 = ds.find(edge.v);
                if (set1 != set2) {
                    result.push_back(edge);
                    ds.unite(set1, set2);
                    numComponents--;
                }
            }
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// SVG图像
void generateSVG(const vector<Edge>& edges, int V, const string& filename) {
    const int nodeRadius = 20;  // 节点圆形半径
    const int width = 500;      // 图像宽度
    const int height = 400;     // 图像高度
    const int centerX = width / 2;  // 中心点X坐标
    const int centerY = height / 2; // 中心点Y坐标
    const int radius = min(width, height) / 3;  // 节点分布半径
    // 创建文件并写入头部声明
    ofstream svgFile(filename + ".svg");
    svgFile << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svgFile << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    // 环形布局节点，极坐标计算
    vector<pair<int, int>> nodePositions(V);
    for (int i = 0; i < V; ++i) {
        double angle = 2.0 * M_PI * i / V;
        nodePositions[i].first = centerX + int(radius * cos(angle));
        nodePositions[i].second = centerY + int(radius * sin(angle));
    }
    // 绘制边
    for (const auto& edge : edges) {
        int x1 = nodePositions[edge.u].first;
        int y1 = nodePositions[edge.u].second;
        int x2 = nodePositions[edge.v].first;
        int y2 = nodePositions[edge.v].second;
        // 黑色边线
        string strokeColor = "black";
        int strokeWidth = 2;
        svgFile << "<line x1=\"" << x1 << "\" y1=\"" << y1
            << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"" << strokeColor << "\" stroke-width=\"" << strokeWidth << "\" />\n";
        // 边中点绘制权重
        int textX = (x1 + x2) / 2;
        int textY = (y1 + y2) / 2 - 10;
        svgFile << "<text x=\"" << textX << "\" y=\"" << textY
            << "\" font-family=\"Arial\" font-size=\"12\" fill=\"blue\">"
            << edge.weight << "</text>\n";
    }
    // 绘制节点
    for (int i = 0; i < V; ++i) {
        int x = nodePositions[i].first;
        int y = nodePositions[i].second;
        // 圆形节点
        svgFile << "<circle cx=\"" << x << "\" cy=\"" << y
            << "\" r=\"" << nodeRadius << "\" fill=\"lightblue\" stroke=\"black\" stroke-width=\"1\" />\n";
        // 节点编号
        svgFile << "<text x=\"" << x << "\" y=\"" << y + 5
            << "\" font-family=\"Arial\" font-size=\"16\" text-anchor=\"middle\" fill=\"black\">"
            << i + 1 << "</text>\n";
    }
    // 写入尾部并关闭文件
    svgFile << "</svg>\n";
    svgFile.close();
}

// 计算最小生成树的总权重
int calculateTotalWeight(const vector<Edge>& mstEdges) {
    int totalWeight = 0;
    for (const auto& edge : mstEdges) {
        totalWeight += edge.weight;
    }
    return totalWeight;
}

int main() {
    int V = 6; // 顶点数
    // 边集合
    vector<Edge> edges = {
        {0, 1, 4}, {0, 2, 3}, {0, 3, 1}, {1, 3, 5}, {1, 4, 7},
        {2, 3, 2}, {2, 5, 3}, {3, 4, 5}, {3, 5, 4}, {4, 5, 6}
    };
    //vector<Edge> edges = {
    //    {0, 1, 2}, {0, 2, 4}, {0, 3, 8}, {0, 4, 3}, {0, 5, 7},
    //    {1, 2, 3}, {1, 3, 5}, {1, 4, 6}, {1, 5, 1},
    //    {2, 3, 2}, {2, 4, 4}, {2, 5, 5},
    //    {3, 4, 3}, {3, 5, 6},
    //    {4, 5, 2}
    //};
    auto adjList = edgesToAdjList(edges, V); // 邻接表
    generateSVG(edges, V, "original_graph"); // 原始图
    // Kruskal
    double kruskalTime;
    vector<Edge> kruskalResult = kruskalMST(edges, V, kruskalTime);
    generateSVG(kruskalResult, V, "kruskal_mst");
    int kruskalTotalWeight = calculateTotalWeight(kruskalResult);
    cout << "Kruskal MST总权重: " << kruskalTotalWeight << endl;
    cout << "Kruskal算法执行时间: " << kruskalTime << " 毫秒" << endl;
    // Prim
    double primTime;
    vector<Edge> primResult = primMST(adjList, V, primTime);
    generateSVG(primResult, V, "prim_mst");
    int primTotalWeight = calculateTotalWeight(primResult);
    cout << "Prim MST总权重: " << primTotalWeight << endl;
    cout << "Prim算法执行时间: " << primTime << " 毫秒" << endl;
    // 反向删除
    double reverseDeleteTime;
    vector<Edge> reverseDeleteResult = reverseDeleteMST(edges, V, reverseDeleteTime);
    generateSVG(reverseDeleteResult, V, "reverse_delete_mst");
    int reverseDeleteTotalWeight = calculateTotalWeight(reverseDeleteResult);
    cout << "反向删除MST总权重: " << reverseDeleteTotalWeight << endl;
    cout << "反向删除算法执行时间: " << reverseDeleteTime << " 毫秒" << endl;
    // Boruvka
    double boruvkaTime;
    vector<Edge> boruvkaResult = boruvkaMST(adjList, V, boruvkaTime);
    generateSVG(boruvkaResult, V, "boruvka_mst");
    int boruvkaTotalWeight = calculateTotalWeight(boruvkaResult);
    cout << "Boruvka MST总权重: " << boruvkaTotalWeight << endl;
    cout << "Boruvka算法执行时间: " << boruvkaTime << " 毫秒" << endl;
    return 0;
}