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

// �߽ṹ��
struct Edge {
    int u, v, weight; // ��ʼ�㡢��ֹ�㡢Ȩ��
    // ���أ���Ȩ�Ƚ�
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// ���б�->�ڽӱ�
vector<vector<pair<int, int>>> edgesToAdjList(const vector<Edge>& edges, int V) {
    vector<vector<pair<int, int>>> adjList(V);
    for (const auto& edge : edges) {
        adjList[edge.u].push_back({ edge.v, edge.weight });
        adjList[edge.v].push_back({ edge.u, edge.weight });
    }
    return adjList;
}

// ���鼯
struct DisjointSet {
    vector<int> parent; // ���ڵ�
    vector<int> rank; // ��
    // ��ʼ��Ϊ���ڵ����Լ�����Ϊ0
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i; 
        }
    }
     // �ݹ���Ҹ��ڵ�
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    // �ϲ�
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        // ͬ������Ҫ�ϲ�
        if (rootX == rootY) 
            return;  
        // ����С�������ӵ��ȴ������
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        }
        else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        }
        // ������ȣ���ѡһ����Ϊ���ڵ㣬�������ȼ�1
        else { 
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
};

// Kruskal����Ȩ�ش�С������ÿ���ߣ��������ñ߲����γɻ����������MST
vector<Edge> kruskalMST(const vector<Edge>& edges, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    vector<Edge> sortedEdges = edges;
    sort(sortedEdges.begin(), sortedEdges.end()); // ����Ȩ��������
    DisjointSet ds(V); // ��⻷·
    // ��Ȩ��������ÿ����
    for (const auto& edge : sortedEdges) {
        int rootU = ds.find(edge.u);
        int rootV = ds.find(edge.v);
        // ��ӱ߲����γɻ�������ͬһ���£������߼���MST���ϲ���
        if (rootU != rootV) { 
            result.push_back(edge);
            ds.unite(rootU, rootV);
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// Prim������ʼ���㿪ʼ��ÿ��ѡ��Ȩ����С������MST���MST����ı�
vector<Edge> primMST(const vector<vector<pair<int, int>>>& adjList, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    vector<bool> visited(V, false); // �Ƿ��Ѽ���MST
    vector<int> key(V, numeric_limits<int>::max()); // �����㵽MST����С��Ȩ
    vector<int> parent(V, -1); // MST�и�����ĸ��ڵ�
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // ���ȶ��У�ѡ��Ȩ����С�ı�
    // ��ʼ����������ȶ���
    key[0] = 0;  
    pq.push({ 0, 0 });
    while (!pq.empty()) {
        int u = pq.top().second; // ��ȡkeyֵ��С�Ķ���
        pq.pop();
        if (visited[u]) // �����ѷ���
            continue; 
        visited[u] = true; // ����ѷ���
        // ������ʼ�ڵ㣬��ӱߵ����
        if (parent[u] != -1) {
            result.push_back({ parent[u], u, key[u] });
        }
        // ����keyֵ
        for (const auto& neighbor : adjList[u]) {
            int v = neighbor.first; 
            int weight = neighbor.second;  
            // δ���ʶ���Ȩ�ر�С
            if (!visited[v] && weight < key[v]) {
                parent[v] = u; // ���¸��ڵ�
                key[v] = weight; // ����keyֵ
                pq.push({ key[v], v }); // �������ȶ���
            }
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    executionTime = chrono::duration<double, milli>(endTime - startTime).count();
    return result;
}

// �����������
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

// �����ͨ�ԣ�DFS�������ж����Ƿ񱻷���
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

// �Ƴ�һ���ߣ�˫��
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

// ����ɾ������Ȩ�شӴ�С����ÿ���ߣ����ɾ���ñ߲��ᵼ��ͼ����ͨ�������ͼ��ɾ��
vector<Edge> reverseDeleteMST(const vector<Edge>& edges, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    // ��ʼͼ�������б�
    vector<vector<pair<int, int>>> graph(V);
    for (const auto& edge : edges) {
        graph[edge.u].push_back({ edge.v, edge.weight });
        graph[edge.v].push_back({ edge.u, edge.weight });
    }
    // ����Ȩ��������
    vector<Edge> sortedEdges = edges;
    sort(sortedEdges.begin(), sortedEdges.end(), [](const Edge& a, const Edge& b) {
        return a.weight > b.weight;
        });
    vector<Edge> result = edges; 
    // ����Ȩ������ÿ���ߣ������Ƴ������Ƿ���ͨ
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

// Boruvka��ÿһ��ͬʱ������������Ϊÿ����Ѱ����СȨ�ص��ⲿ���ӱ�
vector<Edge> boruvkaMST(const vector<vector<pair<int, int>>>& adjList, int V, double& executionTime) {
    auto startTime = chrono::high_resolution_clock::now();
    vector<Edge> result; 
    DisjointSet ds(V); // �ϲ���
    int numComponents = V; // ����
    while (numComponents > 1) {
        // ÿ��������С�ߣ�˫��ֻ����һ����
        vector<Edge> cheapest(V, { -1, -1, numeric_limits<int>::max() });
        for (int u = 0; u < V; u++) {
            for (const auto& neighbor : adjList[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;
                if (u < v) {
                    int set1 = ds.find(u);
                    int set2 = ds.find(v);
                    // ��������Ӳ�ͬ��������鲢���¶�����С��
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
        // ��������ҵ�����С�ߵ�MST
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

// SVGͼ��
void generateSVG(const vector<Edge>& edges, int V, const string& filename) {
    const int nodeRadius = 20;  // �ڵ�Բ�ΰ뾶
    const int width = 500;      // ͼ����
    const int height = 400;     // ͼ��߶�
    const int centerX = width / 2;  // ���ĵ�X����
    const int centerY = height / 2; // ���ĵ�Y����
    const int radius = min(width, height) / 3;  // �ڵ�ֲ��뾶
    // �����ļ���д��ͷ������
    ofstream svgFile(filename + ".svg");
    svgFile << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svgFile << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    // ���β��ֽڵ㣬���������
    vector<pair<int, int>> nodePositions(V);
    for (int i = 0; i < V; ++i) {
        double angle = 2.0 * M_PI * i / V;
        nodePositions[i].first = centerX + int(radius * cos(angle));
        nodePositions[i].second = centerY + int(radius * sin(angle));
    }
    // ���Ʊ�
    for (const auto& edge : edges) {
        int x1 = nodePositions[edge.u].first;
        int y1 = nodePositions[edge.u].second;
        int x2 = nodePositions[edge.v].first;
        int y2 = nodePositions[edge.v].second;
        // ��ɫ����
        string strokeColor = "black";
        int strokeWidth = 2;
        svgFile << "<line x1=\"" << x1 << "\" y1=\"" << y1
            << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"" << strokeColor << "\" stroke-width=\"" << strokeWidth << "\" />\n";
        // ���е����Ȩ��
        int textX = (x1 + x2) / 2;
        int textY = (y1 + y2) / 2 - 10;
        svgFile << "<text x=\"" << textX << "\" y=\"" << textY
            << "\" font-family=\"Arial\" font-size=\"12\" fill=\"blue\">"
            << edge.weight << "</text>\n";
    }
    // ���ƽڵ�
    for (int i = 0; i < V; ++i) {
        int x = nodePositions[i].first;
        int y = nodePositions[i].second;
        // Բ�νڵ�
        svgFile << "<circle cx=\"" << x << "\" cy=\"" << y
            << "\" r=\"" << nodeRadius << "\" fill=\"lightblue\" stroke=\"black\" stroke-width=\"1\" />\n";
        // �ڵ���
        svgFile << "<text x=\"" << x << "\" y=\"" << y + 5
            << "\" font-family=\"Arial\" font-size=\"16\" text-anchor=\"middle\" fill=\"black\">"
            << i + 1 << "</text>\n";
    }
    // д��β�����ر��ļ�
    svgFile << "</svg>\n";
    svgFile.close();
}

// ������С����������Ȩ��
int calculateTotalWeight(const vector<Edge>& mstEdges) {
    int totalWeight = 0;
    for (const auto& edge : mstEdges) {
        totalWeight += edge.weight;
    }
    return totalWeight;
}

int main() {
    int V = 6; // ������
    // �߼���
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
    auto adjList = edgesToAdjList(edges, V); // �ڽӱ�
    generateSVG(edges, V, "original_graph"); // ԭʼͼ
    // Kruskal
    double kruskalTime;
    vector<Edge> kruskalResult = kruskalMST(edges, V, kruskalTime);
    generateSVG(kruskalResult, V, "kruskal_mst");
    int kruskalTotalWeight = calculateTotalWeight(kruskalResult);
    cout << "Kruskal MST��Ȩ��: " << kruskalTotalWeight << endl;
    cout << "Kruskal�㷨ִ��ʱ��: " << kruskalTime << " ����" << endl;
    // Prim
    double primTime;
    vector<Edge> primResult = primMST(adjList, V, primTime);
    generateSVG(primResult, V, "prim_mst");
    int primTotalWeight = calculateTotalWeight(primResult);
    cout << "Prim MST��Ȩ��: " << primTotalWeight << endl;
    cout << "Prim�㷨ִ��ʱ��: " << primTime << " ����" << endl;
    // ����ɾ��
    double reverseDeleteTime;
    vector<Edge> reverseDeleteResult = reverseDeleteMST(edges, V, reverseDeleteTime);
    generateSVG(reverseDeleteResult, V, "reverse_delete_mst");
    int reverseDeleteTotalWeight = calculateTotalWeight(reverseDeleteResult);
    cout << "����ɾ��MST��Ȩ��: " << reverseDeleteTotalWeight << endl;
    cout << "����ɾ���㷨ִ��ʱ��: " << reverseDeleteTime << " ����" << endl;
    // Boruvka
    double boruvkaTime;
    vector<Edge> boruvkaResult = boruvkaMST(adjList, V, boruvkaTime);
    generateSVG(boruvkaResult, V, "boruvka_mst");
    int boruvkaTotalWeight = calculateTotalWeight(boruvkaResult);
    cout << "Boruvka MST��Ȩ��: " << boruvkaTotalWeight << endl;
    cout << "Boruvka�㷨ִ��ʱ��: " << boruvkaTime << " ����" << endl;
    return 0;
}