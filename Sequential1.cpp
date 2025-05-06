#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <fstream>

void logPerformance(const std::string& implType, const std::string& dataset, double execTime) {
    std::ofstream outputFile("performance.csv", std::ios::app);
    if (outputFile.is_open()) {
        outputFile << implType << "," << dataset << "," << execTime << "\n";
    }
}
struct Edge {
    int source;
    int target;
    int weight;
    
    Edge(int s, int t, int w) : source(s), target(t), weight(w) {}
    
    bool operator==(const Edge& other) const {
        return source == other.source && target == other.target;
    }
};

// Custom hash function for Edge
struct EdgeHash {
    size_t operator()(const Edge& edge) const {
        return std::hash<int>()(edge.source) ^ std::hash<int>()(edge.target);
    }
};

class Graph {
private:
    int numVertices;
    std::vector<std::vector<std::pair<int, int>>> adjacencyList; // (target, weight)
    std::vector<int> distances;       // Distances from source
    std::vector<int> predecessors;    // Predecessors in shortest path tree
    
public:
    
    Graph(int n) : numVertices(n) {
        adjacencyList.resize(n);
        distances.resize(n, std::numeric_limits<int>::max());
        predecessors.resize(n, -1);
    }
    
    // Add an edge to the graph
    void addEdge(int source, int target, int weight) {
        // Check if edge already exists and update it if it does
        bool edgeExists = false;
        for (auto& edge : adjacencyList[source]) {
            if (edge.first == target) {
                edge.second = weight;
                edgeExists = true;
                break;
            }
        }
        
        if (!edgeExists) {
            adjacencyList[source].push_back(std::make_pair(target, weight));
        }
    }
    
    // Remove an edge from the graph
    bool removeEdge(int source, int target) {
        auto& edges = adjacencyList[source];
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            if (it->first == target) {
                edges.erase(it);
                return true;
            }
        }
        return false;
    }
    
    // Update weight of an existing edge
    bool updateEdgeWeight(int source, int target, int newWeight) {
        for (auto& edge : adjacencyList[source]) {
            if (edge.first == target) {
                edge.second = newWeight;
                return true;
            }
        }
        return false;
    }
    
    // Add a vertex to the graph
    void addVertex() {
        numVertices++;
        adjacencyList.push_back(std::vector<std::pair<int, int>>());
        distances.push_back(std::numeric_limits<int>::max());
        predecessors.push_back(-1);
    }
    
    // Remove a vertex from the graph
    void removeVertex(int vertex) {
        if (vertex < 0 || vertex >= numVertices) return;
        
        // Remove the vertex's adjacency list
        adjacencyList.erase(adjacencyList.begin() + vertex);
        
        // Remove all edges pointing to this vertex
        for (auto& adj : adjacencyList) {
            adj.erase(std::remove_if(adj.begin(), adj.end(),
                [vertex](const std::pair<int, int>& edge) {
                    return edge.first == vertex;
                }), adj.end());
            
            // Update target indices for vertices that come after the removed one
            for (auto& edge : adj) {
                if (edge.first > vertex) {
                    edge.first--;
                }
            }
        }
        
        // Update the number of vertices
        numVertices--;
        
        // Update distances and predecessors arrays
        distances.erase(distances.begin() + vertex);
        predecessors.erase(predecessors.begin() + vertex);
    }
    
    // Compute initial SSSP using Dijkstra's algorithm
    void computeInitialSSSP(int source) {
        
        std::fill(distances.begin(), distances.end(), std::numeric_limits<int>::max());
        std::fill(predecessors.begin(), predecessors.end(), -1);
        
        
        distances[source] = 0;
        
        // Priority queue for Dijkstra's algorithm
        std::priority_queue<std::pair<int, int>, 
                           std::vector<std::pair<int, int>>, 
                           std::greater<std::pair<int, int>>> pq;
        
        pq.push(std::make_pair(0, source));
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int dist_u = pq.top().first;
            pq.pop();
            
            
            if (dist_u > distances[u]) continue;
            
            
            for (const auto& edge : adjacencyList[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                // Relaxation step
                if (distances[u] != std::numeric_limits<int>::max() && 
                    distances[v] > distances[u] + weight) {
                    distances[v] = distances[u] + weight;
                    predecessors[v] = u;
                    pq.push(std::make_pair(distances[v], v));
                }
            }
        }
    }
    
    // Update SSSP after edge insertion
    void updateAfterEdgeInsertion(int source, int u, int v, int weight) {
        // First, add the edge
        addEdge(u, v, weight);
        
        // Check if this new edge creates a shorter path
        if (distances[u] != std::numeric_limits<int>::max() && 
            (distances[v] > distances[u] + weight)) {
            
            // Step 1: Identify affected nodes
            std::unordered_set<int> affectedNodes;
            affectedNodes.insert(v);
            
            // Queue for BFS traversal of affected nodes
            std::queue<int> queue;
            queue.push(v);
            
            // Update v's distance
            distances[v] = distances[u] + weight;
            predecessors[v] = u;
            
            // Step 2: Iterative updates to propagate changes
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                
                for (const auto& edge : adjacencyList[current]) {
                    int neighbor = edge.first;
                    int edgeWeight = edge.second;
                    
                    // If we can improve the path to neighbor
                    if (distances[neighbor] > distances[current] + edgeWeight) {
                        distances[neighbor] = distances[current] + edgeWeight;
                        predecessors[neighbor] = current;
                        
                        // Add to affected nodes and queue
                        if (affectedNodes.find(neighbor) == affectedNodes.end()) {
                            affectedNodes.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
            }
            
            std::cout << "Edge insertion: (" << u << " -> " << v << ", weight: " << weight << ")" << std::endl;
            std::cout << "Affected nodes: ";
            for (int node : affectedNodes) {
                std::cout << node << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Update SSSP after edge deletion
    void updateAfterEdgeDeletion(int source, int u, int v) {
        // Check if this edge was part of the shortest path tree
        bool wasInSPT = false;
        for (int i = 0; i < numVertices; i++) {
            if (predecessors[i] == u && i == v) {
                wasInSPT = true;
                break;
            }
        }
        
        // Remove the edge
        removeEdge(u, v);
        
        if (wasInSPT) {
            // Step 1: Identify affected nodes
            std::unordered_set<int> affectedNodes;
            
            
            std::queue<int> queue;
            queue.push(v);
            
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                affectedNodes.insert(current);
                
                // Add all children of current node
                for (int i = 0; i < numVertices; i++) {
                    if (predecessors[i] == current) {
                        queue.push(i);
                    }
                }
            }
            
            // Step 2: Reset distances for affected nodes
            for (int node : affectedNodes) {
                distances[node] = std::numeric_limits<int>::max();
                predecessors[node] = -1;
            }
            
            // Step 3: Recompute distances for affected nodes
            std::priority_queue<std::pair<int, int>, 
                              std::vector<std::pair<int, int>>, 
                              std::greater<std::pair<int, int>>> pq;
            
            
            for (int i = 0; i < numVertices; i++) {
                if (affectedNodes.find(i) == affectedNodes.end()) {
                    for (const auto& edge : adjacencyList[i]) {
                        int target = edge.first;
                        int weight = edge.second;
                        
                        if (affectedNodes.find(target) != affectedNodes.end()) {
                            if (distances[i] != std::numeric_limits<int>::max() &&
                                (distances[target] > distances[i] + weight)) {
                                distances[target] = distances[i] + weight;
                                predecessors[target] = i;
                                pq.push(std::make_pair(distances[target], target));
                            }
                        }
                    }
                }
            }
            
            
            while (!pq.empty()) {
                int current = pq.top().second;
                int current_dist = pq.top().first;
                pq.pop();
                
                if (current_dist > distances[current]) continue;
                
                for (const auto& edge : adjacencyList[current]) {
                    int neighbor = edge.first;
                    int weight = edge.second;
                    
                    if (distances[neighbor] > distances[current] + weight) {
                        distances[neighbor] = distances[current] + weight;
                        predecessors[neighbor] = current;
                        pq.push(std::make_pair(distances[neighbor], neighbor));
                    }
                }
            }
            
            std::cout << "Edge deletion: (" << u << " -> " << v << ")" << std::endl;
            std::cout << "Affected nodes: ";
            for (int node : affectedNodes) {
                std::cout << node << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Update SSSP after edge weight update
    void updateAfterEdgeWeightChange(int source, int u, int v, int newWeight) {
        // First check if the edge exists
        bool edgeExists = false;
        int oldWeight = 0;
        
        for (const auto& edge : adjacencyList[u]) {
            if (edge.first == v) {
                edgeExists = true;
                oldWeight = edge.second;
                break;
            }
        }
        
        if (!edgeExists) {
            std::cout << "Edge (" << u << " -> " << v << ") does not exist!" << std::endl;
            return;
        }
        
        // Update the edge weight
        updateEdgeWeight(u, v, newWeight);
        
        // Check if edge is in the shortest path tree
        bool isInSPT = (predecessors[v] == u);
        
        if (isInSPT) {
            if (newWeight < oldWeight) {
                
                
                // Step 1: Identify affected nodes
                std::unordered_set<int> affectedNodes;
                affectedNodes.insert(v);
                
                // Queue for BFS traversal of affected nodes
                std::queue<int> queue;
                queue.push(v);
                
                // Update v's distance
                distances[v] = distances[u] + newWeight;
                
                // Step 2: Iterative updates to propagate changes
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    
                    for (const auto& edge : adjacencyList[current]) {
                        int neighbor = edge.first;
                        int edgeWeight = edge.second;
                        
                        
                        if (distances[neighbor] > distances[current] + edgeWeight) {
                            distances[neighbor] = distances[current] + edgeWeight;
                            predecessors[neighbor] = current;
                            
                            // Add to affected nodes and queue
                            if (affectedNodes.find(neighbor) == affectedNodes.end()) {
                                affectedNodes.insert(neighbor);
                                queue.push(neighbor);
                            }
                        }
                    }
                }
                
                std::cout << "Edge weight decreased: (" << u << " -> " << v << ", " << oldWeight << " -> " << newWeight << ")" << std::endl;
                std::cout << "Affected nodes: ";
                for (int node : affectedNodes) {
                    std::cout << node << " ";
                }
                std::cout << std::endl;
            }
            else if (newWeight > oldWeight) {
                // Weight increased - need to recompute potentially affected paths
                
                // Step 1: Identify affected nodes
                std::unordered_set<int> affectedNodes;
                
                // Find all descendants of v in the SPT
                std::queue<int> queue;
                queue.push(v);
                
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    affectedNodes.insert(current);
                    
                    // Add all children of current node
                    for (int i = 0; i < numVertices; i++) {
                        if (predecessors[i] == current) {
                            queue.push(i);
                        }
                    }
                }
                
                // Step 2: Reset distances for affected nodes
                for (int node : affectedNodes) {
                    distances[node] = std::numeric_limits<int>::max();
                    predecessors[node] = -1;
                }
                
                // Step 3: Recompute distances for affected nodes
                std::priority_queue<std::pair<int, int>, 
                                  std::vector<std::pair<int, int>>, 
                                  std::greater<std::pair<int, int>>> pq;
                
                
                for (int i = 0; i < numVertices; i++) {
                    if (affectedNodes.find(i) == affectedNodes.end()) {
                        for (const auto& edge : adjacencyList[i]) {
                            int target = edge.first;
                            int weight = edge.second;
                            
                            if (affectedNodes.find(target) != affectedNodes.end()) {
                                if (distances[i] != std::numeric_limits<int>::max() &&
                                    (distances[target] > distances[i] + weight)) {
                                    distances[target] = distances[i] + weight;
                                    predecessors[target] = i;
                                    pq.push(std::make_pair(distances[target], target));
                                }
                            }
                        }
                    }
                }
                
                // Process the priority queue
                while (!pq.empty()) {
                    int current = pq.top().second;
                    int current_dist = pq.top().first;
                    pq.pop();
                    
                    if (current_dist > distances[current]) continue;
                    
                    for (const auto& edge : adjacencyList[current]) {
                        int neighbor = edge.first;
                        int weight = edge.second;
                        
                        if (distances[neighbor] > distances[current] + weight) {
                            distances[neighbor] = distances[current] + weight;
                            predecessors[neighbor] = current;
                            pq.push(std::make_pair(distances[neighbor], neighbor));
                        }
                    }
                }
                
                std::cout << "Edge weight increased: (" << u << " -> " << v << ", " << oldWeight << " -> " << newWeight << ")" << std::endl;
                std::cout << "Affected nodes: ";
                for (int node : affectedNodes) {
                    std::cout << node << " ";
                }
                std::cout << std::endl;
            }
        }
        else {
            // Edge not in SPT - check if new weight creates a better path
            if (distances[u] != std::numeric_limits<int>::max() && 
                (distances[v] > distances[u] + newWeight)) {
                
                updateAfterEdgeInsertion(source, u, v, newWeight);
            }
        }
    }
    
    
    void printSSSP(int source) {
        std::cout << "Single-Source Shortest Paths from vertex " << source << ":" << std::endl;
        for (int i = 0; i < numVertices; i++) {
            if (distances[i] == std::numeric_limits<int>::max()) {
                std::cout << "Vertex " << i << ": Infinity (Predecessor: None)" << std::endl;
            } else {
                std::cout << "Vertex " << i << ": " << distances[i] << " (Predecessor: ";
                if (predecessors[i] != -1) {
                    std::cout << predecessors[i] << ")" << std::endl;
                } else {
                    std::cout << "None)" << std::endl;
                }
            }
        }
    }
    
    // Save graph to file
    void saveGraphToFile(const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        
        outFile << numVertices << std::endl;
        
        // Write edges
        for (int i = 0; i < numVertices; i++) {
            for (const auto& edge : adjacencyList[i]) {
                outFile << i << " " << edge.first << " " << edge.second << std::endl;
            }
        }
        
        outFile.close();
        std::cout << "Graph saved to " << filename << std::endl;
    }
    
    // Load graph from file
    static Graph loadGraphFromFile(const std::string& filename) {
        std::ifstream inFile(filename);
        if (!inFile.is_open()) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return Graph(0);
        }
        
        int numVertices;
        inFile >> numVertices;
        
        Graph graph(numVertices);
        
        int source, target, weight;
        while (inFile >> source >> target >> weight) {
            graph.addEdge(source, target, weight);
        }
        
        inFile.close();
        std::cout << "Graph loaded from " << filename << std::endl;
        
        return graph;
    }
};

int main() {
    // Create a sample graph
    Graph g(6);
    
    // Add some edges
    g.addEdge(0, 1, 5);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 3, 3);
    g.addEdge(2, 1, 2);
    g.addEdge(2, 3, 6);
    g.addEdge(2, 4, 3);
    g.addEdge(3, 5, 2);
    g.addEdge(4, 3, 1);
    g.addEdge(4, 5, 4);
    
    int source = 0;
    
    // Compute initial SSSP
    g.computeInitialSSSP(source);
    std::cout << "Initial SSSP:" << std::endl;
    g.printSSSP(source);
    
    // Test edge insertion
    std::cout << "\nInserting edge (0 -> 4, weight: 2)..." << std::endl;
    g.updateAfterEdgeInsertion(source, 0, 4, 2);
    g.printSSSP(source);
    
    // Test edge deletion
    std::cout << "\nDeleting edge (2 -> 4)..." << std::endl;
    g.updateAfterEdgeDeletion(source, 2, 4);
    g.printSSSP(source);
    
    // Test edge weight update
    std::cout << "\nUpdating edge (2 -> 1, weight: 1)..." << std::endl;
    g.updateAfterEdgeWeightChange(source, 2, 1, 1);
    g.printSSSP(source);
    
    // Test adding a new vertex
    std::cout << "\nAdding a new vertex (6)..." << std::endl;
    g.addVertex();
    g.addEdge(5, 6, 1);
    g.computeInitialSSSP(source); // Recompute SSSP after adding a vertex
    g.printSSSP(source);
    
    // Test removing a vertex
    std::cout << "\nRemoving vertex 4..." << std::endl;
    g.removeVertex(4);
    g.computeInitialSSSP(source); // Recompute SSSP after removing a vertex  
    g.printSSSP(source);
    
    // Save graph to file
    g.saveGraphToFile("graph.txt");
    
    // Load graph from file
    Graph g2 = Graph::loadGraphFromFile("graph.txt");
    g2.computeInitialSSSP(source);
    std::cout << "\nLoaded graph SSSP:" << std::endl;
    g2.printSSSP(source);
    
    logPerformance("Sequential", "GraphDataset", duration.count());
    system("python3 Analysis.py");
    return 0;
}