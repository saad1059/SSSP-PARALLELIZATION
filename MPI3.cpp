#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <mpi.h>
#include <metis.h>

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

// For performance measurement
class Timer {
private:
    double startTime;
public:
    Timer() : startTime(MPI_Wtime()) {}
    
    double elapsed() {
        return MPI_Wtime() - startTime;
    }
    
    void reset() {
        startTime = MPI_Wtime();
    }
};

class DistributedGraph {
private:
    int rank;            
    int numProcesses;    
    int numVertices;    
    int localVerticesStart; 
    int localVerticesEnd;   
    std::vector<int> partition;  
    
    // Local graph representation
    std::vector<std::vector<std::pair<int, int>>> localAdjacencyList; 
    std::vector<std::vector<std::pair<int, int>>> ghostAdjacencyList; // Ghost nodes adjacency list
    
    // SSSP data
    std::vector<int> localDistances;       
    std::vector<int> globalDistances;      
    std::vector<int> localPredecessors;    
    std::vector<int> globalPredecessors;   
    
    // Ghost node management
    std::unordered_set<int> ghostNodes;    // Set of ghost nodes
    std::unordered_map<int, std::vector<int>> ghostNodeOwners; // Maps ghost node to its owners
    
    // Profiling info
    double commTime;       
    double computeTime;  
    int messagesSent;  
    int messagesReceived;  
    long long dataSent;    
    
    
    bool isLocal(int vertex) {
        return vertex >= localVerticesStart && vertex < localVerticesEnd;
    }
    
    
    void initializeLocalStructures(const std::vector<std::vector<std::pair<int, int>>>& fullAdjacencyList) {
        // Initialize local adjacency list
        localAdjacencyList.resize(numVertices);
        
        // Determine local vertices range
        int verticesPerProcess = numVertices / numProcesses;
        int remainingVertices = numVertices % numProcesses;
        
        // Calculate start and end indices for local vertices
        localVerticesStart = rank * verticesPerProcess + std::min(rank, remainingVertices);
        int extraVertex = (rank < remainingVertices) ? 1 : 0;
        localVerticesEnd = localVerticesStart + verticesPerProcess + extraVertex;
        
        
        localDistances.resize(numVertices, std::numeric_limits<int>::max());
        localPredecessors.resize(numVertices, -1);
        globalDistances.resize(numVertices, std::numeric_limits<int>::max());
        globalPredecessors.resize(numVertices, -1);
        
        
        for (int i = localVerticesStart; i < localVerticesEnd; i++) {
            localAdjacencyList[i] = fullAdjacencyList[i];
            
            
            for (const auto& edge : localAdjacencyList[i]) {
                int target = edge.first;
                if (!isLocal(target)) {
                    ghostNodes.insert(target);
                    ghostNodeOwners[target].push_back(getProcessForVertex(target));
                }
            }
        }
        
        
        ghostAdjacencyList.resize(numVertices);
        
        // For each ghost node, find its outgoing edges that point to local nodes
        for (int i = 0; i < numVertices; i++) {
            if (isLocal(i)) continue; // Skip local vertices
            
            for (const auto& edge : fullAdjacencyList[i]) {
                int target = edge.first;
                if (isLocal(target)) {
                    // This is a ghost node with an edge to a local node
                    ghostAdjacencyList[i].push_back(edge);
                    ghostNodes.insert(i);
                }
            }
        }
    }
    
    
    int getProcessForVertex(int vertex) {
        int verticesPerProcess = numVertices / numProcesses;
        int remainingVertices = numVertices % numProcesses;
        
        
        if (vertex < remainingVertices * (verticesPerProcess + 1)) {
            return vertex / (verticesPerProcess + 1);
        } else {
            int adjustedVertex = vertex - remainingVertices;
            return remainingVertices + (adjustedVertex / verticesPerProcess);
        }
    }
    
    // Synchronize ghost node data with their owners
    void synchronizeGhostNodes() {
        Timer timer;
        
        
        std::vector<MPI_Request> requests;
        std::vector<std::vector<int>> sendBuffers(numProcesses);
        std::vector<std::vector<int>> recvBuffers(numProcesses);
        
        // First, determine what data to send to each process
        for (int p = 0; p < numProcesses; p++) {
            if (p == rank) continue; // Skip self
            
            for (int v = localVerticesStart; v < localVerticesEnd; v++) {
                // Check if this local vertex is a ghost in process p
                
                bool isGhostInP = false;
                
                for (int i = p * (numVertices / numProcesses); 
                     i < (p + 1) * (numVertices / numProcesses) && i < numVertices; i++) {
                    
                    // Check if any vertex in p has an edge to this local vertex
                    for (const auto& edge : localAdjacencyList[i]) {
                        if (edge.first == v) {
                            isGhostInP = true;
                            break;
                        }
                    }
                    
                    if (isGhostInP) break;
                }
                
                if (isGhostInP) {
                    // Add vertex, distance, and predecessor to send buffer
                    sendBuffers[p].push_back(v);
                    sendBuffers[p].push_back(localDistances[v]);
                    sendBuffers[p].push_back(localPredecessors[v]);
                }
            }
        }
        
        
        for (int p = 0; p < numProcesses; p++) {
            if (p == rank) continue;
            
            int sendSize = sendBuffers[p].size();
            MPI_Request req;
            MPI_Isend(&sendSize, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            
            messagesSent++;
            dataSent += sizeof(int);
        }
        
        // Receive buffer sizes from each process
        for (int p = 0; p < numProcesses; p++) {
            if (p == rank) continue;
            
            int recvSize;
            MPI_Recv(&recvSize, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recvBuffers[p].resize(recvSize);
            
            messagesReceived++;
        }
        
        // Wait for size sends to complete
        if (!requests.empty()) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
        }
        
        // Send data to each process
        for (int p = 0; p < numProcesses; p++) {
            if (p == rank || sendBuffers[p].empty()) continue;
            
            MPI_Request req;
            MPI_Isend(sendBuffers[p].data(), sendBuffers[p].size(), MPI_INT, p, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            
            messagesSent++;
            dataSent += sendBuffers[p].size() * sizeof(int);
        }
        
        // Receive data from each process
        for (int p = 0; p < numProcesses; p++) {
            if (p == rank || recvBuffers[p].empty()) continue;
            
            MPI_Recv(recvBuffers[p].data(), recvBuffers[p].size(), MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            messagesReceived++;
            
            // Update ghost node information
            for (size_t i = 0; i < recvBuffers[p].size(); i += 3) {
                int vertex = recvBuffers[p][i];
                int distance = recvBuffers[p][i + 1];
                int predecessor = recvBuffers[p][i + 2];
                
                // Update only if it's a better distance
                if (distance < localDistances[vertex]) {
                    localDistances[vertex] = distance;
                    localPredecessors[vertex] = predecessor;
                }
            }
        }
        
        // Wait for data sends to complete
        if (!requests.empty()) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        
        commTime += timer.elapsed();
    }
    
    // Broadcast the SSSP tree to all processes
    void broadcastSSSP() {
        Timer timer;
        
        // Gather all distances and predecessors
        MPI_Allreduce(localDistances.data(), globalDistances.data(), numVertices, 
                      MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        MPI_Allreduce(localPredecessors.data(), globalPredecessors.data(), numVertices, 
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        
        localDistances = globalDistances;
        localPredecessors = globalPredecessors;
        
        commTime += timer.elapsed();
    }
    
public:
    // Constructor
    DistributedGraph(int rank, int numProcesses) 
        : rank(rank), numProcesses(numProcesses), numVertices(0),
          commTime(0), computeTime(0), messagesSent(0), messagesReceived(0), dataSent(0) {}
    
    
    void loadAndPartitionGraph(const std::string& filename) {
        std::vector<std::vector<std::pair<int, int>>> fullAdjacencyList;
        std::vector<Edge> allEdges;
        
        
        if (rank == 0) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            file >> numVertices;
            fullAdjacencyList.resize(numVertices);
            
            int src, dst, weight;
            while (file >> src >> dst >> weight) {
                fullAdjacencyList[src].push_back(std::make_pair(dst, weight));
                allEdges.push_back(Edge(src, dst, weight));
            }
            
            file.close();
        }
        
        // Broadcast the number of vertices to all processes
        MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            fullAdjacencyList.resize(numVertices);
        }
        
        
        if (rank == 0) {
            
            // Create CSR format
            std::vector<idx_t> xadj(numVertices + 1, 0);
            std::vector<idx_t> adjncy;
            std::vector<idx_t> adjwgt;
            
            // Count edges per vertex
            std::vector<int> edgeCount(numVertices, 0);
            for (const auto& edge : allEdges) {
                edgeCount[edge.source]++;
            }
            
            // Create xadj array
            xadj[0] = 0;
            for (int i = 0; i < numVertices; i++) {
                xadj[i + 1] = xadj[i] + edgeCount[i];
            }
            
            // Fill adjncy and adjwgt arrays
            adjncy.resize(allEdges.size());
            adjwgt.resize(allEdges.size());
            std::vector<int> currentIndex(numVertices, 0);
            
            for (const auto& edge : allEdges) {
                int pos = xadj[edge.source] + currentIndex[edge.source];
                adjncy[pos] = edge.target;
                adjwgt[pos] = edge.weight;
                currentIndex[edge.source]++;
            }
            
            // Partition the graph using METIS
            partition.resize(numVertices);
            idx_t ncon = 1; // Number of balancing constraints
            idx_t objval; // Stores the edge-cut or communication volume
            
            // Convert to idx_t for METIS
            std::vector<idx_t> partitionIdx(numVertices);
            
            // METIS partitioning options
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // Minimize edge-cut
            
            // Call METIS for partitioning
            int metisResult = METIS_PartGraphKway(
                &numVertices,  // Number of vertices
                &ncon,         // Number of balancing constraints
                xadj.data(),   // Adjacency structure: indices to adjncy
                adjncy.data(), // Adjacency structure: neighbors
                NULL,          // Vertex weights (NULL = all 1's)
                NULL,          // Size of vertices for volume calculation
                adjwgt.data(), // Edge weights
                &numProcesses, // Number of parts
                NULL,          // Target partition weights for each constraint
                NULL,          // Allowed load imbalance for each constraint
                options,       // Array of options
                &objval,       // Output: Objective value (edge-cut or comm volume)
                partitionIdx.data() // Output: Partition vector
            );
            
            if (metisResult != METIS_OK) {
                std::cerr << "METIS partitioning failed with error code: " << metisResult << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Convert partitionIdx to partition
            for (int i = 0; i < numVertices; i++) {
                partition[i] = partitionIdx[i];
            }
            
            std::cout << "METIS partitioning completed with edge-cut: " << objval << std::endl;
            
            
            std::vector<int> partSizes(numProcesses, 0);
            for (int i = 0; i < numVertices; i++) {
                partSizes[partition[i]]++;
            }
            
            std::cout << "Partition sizes: ";
            for (int i = 0; i < numProcesses; i++) {
                std::cout << "P" << i << "=" << partSizes[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // Broadcast partition information to all processes
        partition.resize(numVertices);
        MPI_Bcast(partition.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        
        
        std::vector<int> serializedAdj;
        
        if (rank == 0) {
            // Format: [vertex, number of neighbors, (neighbor, weight), ...]
            for (int i = 0; i < numVertices; i++) {
                serializedAdj.push_back(i);
                serializedAdj.push_back(fullAdjacencyList[i].size());
                
                for (const auto& edge : fullAdjacencyList[i]) {
                    serializedAdj.push_back(edge.first);
                    serializedAdj.push_back(edge.second);
                }
            }
        }
        
        
        int serializedSize = serializedAdj.size();
        MPI_Bcast(&serializedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        
        if (rank != 0) {
            serializedAdj.resize(serializedSize);
        }
        
        MPI_Bcast(serializedAdj.data(), serializedSize, MPI_INT, 0, MPI_COMM_WORLD);
        
        
        if (rank != 0) {
            int index = 0;
            while (index < serializedSize) {
                int vertex = serializedAdj[index++];
                int numNeighbors = serializedAdj[index++];
                
                for (int j = 0; j < numNeighbors; j++) {
                    int neighbor = serializedAdj[index++];
                    int weight = serializedAdj[index++];
                    fullAdjacencyList[vertex].push_back(std::make_pair(neighbor, weight));
                }
            }
        }
        
        
        initializeLocalStructures(fullAdjacencyList);
        
        std::cout << "Process " << rank << " initialized with vertices [" 
                  << localVerticesStart << " - " << localVerticesEnd - 1 << "]" << std::endl;
        std::cout << "Process " << rank << " has " << ghostNodes.size() << " ghost nodes" << std::endl;
    }
    
    
    void computeInitialSSSP(int source) {
        Timer timer;
        
        // Reset distances and predecessors
        std::fill(localDistances.begin(), localDistances.end(), std::numeric_limits<int>::max());
        std::fill(localPredecessors.begin(), localPredecessors.end(), -1);
        
        // Set source distance to 0
        if (isLocal(source)) {
            localDistances[source] = 0;
        }
        
        // Synchronize initial distances
        broadcastSSSP();
        
        
        bool changed = true;
        int iteration = 0;
        
        while (changed) {
            changed = false;
            iteration++;
            
            // Process local vertices
            for (int u = localVerticesStart; u < localVerticesEnd; u++) {
                
                if (localDistances[u] == std::numeric_limits<int>::max()) continue;
                
                
                for (const auto& edge : localAdjacencyList[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    
                    // Relaxation step
                    if (localDistances[v] > localDistances[u] + weight) {
                        localDistances[v] = localDistances[u] + weight;
                        localPredecessors[v] = u;
                        changed = true;
                    }
                }
            }
            
            
            for (int u : ghostNodes) {
                // Skip unreachable vertices
                if (localDistances[u] == std::numeric_limits<int>::max()) continue;
                
                // Explore edges to local vertices
                for (const auto& edge : ghostAdjacencyList[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    
                    // Only process if v is local
                    if (isLocal(v)) {
                        // Relaxation step
                        if (localDistances[v] > localDistances[u] + weight) {
                            localDistances[v] = localDistances[u] + weight;
                            localPredecessors[v] = u;
                            changed = true;
                        }
                    }
                }
            }
            
            
            int localChanged = changed ? 1 : 0;
            int globalChanged;
            
            MPI_Allreduce(&localChanged, &globalChanged, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            changed = (globalChanged != 0);
            
            
            if (changed) {
                synchronizeGhostNodes();
            }
        }
        
        
        broadcastSSSP();
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Initial SSSP computation completed in " << iteration 
                      << " iterations, " << timer.elapsed() << " seconds" << std::endl;
        }
    }
    
    
    void updateAfterEdgeInsertion(int source, int u, int v, int weight) {
        Timer timer;
        
        
        if (isLocal(u)) {
            // Check if edge already exists
            bool edgeExists = false;
            for (auto& edge : localAdjacencyList[u]) {
                if (edge.first == v) {
                    edge.second = weight;
                    edgeExists = true;
                    break;
                }
            }
            
            if (!edgeExists) {
                localAdjacencyList[u].push_back(std::make_pair(v, weight));
                
                // Update ghost nodes if v is not local
                if (!isLocal(v)) {
                    ghostNodes.insert(v);
                    ghostNodeOwners[v].push_back(getProcessForVertex(v));
                }
            }
        }
        
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        bool affectedLocally = false;
        
        if (localDistances[u] != std::numeric_limits<int>::max() && 
            (localDistances[v] > localDistances[u] + weight)) {
            
            // Identify affected nodes
            std::unordered_set<int> affectedNodes;
            std::queue<int> queue;
            
            // Only add v to affected nodes if local or ghost
            if (isLocal(v) || ghostNodes.count(v) > 0) {
                affectedNodes.insert(v);
                queue.push(v);
                
                // Update v's distance
                localDistances[v] = localDistances[u] + weight;
                localPredecessors[v] = u;
                affectedLocally = true;
            }
            
            // Process affected nodes
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                // Check all neighbors of current node
                std::vector<std::pair<int, int>> neighbors;
                
                if (isLocal(current)) {
                    neighbors = localAdjacencyList[current];
                } else if (ghostNodes.count(current) > 0) {
                    neighbors = ghostAdjacencyList[current];
                }
                
                for (const auto& edge : neighbors) {
                    int neighbor = edge.first;
                    int edgeWeight = edge.second;
                    
                    // Skip if neighbor is not local or ghost
                    if (!isLocal(neighbor) && ghostNodes.count(neighbor) == 0) continue;
                    
                    // If we can improve the path to neighbor
                    if (localDistances[neighbor] > localDistances[current] + edgeWeight) {
                        localDistances[neighbor] = localDistances[current] + edgeWeight;
                        localPredecessors[neighbor] = current;
                        affectedLocally = true;
                        
                        // Add to affected nodes and queue
                        if (affectedNodes.find(neighbor) == affectedNodes.end()) {
                            affectedNodes.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
            }
            
            // Synchronize updated distances
            if (affectedLocally) {
                synchronizeGhostNodes();
                broadcastSSSP();
            }
        }
        
        computeTime += timer.elapsed();
        
        // Gather information about affected nodes for reporting
        int locallyAffected = affectedLocally ? 1 : 0;
        int globallyAffected;
        
        MPI_Allreduce(&locallyAffected, &globallyAffected, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        if (rank == 0 && globallyAffected) {
            std::cout << "Edge insertion: (" << u << " -> " << v 
                      << ", weight: " << weight << ") affected the SSSP tree" << std::endl;
        } else if (rank == 0) {
            std::cout << "Edge insertion: (" << u << " -> " << v 
                      << ", weight: " << weight << ") did not affect the SSSP tree" << std::endl;
        }
    }
    
    
    void updateAfterEdgeDeletion(int source, int u, int v) {
        Timer timer;
        
        
        bool wasInSPT = false;
        if (localPredecessors[v] == u) {
            wasInSPT = true;
        }
        
        
        int localInSPT = wasInSPT ? 1 : 0;
        int globalInSPT;
        MPI_Allreduce(&localInSPT, &globalInSPT, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        wasInSPT = (globalInSPT != 0);
        
        
        if (isLocal(u)) {
            auto& edges = localAdjacencyList[u];
            for (auto it = edges.begin(); it != edges.end(); ++it) {
                if (it->first == v) {
                    edges.erase(it);
                    break;
                }
            }
        }
        
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (wasInSPT) {
            // Identify potentially affected nodes
            std::unordered_set<int> locallyAffectedNodes;
            std::queue<int> queue;
            
            // Check if v is local or ghost
            if (isLocal(v) || ghostNodes.count(v) > 0) {
                locallyAffectedNodes.insert(v);
                queue.push(v);
            }
            
            // Find descendants of v in the SPT that are local or ghost
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                // Add all local children
                for (int i = localVerticesStart; i < localVerticesEnd; i++) {
                    if (localPredecessors[i] == current) {
                        locallyAffectedNodes.insert(i);
                        queue.push(i);
                    }
                }
                
                // Add ghost children (if we know about them)
                for (auto& ghostEdge : ghostAdjacencyList[current]) {
                    int ghostChild = ghostEdge.first;
                    if (localPredecessors[ghostChild] == current) {
                        locallyAffectedNodes.insert(ghostChild);
                        queue.push(ghostChild);
                    }
                }
            }
            
            // Reset distances for affected nodes
            for (int node : locallyAffectedNodes) {
                if (isLocal(node) || ghostNodes.count(node) > 0) {
                    localDistances[node] = std::numeric_limits<int>::max();
                    localPredecessors[node] = -1;
                }
            }
            
            // Synchronize reset distances
            synchronizeGhostNodes();
            
            // Set up priority queue for processing boundary nodes
            std::priority_queue<std::pair<int, int>, 
                              std::vector<std::pair<int, int>>, 
                              std::greater<std::pair<int, int>>> pq;
            
            // Check all local nodes and their edges to affected nodes
            for (int i = localVerticesStart; i < localVerticesEnd; i++) {
                if (locallyAffectedNodes.find(i) == locallyAffectedNodes.end() && 
                    localDistances[i] != std::numeric_limits<int>::max()) {
                    
                    for (const auto& edge : localAdjacencyList[i]) {
                        int target = edge.first;
                        int weight = edge.second;
                        
                        if (locallyAffectedNodes.find(target) != locallyAffectedNodes.end()) {
                            if (localDistances[target] > localDistances[i] + weight) {
                                localDistances[target] = localDistances[i] + weight;
                                localPredecessors[target] = i;
                                pq.push(std::make_pair(localDistances[target], target));
                            }
                        }
                    }
                }
            }
            
            // Check ghost nodes and their edges to affected nodes
            for (int ghostNode : ghostNodes) {
                if (locallyAffectedNodes.find(ghostNode) == locallyAffectedNodes.end() && 
                    localDistances[ghostNode] != std::numeric_limits<int>::max()) {
                    
                    for (const auto& edge : ghostAdjacencyList[ghostNode]) {
                        int target = edge.first;
                        int weight = edge.second;
                        
                        if (locallyAffectedNodes.find(target) != locallyAffectedNodes.end() &&
                            isLocal(target)) {
                            if (localDistances[target] > localDistances[ghostNode] + weight) {
                                localDistances[target] = localDistances[ghostNode] + weight;
                                localPredecessors[target] = ghostNode;
                                pq.push(std::make_pair(localDistances[target], target));
                            }
                        }
                    }
                }
            }
            
            // Dijkstra's algorithm for affected nodes
            while (!pq.empty()) {
                int dist = pq.top().first;
                int node = pq.top().second;
                pq.pop();
                
                // Skip if we already found a better path
                if (dist > localDistances[node]) continue;
                
                // Process the node if it's local
                if (isLocal(node)) {
                    for (const auto& edge : localAdjacencyList[node]) {
                        int target = edge.first;
                        int weight = edge.second;
                        
                        if (localDistances[target] > dist + weight) {
                            localDistances[target] = dist + weight;
                            localPredecessors[target] = node;
                            
                            if (isLocal(target) || ghostNodes.count(target) > 0) {
                                pq.push(std::make_pair(localDistances[target], target));
                            }
                        }
                    }
                }
                
                // Process ghost node
                else if (ghostNodes.count(node) > 0) {
                    for (const auto& edge : ghostAdjacencyList[node]) {
                        int target = edge.first;
                        int weight = edge.second;
                        
                        if (isLocal(target) && localDistances[target] > dist + weight) {
                            localDistances[target] = dist + weight;
                            localPredecessors[target] = node;
                            pq.push(std::make_pair(localDistances[target], target));
                        }
                    }
                }
            }
            
            // Synchronize updated distances and predecessors
            synchronizeGhostNodes();
            broadcastSSSP();
        }
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Edge deletion: (" << u << " -> " << v 
                      << ") " << (wasInSPT ? "affected" : "did not affect") 
                      << " the SSSP tree" << std::endl;
        }
    }
    
    // Update SSSP after edge weight change
    void updateAfterEdgeWeightChange(int source, int u, int v, int oldWeight, int newWeight) {
        Timer timer;
        
        // Check if this is a weight increase or decrease
        if (newWeight > oldWeight) {
            // Handle as weight increase (similar to edge deletion)
            bool wasInSPT = false;
            if (localPredecessors[v] == u) {
                wasInSPT = true;
            }
            
            // Synchronize this information
            int localInSPT = wasInSPT ? 1 : 0;
            int globalInSPT;
            MPI_Allreduce(&localInSPT, &globalInSPT, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            wasInSPT = (globalInSPT != 0);
            
            // Update the edge weight if it's local
            if (isLocal(u)) {
                for (auto& edge : localAdjacencyList[u]) {
                    if (edge.first == v) {
                        edge.second = newWeight;
                        break;
                    }
                }
            }
            
            // Update ghost adjacency list if needed
            if (ghostAdjacencyList[u].size() > 0) {
                for (auto& edge : ghostAdjacencyList[u]) {
                    if (edge.first == v) {
                        edge.second = newWeight;
                        break;
                    }
                }
            }
            
            // Synchronize modified adjacency lists
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (wasInSPT && localDistances[u] + newWeight > localDistances[v]) {
                // Current path through u is no longer optimal for v
                // Similar logic as edge deletion
                std::unordered_set<int> locallyAffectedNodes;
                std::queue<int> queue;
                
                if (isLocal(v) || ghostNodes.count(v) > 0) {
                    locallyAffectedNodes.insert(v);
                    queue.push(v);
                }
                
                // Find descendants of v in SPT
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    // Add all local children
                    for (int i = localVerticesStart; i < localVerticesEnd; i++) {
                        if (localPredecessors[i] == current) {
                            locallyAffectedNodes.insert(i);
                            queue.push(i);
                        }
                    }
                    
                    // Add ghost children
                    for (auto& ghostEdge : ghostAdjacencyList[current]) {
                        int ghostChild = ghostEdge.first;
                        if (localPredecessors[ghostChild] == current) {
                            locallyAffectedNodes.insert(ghostChild);
                            queue.push(ghostChild);
                        }
                    }
                }
                
                // Reset distances for affected nodes
                for (int node : locallyAffectedNodes) {
                    if (isLocal(node) || ghostNodes.count(node) > 0) {
                        localDistances[node] = std::numeric_limits<int>::max();
                        localPredecessors[node] = -1;
                    }
                }
                
                // Synchronize reset distances
                synchronizeGhostNodes();
                
                // Set up priority queue for processing boundary nodes
                std::priority_queue<std::pair<int, int>, 
                                  std::vector<std::pair<int, int>>, 
                                  std::greater<std::pair<int, int>>> pq;
                
                // Add boundary nodes to priority queue
                for (int i = localVerticesStart; i < localVerticesEnd; i++) {
                    if (locallyAffectedNodes.find(i) == locallyAffectedNodes.end() && 
                        localDistances[i] != std::numeric_limits<int>::max()) {
                        
                        for (const auto& edge : localAdjacencyList[i]) {
                            int target = edge.first;
                            int weight = edge.second;
                            
                            if (locallyAffectedNodes.find(target) != locallyAffectedNodes.end()) {
                                if (localDistances[target] > localDistances[i] + weight) {
                                    localDistances[target] = localDistances[i] + weight;
                                    localPredecessors[target] = i;
                                    pq.push(std::make_pair(localDistances[target], target));
                                }
                            }
                        }
                    }
                }
                
                // Check ghost nodes and their edges to affected nodes
                for (int ghostNode : ghostNodes) {
                    if (locallyAffectedNodes.find(ghostNode) == locallyAffectedNodes.end() && 
                        localDistances[ghostNode] != std::numeric_limits<int>::max()) {
                        
                        for (const auto& edge : ghostAdjacencyList[ghostNode]) {
                            int target = edge.first;
                            int weight = edge.second;
                            
                            if (locallyAffectedNodes.find(target) != locallyAffectedNodes.end() &&
                                isLocal(target)) {
                                if (localDistances[target] > localDistances[ghostNode] + weight) {
                                    localDistances[target] = localDistances[ghostNode] + weight;
                                    localPredecessors[target] = ghostNode;
                                    pq.push(std::make_pair(localDistances[target], target));
                                }
                            }
                        }
                    }
                }
                
                // Dijkstra's algorithm for affected nodes
                while (!pq.empty()) {
                    int dist = pq.top().first;
                    int node = pq.top().second;
                    pq.pop();
                    
                    // Skip if we already found a better path
                    if (dist > localDistances[node]) continue;
                    
                    // Process the node if it's local
                    if (isLocal(node)) {
                        for (const auto& edge : localAdjacencyList[node]) {
                            int target = edge.first;
                            int weight = edge.second;
                            
                            if (localDistances[target] > dist + weight) {
                                localDistances[target] = dist + weight;
                                localPredecessors[target] = node;
                                
                                if (isLocal(target) || ghostNodes.count(target) > 0) {
                                    pq.push(std::make_pair(localDistances[target], target));
                                }
                            }
                        }
                    }
                    
                    // Process ghost node
                    else if (ghostNodes.count(node) > 0) {
                        for (const auto& edge : ghostAdjacencyList[node]) {
                            int target = edge.first;
                            int weight = edge.second;
                            
                            if (isLocal(target) && localDistances[target] > dist + weight) {
                                localDistances[target] = dist + weight;
                                localPredecessors[target] = node;
                                pq.push(std::make_pair(localDistances[target], target));
                            }
                        }
                    }
                }
                
                // Synchronize updated distances and predecessors
                synchronizeGhostNodes();
                broadcastSSSP();
            }
        } else {
            // Handle as weight decrease (similar to edge insertion)
            // Update the edge weight if it's local
            if (isLocal(u)) {
                for (auto& edge : localAdjacencyList[u]) {
                    if (edge.first == v) {
                        edge.second = newWeight;
                        break;
                    }
                }
            }
            
            // Update ghost adjacency list if needed
            if (ghostAdjacencyList[u].size() > 0) {
                for (auto& edge : ghostAdjacencyList[u]) {
                    if (edge.first == v) {
                        edge.second = newWeight;
                        break;
                    }
                }
            }
            
            // Synchronize modified adjacency lists
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Check if this creates a shorter path
            bool affectedLocally = false;
            
            if (localDistances[u] != std::numeric_limits<int>::max() && 
                (localDistances[v] > localDistances[u] + newWeight)) {
                
                // Identify affected nodes
                std::unordered_set<int> affectedNodes;
                std::queue<int> queue;
                
                // Only add v to affected nodes if local or ghost
                if (isLocal(v) || ghostNodes.count(v) > 0) {
                    affectedNodes.insert(v);
                    queue.push(v);
                    
                    // Update v's distance
                    localDistances[v] = localDistances[u] + newWeight;
                    localPredecessors[v] = u;
                    affectedLocally = true;
                }
                
                // Process affected nodes
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    // Check all neighbors of current node
                    std::vector<std::pair<int, int>> neighbors;
                    
                    if (isLocal(current)) {
                        neighbors = localAdjacencyList[current];
                    } else if (ghostNodes.count(current) > 0) {
                        neighbors = ghostAdjacencyList[current];
                    }
                    
                    for (const auto& edge : neighbors) {
                        int neighbor = edge.first;
                        int edgeWeight = edge.second;
                        
                        // Skip if neighbor is not local or ghost
                        if (!isLocal(neighbor) && ghostNodes.count(neighbor) == 0) continue;
                        
                        // If we can improve the path to neighbor
                        if (localDistances[neighbor] > localDistances[current] + edgeWeight) {
                            localDistances[neighbor] = localDistances[current] + edgeWeight;
                            localPredecessors[neighbor] = current;
                            affectedLocally = true;
                            
                            // Add to affected nodes and queue
                            if (affectedNodes.find(neighbor) == affectedNodes.end()) {
                                affectedNodes.insert(neighbor);
                                queue.push(neighbor);
                            }
                        }
                    }
                }
                
                // Synchronize updated distances
                if (affectedLocally) {
                    synchronizeGhostNodes();
                    broadcastSSSP();
                }
            }
        }
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Edge weight change: (" << u << " -> " << v 
                      << ") from " << oldWeight << " to " << newWeight << std::endl;
        }
    }
    
    // Get shortest path from source to target
    std::vector<int> getShortestPath(int source, int target) {
        std::vector<int> path;
        
        // Check if target is unreachable
        if (globalDistances[target] == std::numeric_limits<int>::max()) {
            return path; // Empty path means unreachable
        }
        
        // Build path by following predecessors
        int current = target;
        while (current != source) {
            path.push_back(current);
            current = globalPredecessors[current];
            
            // Safety check for cycles (shouldn't happen with correct algorithm)
            if (current == -1) {
                path.clear();
                return path; // Error in path
            }
        }
        path.push_back(source);
        
        // Reverse to get path from source to target
        std::reverse(path.begin(), path.end());
        
        return path;
    }
    
    // Output performance metrics
    void reportPerformanceMetrics() {
        double totalTime = commTime + computeTime;
        
        // Gather metrics from all processes
        double maxCommTime, maxComputeTime, maxTotalTime;
        int totalMessagesSent, totalMessagesReceived;
        long long totalDataSent;
        
        MPI_Reduce(&commTime, &maxCommTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&computeTime, &maxComputeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&totalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        MPI_Reduce(&messagesSent, &totalMessagesSent, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&messagesReceived, &totalMessagesReceived, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&dataSent, &totalDataSent, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "\n--- Performance Metrics ---" << std::endl;
            std::cout << "Max Communication Time: " << maxCommTime << " seconds" << std::endl;
            std::cout << "Max Computation Time: " << maxComputeTime << " seconds" << std::endl;
            std::cout << "Max Total Time: " << maxTotalTime << " seconds" << std::endl;
            std::cout << "Total Messages Sent: " << totalMessagesSent << std::endl;
            std::cout << "Total Messages Received: " << totalMessagesReceived << std::endl;
            std::cout << "Total Data Sent: " << (double)totalDataSent / (1024*1024) << " MB" << std::endl;
            std::cout << "----------------------------" << std::endl;
        }
    }
    
    // Print the shortest paths from source to all vertices
    void printAllShortestPaths(int source) {
        if (rank == 0) {
            std::cout << "\n--- Shortest Paths from Source " << source << " ---" << std::endl;
            for (int i = 0; i < numVertices; i++) {
                std::cout << "To " << i << ": ";
                
                if (globalDistances[i] == std::numeric_limits<int>::max()) {
                    std::cout << "Unreachable" << std::endl;
                } else {
                    std::cout << "Distance = " << globalDistances[i] << ", Path = ";
                    std::vector<int> path = getShortestPath(source, i);
                    for (size_t j = 0; j < path.size(); j++) {
                        std::cout << path[j];
                        if (j < path.size() - 1) std::cout << " -> ";
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << "----------------------------------------" << std::endl;
        }
    }
};

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    Timer totalTimer;
    
    // Check command line arguments
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex> [<operation_file>]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    const std::string graphFile = argv[1];
    const int sourceVertex = std::stoi(argv[2]);
    
    DistributedGraph graph(rank, numProcesses);
    
    if (rank == 0) {
        std::cout << "Loading and partitioning graph from " << graphFile << std::endl;
    }
    
    // Load and partition the graph
    graph.loadAndPartitionGraph(graphFile);
    
    if (rank == 0) {
        std::cout << "Computing initial SSSP from source vertex " << sourceVertex << std::endl;
    }
    
    // Compute initial SSSP
    graph.computeInitialSSSP(sourceVertex);
    
    // Print all shortest paths
    graph.printAllShortestPaths(sourceVertex);
    
    // Process updates from file if provided
    if (argc >= 4) {
        const std::string operationFile = argv[3];
        
        if (rank == 0) {
            std::cout << "Processing graph updates from " << operationFile << std::endl;
            
            std::ifstream file(operationFile);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open operation file " << operationFile << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            std::string operation;
            int u, v, weight, oldWeight;
            
            while (file >> operation) {
                if (operation == "INSERT") {
                    file >> u >> v >> weight;
                    // Broadcast operation to all processes
                    int opCode = 1; // 1 for INSERT
                    MPI_Bcast(&opCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&weight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeInsertion(sourceVertex, u, v, weight);
                }
                else if (operation == "DELETE") {
                    file >> u >> v;
                    // Broadcast operation to all processes
                    int opCode = 2; // 2 for DELETE
                    MPI_Bcast(&opCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeDeletion(sourceVertex, u, v);
                }
                else if (operation == "CHANGE") {
                    file >> u >> v >> oldWeight >> weight;
                    // Broadcast operation to all processes
                    int opCode = 3; // 3 for CHANGE
                    MPI_Bcast(&opCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&oldWeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&weight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeWeightChange(sourceVertex, u, v, oldWeight, weight);
                }
                else {
                    std::cerr << "Unknown operation: " << operation << std::endl;
                }
            }
            
            file.close();
            
            // Signal end of operations
            int opCode = 0;
            MPI_Bcast(&opCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            // Other processes receive and perform operations
            while (true) {
                int opCode, u, v, weight, oldWeight;
                MPI_Bcast(&opCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
                
                if (opCode == 0) break; // End of operations
                
                if (opCode == 1) { // INSERT
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&weight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeInsertion(sourceVertex, u, v, weight);
                }
                else if (opCode == 2) { // DELETE
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeDeletion(sourceVertex, u, v);
                }
                else if (opCode == 3) { // CHANGE
                    MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&oldWeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&weight, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    
                    graph.updateAfterEdgeWeightChange(sourceVertex, u, v, oldWeight, weight);
                }
            }
        }
        
        // Print updated shortest paths
        graph.printAllShortestPaths(sourceVertex);
    }
    
    // Report performance metrics
    graph.reportPerformanceMetrics();
    
    double totalTime = totalTimer.elapsed();
    double maxTotalTime;
    MPI_Reduce(&totalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Total execution time: " << maxTotalTime << " seconds" << std::endl;
    }
    
    MPI_Finalize();
    logPerformance("MPI", "GraphDataset", maxTotalTime);
    
    system("python3 Analysis.py");
    return 0;
}