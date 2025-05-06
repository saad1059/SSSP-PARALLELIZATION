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
#include <omp.h>
#include <metis.h>

// Represents a weighted edge in the graph
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
    int rank;            // MPI process rank
    int numProcesses;    // Total number of MPI processes
    int numThreads;      // Number of OpenMP threads per process
    int numVertices;     // Total number of vertices in the graph
    int localVerticesStart; // Starting index of local vertices
    int localVerticesEnd;   // Ending index of local vertices
    std::vector<int> partition;  // Maps vertex id to partition id
    
    // Local graph representation
    std::vector<std::vector<std::pair<int, int>>> localAdjacencyList; // Local portion of adjacency list
    std::vector<std::vector<std::pair<int, int>>> ghostAdjacencyList; // Ghost nodes adjacency list
    
    // Thread-local storage for OpenMP
    std::vector<std::vector<int>> threadLocalDistances;       // Thread-local distances for parallelization
    std::vector<std::vector<int>> threadLocalPredecessors;    // Thread-local predecessors
    
    // SSSP data
    std::vector<int> localDistances;       // Distances for local vertices
    std::vector<int> globalDistances;      // Global distances (for collective operations)
    std::vector<int> localPredecessors;    // Predecessor for local vertices
    std::vector<int> globalPredecessors;   // Global predecessors (for collective operations)
    
    // Ghost node management
    std::unordered_set<int> ghostNodes;    // Set of ghost nodes
    std::unordered_map<int, std::vector<int>> ghostNodeOwners; // Maps ghost node to its owners
    
    // Profiling info
    double commTime;       // Time spent in communication
    double computeTime;    // Time spent in computation
    double ompOverhead;    // Time spent in OpenMP overhead
    int messagesSent;      // Number of messages sent
    int messagesReceived;  // Number of messages received
    long long dataSent;    // Amount of data sent (bytes)
    
    // Mutex for thread synchronization
    #pragma omp threadprivate(threadLocalDistances, threadLocalPredecessors)
    std::vector<omp_lock_t> vertexLocks;  // Locks for each vertex to prevent race conditions
    
    // Helper function to check if a vertex is local
    bool isLocal(int vertex) {
        return vertex >= localVerticesStart && vertex < localVerticesEnd;
    }
    
    // Initialize local and ghost adjacency lists based on partition
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
        
        // Initialize distances and predecessors for all vertices (will be updated later)
        localDistances.resize(numVertices, std::numeric_limits<int>::max());
        localPredecessors.resize(numVertices, -1);
        globalDistances.resize(numVertices, std::numeric_limits<int>::max());
        globalPredecessors.resize(numVertices, -1);
        
        // Initialize locks for thread synchronization
        vertexLocks.resize(numVertices);
        #pragma omp parallel for
        for (int i = 0; i < numVertices; i++) {
            omp_init_lock(&vertexLocks[i]);
        }
        
        // Initialize thread-local storage
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            threadLocalDistances.resize(numVertices, std::numeric_limits<int>::max());
            threadLocalPredecessors.resize(numVertices, -1);
        }
        
        // Copy local portion of the adjacency list
        #pragma omp parallel for schedule(dynamic)
        for (int i = localVerticesStart; i < localVerticesEnd; i++) {
            localAdjacencyList[i] = fullAdjacencyList[i];
            
            // Identify ghost nodes (nodes connected to local nodes but not local)
            for (const auto& edge : localAdjacencyList[i]) {
                int target = edge.first;
                if (!isLocal(target)) {
                    #pragma omp critical
                    {
                        ghostNodes.insert(target);
                        ghostNodeOwners[target].push_back(getProcessForVertex(target));
                    }
                }
            }
        }
        
        // Initialize ghost adjacency list
        ghostAdjacencyList.resize(numVertices);
        
        // For each ghost node, find its outgoing edges that point to local nodes
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numVertices; i++) {
            if (isLocal(i)) continue; // Skip local vertices
            
            std::vector<std::pair<int, int>> localGhostEdges;
            
            for (const auto& edge : fullAdjacencyList[i]) {
                int target = edge.first;
                if (isLocal(target)) {
                    // This is a ghost node with an edge to a local node
                    localGhostEdges.push_back(edge);
                }
            }
            
            if (!localGhostEdges.empty()) {
                #pragma omp critical
                {
                    ghostAdjacencyList[i] = localGhostEdges;
                    ghostNodes.insert(i);
                }
            }
        }
    }
    
    // Get the process responsible for a vertex
    int getProcessForVertex(int vertex) {
        int verticesPerProcess = numVertices / numProcesses;
        int remainingVertices = numVertices % numProcesses;
        
        // Handle the case where some processes get an extra vertex
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
        
        // Prepare data structures for communication
        std::vector<MPI_Request> requests;
        std::vector<std::vector<int>> sendBuffers(numProcesses);
        std::vector<std::vector<int>> recvBuffers(numProcesses);
        
        // First, determine what data to send to each process
        #pragma omp parallel
        {
            std::vector<std::vector<int>> threadSendBuffers(numProcesses);
            
            #pragma omp for schedule(dynamic)
            for (int v = localVerticesStart; v < localVerticesEnd; v++) {
                // For each process
                for (int p = 0; p < numProcesses; p++) {
                    if (p == rank) continue; // Skip self
                    
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
                        // Add vertex, distance, and predecessor to thread-local send buffer
                        threadSendBuffers[p].push_back(v);
                        threadSendBuffers[p].push_back(localDistances[v]);
                        threadSendBuffers[p].push_back(localPredecessors[v]);
                    }
                }
            }
            
            // Merge thread-local send buffers
            #pragma omp critical
            {
                for (int p = 0; p < numProcesses; p++) {
                    if (!threadSendBuffers[p].empty()) {
                        sendBuffers[p].insert(sendBuffers[p].end(), 
                                              threadSendBuffers[p].begin(), 
                                              threadSendBuffers[p].end());
                    }
                }
            }
        }
        
        // Send buffer sizes to each process
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
            
            // Update ghost node information in parallel
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < recvBuffers[p].size(); i += 3) {
                int vertex = recvBuffers[p][i];
                int distance = recvBuffers[p][i + 1];
                int predecessor = recvBuffers[p][i + 2];
                
                // Update only if it's a better distance
                if (distance < localDistances[vertex]) {
                    omp_set_lock(&vertexLocks[vertex]);
                    if (distance < localDistances[vertex]) {
                        localDistances[vertex] = distance;
                        localPredecessors[vertex] = predecessor;
                    }
                    omp_unset_lock(&vertexLocks[vertex]);
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
        
        // Update local copies with the global values
        #pragma omp parallel for
        for (int i = 0; i < numVertices; i++) {
            localDistances[i] = globalDistances[i];
            localPredecessors[i] = globalPredecessors[i];
        }
        
        commTime += timer.elapsed();
    }
    
public:
    // Constructor
    DistributedGraph(int rank, int numProcesses) 
        : rank(rank), numProcesses(numProcesses), numVertices(0),
          commTime(0), computeTime(0), ompOverhead(0),
          messagesSent(0), messagesReceived(0), dataSent(0) {
        // Set number of threads from environment variable or default
        numThreads = omp_get_max_threads();
    }
    
    // Destructor to clean up resources
    ~DistributedGraph() {
        // Clean up OpenMP locks
        for (int i = 0; i < vertexLocks.size(); i++) {
            omp_destroy_lock(&vertexLocks[i]);
        }
    }
    
    // Set number of OpenMP threads
    void setNumThreads(int threads) {
        numThreads = threads;
        omp_set_num_threads(threads);
    }
    
    // Load graph from file and partition it using METIS
    void loadAndPartitionGraph(const std::string& filename) {
        std::vector<std::vector<std::pair<int, int>>> fullAdjacencyList;
        std::vector<Edge> allEdges;
        
        // Only rank 0 reads the file
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
        
        // Convert adjacency list to CSR format for METIS
        if (rank == 0) {
            // For METIS, we need an undirected graph
            // Create CSR format
            std::vector<idx_t> xadj(numVertices + 1, 0);
            std::vector<idx_t> adjncy;
            std::vector<idx_t> adjwgt;
            
            // Count edges per vertex - parallelize this count
            std::vector<int> edgeCount(numVertices, 0);
            #pragma omp parallel for
            for (int i = 0; i < allEdges.size(); i++) {
                const auto& edge = allEdges[i];
                #pragma omp atomic
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
            
            // This needs to be sequential due to the currentIndex access
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
            #pragma omp parallel for
            for (int i = 0; i < numVertices; i++) {
                partition[i] = partitionIdx[i];
            }
            
            std::cout << "METIS partitioning completed with edge-cut: " << objval << std::endl;
            
            // Print partition distribution
            std::vector<int> partSizes(numProcesses, 0);
            #pragma omp parallel for
            for (int i = 0; i < numVertices; i++) {
                #pragma omp atomic
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
        
        // Send adjacency list data to all processes
        // First, serialize the adjacency list
        std::vector<int> serializedAdj;
        
        if (rank == 0) {
            // Format: [vertex, number of neighbors, (neighbor, weight), ...]
            serializedAdj.reserve(numVertices * 10); // Estimate size to avoid reallocations
            
            for (int i = 0; i < numVertices; i++) {
                serializedAdj.push_back(i);
                serializedAdj.push_back(fullAdjacencyList[i].size());
                
                for (const auto& edge : fullAdjacencyList[i]) {
                    serializedAdj.push_back(edge.first);
                    serializedAdj.push_back(edge.second);
                }
            }
        }
        
        // Broadcast serialized adjacency list size
        int serializedSize = serializedAdj.size();
        MPI_Bcast(&serializedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Resize and broadcast the full data
        if (rank != 0) {
            serializedAdj.resize(serializedSize);
        }
        
        MPI_Bcast(serializedAdj.data(), serializedSize, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Deserialize the adjacency list
        if (rank != 0) {
            int index = 0;
            fullAdjacencyList.resize(numVertices);
            
            #pragma omp parallel
            {
                std::vector<std::vector<std::pair<int, int>>> threadLocalAdjList(numVertices);
                
                #pragma omp for
                for (int idx = 0; idx < serializedSize;) {
                    int vertex = serializedAdj[idx++];
                    int numNeighbors = serializedAdj[idx++];
                    
                    for (int j = 0; j < numNeighbors; j++) {
                        int neighbor = serializedAdj[idx++];
                        int weight = serializedAdj[idx++];
                        threadLocalAdjList[vertex].push_back(std::make_pair(neighbor, weight));
                    }
                }
                
                #pragma omp critical
                {
                    for (int v = 0; v < numVertices; v++) {
                        if (!threadLocalAdjList[v].empty()) {
                            fullAdjacencyList[v].insert(fullAdjacencyList[v].end(),
                                                       threadLocalAdjList[v].begin(),
                                                       threadLocalAdjList[v].end());
                        }
                    }
                }
            }
        }
        
        // Initialize local data structures
        initializeLocalStructures(fullAdjacencyList);
        
        std::cout << "Process " << rank << " initialized with vertices [" 
                  << localVerticesStart << " - " << localVerticesEnd - 1 << "] using "
                  << numThreads << " OpenMP threads" << std::endl;
        std::cout << "Process " << rank << " has " << ghostNodes.size() << " ghost nodes" << std::endl;
    }
    
    // Compute initial SSSP using distributed Dijkstra's algorithm
    void computeInitialSSSP(int source) {
        Timer timer;
        
        // Reset distances and predecessors
        #pragma omp parallel for
        for (int i = 0; i < numVertices; i++) {
            localDistances[i] = std::numeric_limits<int>::max();
            localPredecessors[i] = -1;
        }
        
        // Set source distance to 0
        if (isLocal(source)) {
            localDistances[source] = 0;
        }
        
        // Synchronize initial distances
        broadcastSSSP();
        
        // Each process handles its local portion of the graph
        bool changed = true;
        int iteration = 0;
        
        while (changed) {
            Timer iterTimer;
            changed = false;
            iteration++;
            
            // Using thread-local arrays to avoid contention
            std::vector<bool> threadChanged(numThreads, false);
            
            // Process local vertices with OpenMP
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                
                // Initialize thread-local distances with current values
                for (int i = 0; i < numVertices; i++) {
                    threadLocalDistances[i] = localDistances[i];
                    threadLocalPredecessors[i] = localPredecessors[i];
                }
                
                // Divide local vertices among threads
                #pragma omp for schedule(dynamic, 64)
                for (int u = localVerticesStart; u < localVerticesEnd; u++) {
                    // Skip unreachable vertices
                    if (localDistances[u] == std::numeric_limits<int>::max()) continue;
                    
                    // Explore neighbors
                    for (const auto& edge : localAdjacencyList[u]) {
                        int v = edge.first;
                        int weight = edge.second;
                        
                        // Relaxation step - use thread-local arrays first
                        if (threadLocalDistances[v] > localDistances[u] + weight) {
                            threadLocalDistances[v] = localDistances[u] + weight;
                            threadLocalPredecessors[v] = u;
                            threadChanged[tid] = true;
                        }
                    }
                }
                
                // Process ghost vertices that have edges to local vertices
                #pragma omp for schedule(dynamic, 16)
                for (auto it = ghostNodes.begin(); it != ghostNodes.end(); ++it) {
                    int u = *it;
                    
                    // Skip unreachable vertices
                    if (localDistances[u] == std::numeric_limits<int>::max()) continue;
                    
                    // Explore edges to local vertices
                    for (const auto& edge : ghostAdjacencyList[u]) {
                        int v = edge.first;
                        int weight = edge.second;
                        
                        // Only process if v is local
                        if (isLocal(v) && threadLocalDistances[v] > localDistances[u] + weight) {
                            threadLocalDistances[v] = localDistances[u] + weight;
                            threadLocalPredecessors[v] = u;
                            threadChanged[tid] = true;
                        }
                    }
                }
                
                // Merge thread-local results
                #pragma omp barrier
                
                // Update global arrays with thread-local results
                #pragma omp for schedule(dynamic, 128)
                for (int v = 0; v < numVertices; v++) {
                    for (int t = 0; t < numThreads; t++) {
                        if (threadLocalDistances[t][v] < localDistances[v]) {
                            omp_set_lock(&vertexLocks[v]);
                            if (threadLocalDistances[t][v] < localDistances[v]) {
                                localDistances[v] = threadLocalDistances[t][v];
                                localPredecessors[v] = threadLocalPredecessors[t][v];
                                changed = true;
                            }
                            omp_unset_lock(&vertexLocks[v]);
                        }
                    }
                }
                
                // Check if any thread made changes
                #pragma omp single
                {
                    for (int t = 0; t < numThreads; t++) {
                        if (threadChanged[t]) {
                            changed = true;
                        }
                    }
                }
            }
            
            // Synchronize updated distances across processes
            int localChanged = changed ? 1 : 0;
            int globalChanged;
            
            MPI_Allreduce(&localChanged, &globalChanged, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            changed = (globalChanged != 0);
            
            // Exchange ghost node information
            if (changed) {
                synchronizeGhostNodes();
            }
            
            double iterTime = iterTimer.elapsed();
            if (rank == 0 && iteration % 10 == 0) { // Print progress every 10 iterations
                std::cout << "Iteration " << iteration << " completed in " << iterTime 
                          << " seconds, changed: " << (changed ? "yes" : "no") << std::endl;
            }
        }
        
        // Final synchronization to ensure consistent state
        broadcastSSSP();
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Initial SSSP computation completed in " << iteration 
                      << " iterations, " << timer.elapsed() << " seconds" << std::endl;
        }
    }
    
    // Update SSSP after edge insertion
    void updateAfterEdgeInsertion(int source, int u, int v, int weight) {
        Timer timer;
        
        // First, update the local adjacency list if needed
        if (isLocal(u)) {
            // Check if edge already exists
            bool edgeExists = false;
            #pragma omp parallel for shared(edgeExists)
            for (int i = 0; i < localAdjacencyList[u].size(); i++) {
                auto& edge = localAdjacencyList[u][i];
                if (edge.first == v) {
                    #pragma omp critical
                    {
                        edge.second = weight;
                        edgeExists = true;
                    }
                }
            }
            
            if (!edgeExists) {
                #pragma omp critical
                {
                    localAdjacencyList[u].push_back(std::make_pair(v, weight));
                    
                    // Update ghost nodes if v is not local
                    if (!isLocal(v)) {
                        ghostNodes.insert(v);
                        ghostNodeOwners[v].push_back(getProcessForVertex(v));
                    }
                }
            }
        }
        
        // Synchronize modified adjacency lists
        MPI_Barrier(MPI_COMM_WORLD);
        
        // If u is not local but v is, we need to update ghost adjacency list
        if (!isLocal(u) && isLocal(v)) {
            #pragma omp critical
            {
                bool edgeExists = false;
                for (auto& edge : ghostAdjacencyList[u]) {
                    if (edge.first == v) {
                        edge.second = weight;
                        edgeExists = true;
                        break;
                    }
                }
                
                if (!edgeExists) {
                    ghostAdjacencyList[u].push_back(std::make_pair(v, weight));
                    ghostNodes.insert(u);
                }
            }
        }
        
        // Incremental update of SSSP
        // We only need to consider updates if the new edge provides a shorter path
        
        // Check if the new edge creates a shorter path
        if (localDistances[u] != std::numeric_limits<int>::max() && 
            (localDistances[v] > localDistances[u] + weight)) {
            
            // Update the distance and predecessor for v
            localDistances[v] = localDistances[u] + weight;
            localPredecessors[v] = u;
            
            // Initialize a priority queue for affected vertices
            std::priority_queue<std::pair<int, int>, 
                               std::vector<std::pair<int, int>>, 
                               std::greater<std::pair<int, int>>> pq;
            
            // Add v to the priority queue (distance, vertex)
            pq.push(std::make_pair(localDistances[v], v));
            
            // Process affected vertices
            while (!pq.empty()) {
                int dist = pq.top().first;
                int vertex = pq.top().second;
                pq.pop();
                
                // Skip if this distance is outdated
                if (dist != localDistances[vertex]) continue;
                
                // Process outgoing edges from this vertex
                if (isLocal(vertex)) {
                    for (const auto& edge : localAdjacencyList[vertex]) {
                        int target = edge.first;
                        int edgeWeight = edge.second;
                        
                        // Relaxation step
                        if (localDistances[target] > dist + edgeWeight) {
                            localDistances[target] = dist + edgeWeight;
                            localPredecessors[target] = vertex;
                            pq.push(std::make_pair(localDistances[target], target));
                        }
                    }
                }
                
                // If it's a ghost node, we need to update its neighbors too
                if (ghostAdjacencyList.size() > vertex && !ghostAdjacencyList[vertex].empty()) {
                    for (const auto& edge : ghostAdjacencyList[vertex]) {
                        int target = edge.first;
                        int edgeWeight = edge.second;
                        
                        // Only process local targets
                        if (isLocal(target) && localDistances[target] > dist + edgeWeight) {
                            localDistances[target] = dist + edgeWeight;
                            localPredecessors[target] = vertex;
                            pq.push(std::make_pair(localDistances[target], target));
                        }
                    }
                }
            }
        }
        
        // Synchronize updated SSSP tree across all processes
        synchronizeGhostNodes();
        broadcastSSSP();
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "SSSP updated after edge insertion (" << u << " -> " << v 
                      << ", weight " << weight << ") in " << timer.elapsed() 
                      << " seconds" << std::endl;
        }
    }
    
    // Update SSSP after edge deletion
    void updateAfterEdgeDeletion(int source, int u, int v) {
        Timer timer;
        
        // Remove the edge from the adjacency list
        bool edgeRemoved = false;
        
        if (isLocal(u)) {
            #pragma omp parallel for shared(edgeRemoved)
            for (size_t i = 0; i < localAdjacencyList[u].size(); i++) {
                if (localAdjacencyList[u][i].first == v) {
                    #pragma omp critical
                    {
                        localAdjacencyList[u].erase(localAdjacencyList[u].begin() + i);
                        edgeRemoved = true;
                    }
                    break;
                }
            }
        }
        
        // If u is a ghost node and v is local, remove from ghost adjacency list
        if (!isLocal(u) && isLocal(v)) {
            #pragma omp parallel for shared(edgeRemoved)
            for (size_t i = 0; i < ghostAdjacencyList[u].size(); i++) {
                if (ghostAdjacencyList[u][i].first == v) {
                    #pragma omp critical
                    {
                        ghostAdjacencyList[u].erase(ghostAdjacencyList[u].begin() + i);
                        edgeRemoved = true;
                        
                        // If no more edges to local vertices, remove from ghost nodes
                        if (ghostAdjacencyList[u].empty()) {
                            ghostNodes.erase(u);
                            ghostNodeOwners.erase(u);
                        }
                    }
                    break;
                }
            }
        }
        
        // Synchronize modification
        MPI_Barrier(MPI_COMM_WORLD);
        
        // If the edge wasn't part of the shortest path tree, we're done
        if (localPredecessors[v] != u) {
            if (rank == 0) {
                std::cout << "Edge (" << u << " -> " << v 
                          << ") was not in the shortest path tree, no update needed." << std::endl;
            }
            computeTime += timer.elapsed();
            return;
        }
        
        // The edge was part of the shortest path tree, we need to recompute affected paths
        // Mark all vertices in the subtree rooted at v as affected
        std::vector<bool> affected(numVertices, false);
        std::queue<int> q;
        q.push(v);
        affected[v] = true;
        
        while (!q.empty()) {
            int vertex = q.front();
            q.pop();
            
            // Reset distance and predecessor for affected vertices
            localDistances[vertex] = std::numeric_limits<int>::max();
            localPredecessors[vertex] = -1;
            
            // Find all vertices whose predecessor is the current vertex
            for (int i = 0; i < numVertices; i++) {
                if (localPredecessors[i] == vertex && !affected[i]) {
                    affected[i] = true;
                    q.push(i);
                }
            }
        }
        
        // Synchronize the reset of affected vertices
        broadcastSSSP();
        
        // Incrementally recompute shortest paths for affected vertices
        // We use a similar approach to Dijkstra but only consider affected vertices
        
        // Find potential new parents for affected vertices
        #pragma omp parallel
        {
            // For each affected vertex, check if it has incoming edges from unaffected vertices
            #pragma omp for schedule(dynamic)
            for (int v = 0; v < numVertices; v++) {
                if (!affected[v]) continue; // Skip unaffected vertices
                
                // Check all incoming edges to v
                if (isLocal(v)) {
                    for (int u = 0; u < numVertices; u++) {
                        if (affected[u]) continue; // Skip affected vertices as sources
                        
                        // Check if u has an edge to v
                        if (isLocal(u)) {
                            for (const auto& edge : localAdjacencyList[u]) {
                                if (edge.first == v && localDistances[u] != std::numeric_limits<int>::max()) {
                                    int newDist = localDistances[u] + edge.second;
                                    if (newDist < localDistances[v]) {
                                        omp_set_lock(&vertexLocks[v]);
                                        if (newDist < localDistances[v]) {
                                            localDistances[v] = newDist;
                                            localPredecessors[v] = u;
                                        }
                                        omp_unset_lock(&vertexLocks[v]);
                                    }
                                }
                            }
                        }
                        // Check ghost nodes
                        else if (ghostNodes.count(u) > 0) {
                            for (const auto& edge : ghostAdjacencyList[u]) {
                                if (edge.first == v && localDistances[u] != std::numeric_limits<int>::max()) {
                                    int newDist = localDistances[u] + edge.second;
                                    if (newDist < localDistances[v]) {
                                        omp_set_lock(&vertexLocks[v]);
                                        if (newDist < localDistances[v]) {
                                            localDistances[v] = newDist;
                                            localPredecessors[v] = u;
                                        }
                                        omp_unset_lock(&vertexLocks[v]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Synchronize after initial update
        synchronizeGhostNodes();
        broadcastSSSP();
        
        // Now run a full Dijkstra-like algorithm on the affected vertices
        bool changed = true;
        int iteration = 0;
        
        while (changed) {
            Timer iterTimer;
            changed = false;
            iteration++;
            
            // Using thread-local arrays to avoid contention
            std::vector<bool> threadChanged(numThreads, false);
            
            // Process vertices with OpenMP
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                
                // Initialize thread-local distances with current values
                for (int i = 0; i < numVertices; i++) {
                    threadLocalDistances[i] = localDistances[i];
                    threadLocalPredecessors[i] = localPredecessors[i];
                }
                
                // Process local vertices
                #pragma omp for schedule(dynamic, 64)
                for (int u = localVerticesStart; u < localVerticesEnd; u++) {
                    // Skip unreachable or unaffected vertices
                    if (localDistances[u] == std::numeric_limits<int>::max() || !affected[u]) continue;
                    
                    // Explore neighbors
                    for (const auto& edge : localAdjacencyList[u]) {
                        int v = edge.first;
                        int weight = edge.second;
                        
                        // Relaxation step
                        if (threadLocalDistances[v] > localDistances[u] + weight) {
                            threadLocalDistances[v] = localDistances[u] + weight;
                            threadLocalPredecessors[v] = u;
                            threadChanged[tid] = true;
                            affected[v] = true; // Mark this vertex as affected
                        }
                    }
                }
                
                // Process ghost vertices that might affect local vertices
                #pragma omp for schedule(dynamic, 16)
                for (auto it = ghostNodes.begin(); it != ghostNodes.end(); ++it) {
                    int u = *it;
                    
                    // Skip unreachable or unaffected vertices
                    if (localDistances[u] == std::numeric_limits<int>::max() || !affected[u]) continue;
                    
                    // Explore edges to local vertices
                    for (const auto& edge : ghostAdjacencyList[u]) {
                        int v = edge.first;
                        int weight = edge.second;
                        
                        // Only process if v is local
                        if (isLocal(v) && threadLocalDistances[v] > localDistances[u] + weight) {
                            threadLocalDistances[v] = localDistances[u] + weight;
                            threadLocalPredecessors[v] = u;
                            threadChanged[tid] = true;
                            affected[v] = true; // Mark this vertex as affected
                        }
                    }
                }
                
                // Merge thread-local results
                #pragma omp barrier
                
                // Update global arrays with thread-local results
                #pragma omp for schedule(dynamic, 128)
                for (int v = 0; v < numVertices; v++) {
                    for (int t = 0; t < numThreads; t++) {
                        if (threadLocalDistances[t][v] < localDistances[v]) {
                            omp_set_lock(&vertexLocks[v]);
                            if (threadLocalDistances[t][v] < localDistances[v]) {
                                localDistances[v] = threadLocalDistances[t][v];
                                localPredecessors[v] = threadLocalPredecessors[t][v];
                                changed = true;
                            }
                            omp_unset_lock(&vertexLocks[v]);
                        }
                    }
                }
                
                // Check if any thread made changes
                #pragma omp single
                {
                    for (int t = 0; t < numThreads; t++) {
                        if (threadChanged[t]) {
                            changed = true;
                        }
                    }
                }
            }
            
            // Synchronize updated distances across processes
            int localChanged = changed ? 1 : 0;
            int globalChanged;
            
            MPI_Allreduce(&localChanged, &globalChanged, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            changed = (globalChanged != 0);
            
            // Exchange ghost node information
            if (changed) {
                synchronizeGhostNodes();
            }
            
            double iterTime = iterTimer.elapsed();
            if (rank == 0 && iteration % 10 == 0) { // Print progress every 10 iterations
                std::cout << "Edge deletion update - Iteration " << iteration << " completed in " 
                          << iterTime << " seconds, changed: " << (changed ? "yes" : "no") << std::endl;
            }
        }
        
        // Final synchronization to ensure consistent state
        broadcastSSSP();
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "SSSP updated after edge deletion (" << u << " -> " << v 
                      << ") in " << timer.elapsed() << " seconds, " << iteration 
                      << " iterations" << std::endl;
        }
    }
    
    // Update SSSP after edge weight change
    void updateAfterEdgeWeightChange(int source, int u, int v, int newWeight) {
        Timer timer;
        
        // Find current weight of the edge
        int oldWeight = -1;
        
        if (isLocal(u)) {
            for (auto& edge : localAdjacencyList[u]) {
                if (edge.first == v) {
                    oldWeight = edge.second;
                    edge.second = newWeight; // Update the weight
                    break;
                }
            }
        }
        
        // If u is a ghost node and v is local, update in ghost adjacency list
        if (!isLocal(u) && isLocal(v)) {
            for (auto& edge : ghostAdjacencyList[u]) {
                if (edge.first == v) {
                    oldWeight = edge.second;
                    edge.second = newWeight; // Update the weight
                    break;
                }
            }
        }
        
        // If edge not found, it's an error
        if (oldWeight == -1) {
            if (rank == 0) {
                std::cerr << "Error: Edge (" << u << " -> " << v << ") not found for weight update" << std::endl;
            }
            return;
        }
        
        // Synchronize modification
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Check if this edge is part of the shortest path tree
        if (localPredecessors[v] == u) {
            if (newWeight < oldWeight) {
                // Weight decreased, this might improve paths through v
                updateAfterEdgeInsertion(source, u, v, newWeight);
            } else {
                // Weight increased, need to find alternative paths
                updateAfterEdgeDeletion(source, u, v);
                // Then add the edge back with new weight
                updateAfterEdgeInsertion(source, u, v, newWeight);
            }
        } else if (newWeight < oldWeight) {
            // Edge not in shortest path tree, but weight decreased
            // Check if it provides a better path
            if (localDistances[u] != std::numeric_limits<int>::max() && 
                (localDistances[v] > localDistances[u] + newWeight)) {
                updateAfterEdgeInsertion(source, u, v, newWeight);
            }
        }
        // If weight increased and edge not in shortest path tree, no update needed
        
        computeTime += timer.elapsed();
        
        if (rank == 0) {
            std::cout << "SSSP updated after edge weight change (" << u << " -> " << v 
                      << ", " << oldWeight << " -> " << newWeight << ") in " 
                      << timer.elapsed() << " seconds" << std::endl;
        }
    }
    
    // Get local shortest path distance
    int getDistance(int vertex) {
        if (localDistances[vertex] == std::numeric_limits<int>::max()) {
            return -1; // Indicate unreachable
        }
        return localDistances[vertex];
    }
    
    // Get predecessor in shortest path
    int getPredecessor(int vertex) {
        return localPredecessors[vertex];
    }
    
    // Get full shortest path from source to target
    std::vector<int> getShortestPath(int source, int target) {
        std::vector<int> path;
        
        // Check if target is reachable
        if (localDistances[target] == std::numeric_limits<int>::max()) {
            return path; // Empty path indicates unreachable
        }
        
        // Build path from target back to source
        int current = target;
        while (current != source) {
            path.push_back(current);
            current = localPredecessors[current];
            
            // Check for invalid path
            if (current == -1) {
                return std::vector<int>(); // Empty path indicates error
            }
        }
        path.push_back(source);
        
        // Reverse to get path from source to target
        std::reverse(path.begin(), path.end());
        
        return path;
    }
    
    // Print profiling information
    void printProfilingInfo() {
        // Gather profiling data from all processes
        std::vector<double> allCommTimes(numProcesses);
        std::vector<double> allComputeTimes(numProcesses);
        std::vector<int> allMessagesSent(numProcesses);
        std::vector<int> allMessagesReceived(numProcesses);
        std::vector<long long> allDataSent(numProcesses);
        
        MPI_Gather(&commTime, 1, MPI_DOUBLE, allCommTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&computeTime, 1, MPI_DOUBLE, allComputeTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&messagesSent, 1, MPI_INT, allMessagesSent.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&messagesReceived, 1, MPI_INT, allMessagesReceived.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&dataSent, 1, MPI_LONG_LONG, allDataSent.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double totalCommTime = 0, maxCommTime = 0, minCommTime = allCommTimes[0];
            double totalComputeTime = 0, maxComputeTime = 0, minComputeTime = allComputeTimes[0];
            int totalMessagesSent = 0, totalMessagesReceived = 0;
            long long totalDataSent = 0;
            
            for (int i = 0; i < numProcesses; i++) {
                totalCommTime += allCommTimes[i];
                maxCommTime = std::max(maxCommTime, allCommTimes[i]);
                minCommTime = std::min(minCommTime, allCommTimes[i]);
                
                totalComputeTime += allComputeTimes[i];
                maxComputeTime = std::max(maxComputeTime, allComputeTimes[i]);
                minComputeTime = std::min(minComputeTime, allComputeTimes[i]);
                
                totalMessagesSent += allMessagesSent[i];
                totalMessagesReceived += allMessagesReceived[i];
                totalDataSent += allDataSent[i];
            }
            
            std::cout << "\n===== Profiling Information =====" << std::endl;
            std::cout << "Number of processes: " << numProcesses << std::endl;
            std::cout << "Threads per process: " << numThreads << std::endl;
            std::cout << "Total vertices: " << numVertices << std::endl;
            
            std::cout << "\nCommunication Time:" << std::endl;
            std::cout << "  Total: " << totalCommTime << " seconds" << std::endl;
            std::cout << "  Average: " << totalCommTime / numProcesses << " seconds" << std::endl;
            std::cout << "  Min: " << minCommTime << " seconds" << std::endl;
            std::cout << "  Max: " << maxCommTime << " seconds" << std::endl;
            
            std::cout << "\nComputation Time:" << std::endl;
            std::cout << "  Total: " << totalComputeTime << " seconds" << std::endl;
            std::cout << "  Average: " << totalComputeTime / numProcesses << " seconds" << std::endl;
            std::cout << "  Min: " << minComputeTime << " seconds" << std::endl;
            std::cout << "  Max: " << maxComputeTime << " seconds" << std::endl;
            
            std::cout << "\nMessages:" << std::endl;
            std::cout << "  Total sent: " << totalMessagesSent << std::endl;
            std::cout << "  Total received: " << totalMessagesReceived << std::endl;
            std::cout << "  Total data sent: " << (double)totalDataSent / (1024 * 1024) << " MB" << std::endl;
            
            std::cout << "\nEfficiency:" << std::endl;
            double parallelEfficiency = totalComputeTime / (totalCommTime + totalComputeTime);
            std::cout << "  Parallel efficiency: " << parallelEfficiency * 100 << "%" << std::endl;
            std::cout << "  Communication overhead: " << (1 - parallelEfficiency) * 100 << "%" << std::endl;
            
            std::cout << "===============================" << std::endl;
        }
    }
    
    // Verify correctness of distributed SSSP
    bool verifySSSP(int source) {
        bool isCorrect = true;
        
        // Gather all distances and predecessors to rank 0
        std::vector<int> allDistances;
        std::vector<int> allPredecessors;
        
        if (rank == 0) {
            allDistances = globalDistances;
            allPredecessors = globalPredecessors;
        }
        
        // Check basic properties
        if (rank == 0) {
            // Source distance should be 0
            if (allDistances[source] != 0) {
                std::cerr << "Error: Source distance is not 0" << std::endl;
                isCorrect = false;
            }
            
            // Source predecessor should be -1
            if (allPredecessors[source] != -1) {
                std::cerr << "Error: Source predecessor is not -1" << std::endl;
                isCorrect = false;
            }
            
            // Verify triangle inequality
            for (int u = 0; u < numVertices; u++) {
                if (allDistances[u] == std::numeric_limits<int>::max()) continue;
                
                for (const auto& edge : localAdjacencyList[u]) {
                    int v = edge.first;
                    int w = edge.second;
                    
                    if (allDistances[v] != std::numeric_limits<int>::max() && 
                        allDistances[v] > allDistances[u] + w) {
                        std::cerr << "Error: Triangle inequality violated for edge (" 
                                  << u << " -> " << v << ")" << std::endl;
                        isCorrect = false;
                    }
                }
            }
        }
        
        // Broadcast result
        MPI_Bcast(&isCorrect, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        return isCorrect;
    }
};

// Main function
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    // Set OpenMP threads from command line or default
    int numThreads = omp_get_max_threads();
    if (argc > 3) {
        numThreads = std::atoi(argv[3]);
        omp_set_num_threads(numThreads);
    }
    
    // Process command line arguments
    if (argc < 3 && rank == 0) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex> [num_threads]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::string graphFile = argv[1];
    int sourceVertex = std::atoi(argv[2]);
    
    Timer totalTimer;
    
    // Create distributed graph
    DistributedGraph graph(rank, numProcesses);
    graph.setNumThreads(numThreads);
    
    // Load and partition graph
    if (rank == 0) {
        std::cout << "Loading graph from " << graphFile << "..." << std::endl;
    }
    
    graph.loadAndPartitionGraph(graphFile);
    
    // Compute initial SSSP
    if (rank == 0) {
        std::cout << "Computing initial SSSP from source vertex " << sourceVertex << "..." << std::endl;
    }
    
    graph.computeInitialSSSP(sourceVertex);
    
    // Verify correctness
    if (graph.verifySSSP(sourceVertex)) {
        if (rank == 0) {
            std::cout << "SSSP computation is correct" << std::endl;
        }
    } else {
        if (rank == 0) {
            std::cerr << "ERROR: SSSP computation is incorrect!" << std::endl;
        }
    }
    
    // Example of dynamic updates
    if (rank == 0) {
        std::cout << "\nTesting dynamic updates..." << std::endl;
    }
    
    // Edge insertion
    int u = sourceVertex;
    int v = (sourceVertex + 1) % graph.numVertices;
    int weight = 5;
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nInserting edge (" << u << " -> " << v << ") with weight " << weight << std::endl;
    }
    graph.updateAfterEdgeInsertion(sourceVertex, u, v, weight);
    
    // Edge weight change
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nChanging weight of edge (" << u << " -> " << v << ") to " << weight * 2 << std::endl;
    }
    graph.updateAfterEdgeWeightChange(sourceVertex, u, v, weight * 2);
    
    // Edge deletion
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nDeleting edge (" << u << " -> " << v << ")" << std::endl;
    }
    graph.updateAfterEdgeDeletion(sourceVertex, u, v);
    
    // Print profiling information
    MPI_Barrier(MPI_COMM_WORLD);
    double totalTime = totalTimer.elapsed();
    
    if (rank == 0) {
        std::cout << "\nTotal execution time: " << totalTime << " seconds" << std::endl;
    }
    
    graph.printProfilingInfo();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

/*
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include <atomic>

using namespace std;

const int INF = numeric_limits<int>::max();

// Simplified edge structure to improve cache locality
struct Edge {
    int to, weight;
    Edge(int t, int w = 1) : to(t), weight(w) {}
};

class Graph {
public:
    int n; // number of nodes
    vector<vector<Edge>> adjList;
    vector<int> dist, parent;
    vector<bool> affected; // Track affected vertices for dynamic updates

    Graph(int nodes) : n(nodes) {
        adjList.resize(n);
        dist.resize(n, INF);
        parent.resize(n, -1);
        affected.resize(n, false);
    }

    void addEdge(int u, int v, int w = 1) {
        adjList[u].emplace_back(v, w);
        adjList[v].emplace_back(u, w); // undirected
    }

    void removeEdge(int u, int v) {
        // Mark vertices as affected by the deletion
        if (parent[v] == u) {
            affected[v] = true;
        } else if (parent[u] == v) {
            affected[u] = true;
        }
        
        // Remove edge from adjacency lists
        adjList[u].erase(remove_if(adjList[u].begin(), adjList[u].end(),
                                   [v](const Edge &e) { return e.to == v; }),
                         adjList[u].end());

        adjList[v].erase(remove_if(adjList[v].begin(), adjList[v].end(),
                                   [u](const Edge &e) { return e.to == u; }),
                         adjList[v].end());
    }
    
    // Method to process edge deletion updates asynchronously (based on Algorithm 4)
    void processEdgeDeletionUpdates(int levelOfAsynchrony = 2) {
        bool change = true;
        
        while (change) {
            change = false;
            
            #pragma omp parallel
            {
                vector<int> localAffected;
                
                #pragma omp for schedule(dynamic, 64)
                for (int v = 0; v < n; v++) {
                    if (affected[v]) {
                        // Initialize queue for BFS
                        queue<int> Q;
                        Q.push(v);
                        
                        int level = 0;
                        
                        while (!Q.empty()) {
                            int x = Q.front();
                            Q.pop();
                            
                            // Process children in the shortest path tree
                            for (int c = 0; c < n; c++) {
                                if (parent[c] == x) {
                                    // Mark child as affected
                                    #pragma omp critical
                                    {
                                        if (!affected[c]) {
                                            affected[c] = true;
                                            localAffected.push_back(c);
                                            change = true;
                                        }
                                    }
                                    
                                    // Set distance to infinity
                                    dist[c] = INF;
                                    
                                    // Process next level asynchronously if within limit
                                    level++;
                                    if (level <= levelOfAsynchrony) {
                                        Q.push(c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sequential Dijkstra implementation for comparison
    void dijkstra_sequential(int src) {
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);
        dist[src] = 0;
        
        vector<bool> processed(n, false);
        
        for (int i = 0; i < n; i++) {
            // Find minimum distance vertex
            int u = -1;
            int minDist = INF;
            for (int j = 0; j < n; j++) {
                if (!processed[j] && dist[j] < minDist) {
                    minDist = dist[j];
                    u = j;
                }
            }
            
            if (u == -1) break;  // No reachable nodes left
            processed[u] = true;
            
            // Process all neighbors
            for (const Edge& edge : adjList[u]) {
                int v = edge.to;
                int weight = edge.weight;
                
                if (dist[u] != INF && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    parent[v] = u;
                }
            }
        }
    }

    // Optimized OpenMP implementation
    void dijkstra_openmp_optimized(int src) {
        const int CHUNK_SIZE = 1024; // Tune this based on your graph
        
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);
        dist[src] = 0;
        
        vector<bool> processed(n, false);
        
        // Use an array-based approach for larger graphs
        for (int iter = 0; iter < n; iter++) {
            // Find minimum distance vertex (sequential, as it's typically fast)
            int u = -1;
            int minDist = INF;
            
            for (int j = 0; j < n; j++) {
                if (!processed[j] && dist[j] < minDist) {
                    minDist = dist[j];
                    u = j;
                }
            }
            
            if (u == -1) break;  // No reachable nodes left
            processed[u] = true;
            
            // Parallelize the edge processing with better chunking
            if (adjList[u].size() > CHUNK_SIZE) {
                #pragma omp parallel
                {
                    // Use thread-local variables to avoid false sharing
                    vector<pair<int, int>> localUpdates;
                    
                    #pragma omp for schedule(dynamic, 64)
                    for (size_t j = 0; j < adjList[u].size(); j++) {
                        const Edge& edge = adjList[u][j];
                        int v = edge.to;
                        int weight = edge.weight;
                        
                        if (dist[u] != INF && dist[u] + weight < dist[v]) {
                            localUpdates.emplace_back(v, dist[u] + weight);
                        }
                    }
                    
                    // Apply updates with minimal critical section
                    #pragma omp critical
                    {
                        for (const auto& update : localUpdates) {
                            int v = update.first;
                            int newDist = update.second;
                            if (newDist < dist[v]) {
                                dist[v] = newDist;
                                parent[v] = u;
                            }
                        }
                    }
                }
            } else {
                // Process sequentially for small adjacency lists
                for (const Edge& edge : adjList[u]) {
                    int v = edge.to;
                    int weight = edge.weight;
                    
                    if (dist[u] != INF && dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        parent[v] = u;
                    }
                }
            }
        }
    }

    // Delta-stepping algorithm - more efficient for large graphs
    void delta_stepping(int src, int delta = 3) {
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);
        dist[src] = 0;
        
        // Buckets to store vertices based on distance
        vector<vector<int>> buckets;
        buckets.resize(n); // We'll resize as needed
        buckets[0].push_back(src);
        
        int currentBucket = 0;
        
        while (true) {
            // Find the first non-empty bucket
            while (currentBucket < buckets.size() && buckets[currentBucket].empty()) {
                currentBucket++;
            }
            
            if (currentBucket >= buckets.size()) break;
            
            // Process all vertices in the current bucket in parallel batches
            vector<int> current = std::move(buckets[currentBucket]);
            
            while (!current.empty()) {
                vector<tuple<int, int, int>> updates; // (node, dist, parent)
                
                #pragma omp parallel
                {
                    vector<tuple<int, int, int>> localUpdates;
                    
                    #pragma omp for schedule(dynamic, 256)
                    for (size_t i = 0; i < current.size(); i++) {
                        int u = current[i];
                        
                        // Process light edges (weight <= delta)
                        for (const Edge& edge : adjList[u]) {
                            int v = edge.to;
                            int weight = edge.weight;
                            
                            if (weight <= delta) {
                                int newDist = dist[u] + weight;
                                if (newDist < dist[v]) {
                                    localUpdates.emplace_back(v, newDist, u);
                                }
                            }
                        }
                    }
                    
                    #pragma omp critical
                    {
                        updates.insert(updates.end(), localUpdates.begin(), localUpdates.end());
                    }
                }
                
                // Apply updates sequentially to avoid race conditions
                vector<int> newVertices;
                for (const auto& [v, newDist, p] : updates) {
                    if (newDist < dist[v]) {
                        if (dist[v] == INF) {
                            // New vertex discovered
                            newVertices.push_back(v);
                        } else {
                            // Remove from its current bucket if it exists
                            int oldBucket = dist[v] / delta;
                            if (oldBucket < buckets.size()) {
                                auto& bucket = buckets[oldBucket];
                                bucket.erase(remove(bucket.begin(), bucket.end(), v), bucket.end());
                            }
                        }
                        
                        // Update distance and parent
                        dist[v] = newDist;
                        parent[v] = p;
                        
                        // Add to appropriate bucket
                        int newBucket = newDist / delta;
                        if (newBucket >= buckets.size()) {
                            buckets.resize(newBucket + 1);
                        }
                        buckets[newBucket].push_back(v);
                    }
                }
                
                // Process heavy edges for the new vertices
                updates.clear();
                
                #pragma omp parallel
                {
                    vector<tuple<int, int, int>> localUpdates;
                    
                    #pragma omp for schedule(dynamic, 256)
                    for (size_t i = 0; i < newVertices.size(); i++) {
                        int u = newVertices[i];
                        
                        // Process heavy edges (weight > delta)
                        for (const Edge& edge : adjList[u]) {
                            int v = edge.to;
                            int weight = edge.weight;
                            
                            if (weight > delta) {
                                int newDist = dist[u] + weight;
                                if (newDist < dist[v]) {
                                    localUpdates.emplace_back(v, newDist, u);
                                }
                            }
                        }
                    }
                    
                    #pragma omp critical
                    {
                        updates.insert(updates.end(), localUpdates.begin(), localUpdates.end());
                    }
                }
                
                // Apply heavy edge updates
                for (const auto& [v, newDist, p] : updates) {
                    if (newDist < dist[v]) {
                        if (dist[v] == INF) {
                            // New vertex discovered
                            current.push_back(v);
                        } else {
                            // Remove from its current bucket if it exists
                            int oldBucket = dist[v] / delta;
                            if (oldBucket < buckets.size()) {
                                auto& bucket = buckets[oldBucket];
                                bucket.erase(remove(bucket.begin(), bucket.end(), v), bucket.end());
                            }
                        }
                        
                        // Update distance and parent
                        dist[v] = newDist;
                        parent[v] = p;
                        
                        // Add to appropriate bucket
                        int newBucket = newDist / delta;
                        if (newBucket >= buckets.size()) {
                            buckets.resize(newBucket + 1);
                        }
                        buckets[newBucket].push_back(v);
                    }
                }
                
                // Clear current vertices since we've processed them
                current.clear();
                
                // Get next batch from the current bucket
                if (!buckets[currentBucket].empty()) {
                    current = std::move(buckets[currentBucket]);
                }
            }
        }
    }

    void saveOutputToFile(const string& filename) {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error opening output file!" << endl;
            return;
        }

        for (int i = 0; i < n; ++i) {
            outFile << "Vertex " << i << ": ";
            if (dist[i] == INF)
                outFile << "Unreachable";
            else
                outFile << "Distance = " << dist[i] << ", Parent = " << parent[i];
            outFile << endl;
        }

        outFile.close();
    }

    void printShortestPaths(int limit = 10) {
        for (int i = 0; i < min(limit, n); ++i) {
            cout << "Vertex " << i << ": ";
            if (dist[i] == INF)
                cout << "Unreachable";
            else
                cout << "Distance = " << dist[i] << ", Parent = " << parent[i];
            cout << endl;
        }
    }
};

int getMaxNodeInFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening input file: " << filename << endl;
        return -1;
    }
    
    string line;
    int maxNode = 0;

    while (getline(file, line)) {
        istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            maxNode = max(maxNode, max(u, v));
        }
    }
    
    file.close();
    return maxNode;
}

// Function to load graph efficiently
bool loadGraphEfficiently(const string& filename, Graph& g) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening input file: " << filename << endl;
        return false;
    }
    
    // Pre-allocate edge vectors based on estimated average degree
    int avgDegree = 5; // Adjust based on your graph characteristics
    for (int i = 0; i < g.n; i++) {
        g.adjList[i].reserve(avgDegree);
    }
    
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            g.addEdge(u, v, 1);
        }
    }
    file.close();
    return true;
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Parse command line arguments
    string filename = "road.txt";
    bool useMPI = (numProcesses > 1);
    int algorithm = 0; // 0=sequential, 1=openmp, 2=delta-stepping
    int asynchronyLevel = 2; // Default level of asynchrony for edge deletions
    vector<pair<int, int>> edgesToRemove; // Edges to remove for dynamic updates
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "-a" && i + 1 < argc) {
            algorithm = atoi(argv[++i]);
        } else if (arg == "-async" && i + 1 < argc) {
            asynchronyLevel = atoi(argv[++i]);
        } else if (arg == "-del" && i + 2 < argc) {
            // Format: -del u v (to remove edge u-v)
            int u = atoi(argv[++i]);
            int v = atoi(argv[++i]);
            edgesToRemove.emplace_back(u, v);
        }
    }
    
    // Autodetect optimal thread count if not specified
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    
    if (rank == 0) {
        cout << "Using " << max_threads << " OpenMP threads" << endl;
        cout << "Using algorithm: " << 
            (algorithm == 0 ? "Sequential" : 
             algorithm == 1 ? "OpenMP Optimized" : "Delta-Stepping") << endl;
    }

    int maxNode = 0;
    
    // Only rank 0 reads the file and determines maxNode
    if (rank == 0) {
        maxNode = getMaxNodeInFile(filename);
        if (maxNode < 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            return 1;
        }
        cout << "Max node ID: " << maxNode << " (Total nodes: " << (maxNode+1) << ")" << endl;
    }
    
    // Broadcast maxNode to all processes
    MPI_Bcast(&maxNode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    Graph g(maxNode + 1); // nodes are assumed to be 0-indexed

    // Load graph data
    if (rank == 0) {
        double start_time = MPI_Wtime();
        
        if (!loadGraphEfficiently(filename, g)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            return 1;
        }
        
        // Process any edge deletions requested via command line
        if (!edgesToRemove.empty()) {
            cout << "Processing " << edgesToRemove.size() << " edge deletions..." << endl;
            
            // First run a baseline SSSP to establish the tree
            g.dijkstra_sequential(0);
            
            // Now remove edges
            for (const auto& edge : edgesToRemove) {
                cout << "Removing edge: " << edge.first << " - " << edge.second << endl;
                g.removeEdge(edge.first, edge.second);
            }
            
            // Process deletion updates asynchronously according to Algorithm 4
            double update_start = MPI_Wtime();
            g.processEdgeDeletionUpdates(asynchronyLevel);
            double update_end = MPI_Wtime();
            cout << "Time to process edge deletions: " << (update_end - update_start) << " seconds" << endl;
        }
        
        double end_time = MPI_Wtime();
        cout << "Time to read graph and process initial updates: " << (end_time - start_time) << " seconds" << endl;
    }
    
    // For large graphs and multiple MPI processes, we need an efficient way to share the graph
    if (useMPI) {
        // This is a simplified version - for very large graphs you'd want to use a more
        // sophisticated approach like distributing the graph by partitioning
        
        // Count total edges in each node's adjacency list
        vector<int> adjListSizes(g.n);
        if (rank == 0) {
            for (int i = 0; i < g.n; i++) {
                adjListSizes[i] = g.adjList[i].size();
            }
        }
        
        // Broadcast adjacency list sizes
        MPI_Bcast(adjListSizes.data(), g.n, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Each process allocates memory for adjacency lists
        if (rank != 0) {
            for (int i = 0; i < g.n; i++) {
                g.adjList[i].reserve(adjListSizes[i]);
            }
        }
        
        // For each vertex, broadcast its adjacency list
        for (int i = 0; i < g.n; i++) {
            int size = adjListSizes[i];
            
            if (size > 0) {
                vector<int> edgeDest(size);
                vector<int> edgeWeight(size);
                
                if (rank == 0) {
                    for (int j = 0; j < size; j++) {
                        edgeDest[j] = g.adjList[i][j].to;
                        edgeWeight[j] = g.adjList[i][j].weight;
                    }
                }
                
                MPI_Bcast(edgeDest.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(edgeWeight.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
                
                if (rank != 0) {
                    g.adjList[i].clear();
                    for (int j = 0; j < size; j++) {
                        g.adjList[i].emplace_back(edgeDest[j], edgeWeight[j]);
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    
    // Run the selected algorithm
    double start_time = MPI_Wtime();
    
    // For MPI, assign source vertices to processes (simple partitioning)
    int source = 0;  // Default source
    
    // Check if we need to run the full algorithm or just process distance updates
    bool needFullAlgorithm = true;
    
    // If edges were removed and we already computed the initial tree on rank 0,
    // we only need to run the distance update phase on all processes
    if (rank == 0 && !edgesToRemove.empty()) {
        // We already processed deletions, now we need to update distances
        // This follows the second part of Algorithm 4 (lines 21-47)
        needFullAlgorithm = false;
        
        // Asynchronous distance update phase
        double update_start = MPI_Wtime();
        
        bool change = true;
        while (change) {
            change = false;
            
            #pragma omp parallel
            {
                vector<pair<int, int>> localUpdates; // (node, new parent)
                
                #pragma omp for schedule(dynamic, 64)
                for (int v = 0; v < g.n; v++) {
                    if (g.affected[v]) {
                        g.affected[v] = false;
                        
                        queue<int> Q;
                        Q.push(v);
                        
                        int level = 0;
                        
                        while (!Q.empty()) {
                            int x = Q.front();
                            Q.pop();
                            
                            // Process neighbors
                            for (const Edge& edge : g.adjList[x]) {
                                int n = edge.to;
                                int weight = edge.weight;
                                level++;
                                
                                // Check for better paths through x
                                if (g.dist[x] > g.dist[n] + weight) {
                                    localUpdates.emplace_back(x, n);
                                }
                                
                                // Check for better paths through n
                                if (g.dist[n] > g.dist[x] + weight) {
                                    localUpdates.emplace_back(n, x);
                                }
                                
                                // Continue asynchronously if within level limit
                                if (level <= asynchronyLevel) {
                                    Q.push(n);
                                }
                            }
                        }
                    }
                }
                
                // Apply updates
                #pragma omp critical
                {
                    for (const auto& update : localUpdates) {
                        int node = update.first;
                        int newParent = update.second;
                        
                        if (g.dist[newParent] != INF) {
                            // Find the edge weight
                            int weight = 1; // Default
                            for (const Edge& e : g.adjList[newParent]) {
                                if (e.to == node) {
                                    weight = e.weight;
                                    break;
                                }
                            }
                            
                            int newDist = g.dist[newParent] + weight;
                            if (newDist < g.dist[node]) {
                                g.dist[node] = newDist;
                                g.parent[node] = newParent;
                                g.affected[node] = true;
                                change = true;
                            }
                        }
                    }
                }
            }
        }
        
        double update_end = MPI_Wtime();
        cout << "Time to update distances after deletions: " << (update_end - update_start) << " seconds" << endl;
    }
    
    // Run the full algorithm if needed
    if (needFullAlgorithm) {
        if (useMPI) {
            // Each process handles a subset of source vertices
            int sourcesPerProcess = g.n / numProcesses;
            int startSource = rank * sourcesPerProcess;
            int endSource = (rank == numProcesses - 1) ? g.n : startSource + sourcesPerProcess;
            
            // For this example, we'll just use vertex 0 as the source
            // In a real application, you'd run the algorithm for each source in your range
            if (startSource <= source && source < endSource) {
                if (algorithm == 0) {
                    g.dijkstra_sequential(source);
                } else if (algorithm == 1) {
                    g.dijkstra_openmp_optimized(source);
                } else {
                    g.delta_stepping(source);
                }
            }
        } else {
            // Single process - run the selected algorithm
            if (algorithm == 0) {
                g.dijkstra_sequential(source);
            } else if (algorithm == 1) {
                g.dijkstra_openmp_optimized(source);
            } else {
                g.delta_stepping(source);
            }
        }
    }
    
    // If using MPI, combine results
    if (useMPI) {
        vector<int> globalDist(g.n, INF);
        vector<int> globalParent(g.n, -1);
        
        // Each process may have computed SSSP for different source vertices
        // Here we're just combining results for source vertex 0
        MPI_Allreduce(g.dist.data(), globalDist.data(), g.n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        // For parent pointers, we need to be careful
        // The process that found the minimum distance provides the parent
        vector<int> distWithRank(g.n);
        for (int i = 0; i < g.n; i++) {
            distWithRank[i] = (g.dist[i] == globalDist[i] && g.dist[i] != INF) ? 
                              rank + 1 : 0;
        }
        
        vector<int> winningRank(g.n);
        MPI_Allreduce(distWithRank.data(), winningRank.data(), g.n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        // Only the winning rank broadcasts its parent pointer
        for (int i = 0; i < g.n; i++) {
            if (winningRank[i] == rank + 1) {
                globalParent[i] = g.parent[i];
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, globalParent.data(), g.n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        // Update local distance and parent arrays
        g.dist = globalDist;
        g.parent = globalParent;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        cout << "Time to run SSSP algorithm: " << (end_time - start_time) << " seconds" << endl;
        
        // Count unreachable vertices (for validation)
        int unreachable = 0;
        for (int i = 0; i < g.n; i++) {
            if (g.dist[i] == INF) {
                unreachable++;
            }
        }
        cout << "Unreachable vertices: " << unreachable << " out of " << g.n << endl;
        
        // Save and print output
        g.saveOutputToFile("output.txt");
        cout << "First 10 vertices:" << endl;
        g.printShortestPaths(10);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
*/