#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>


// CUDA kernel to generate all paths for each ant
__global__ void generatePaths(float* pheromone, float* distances, int* paths, int numAnts, int numCities, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numAnts) {
        curandState_t state;
        curand_init(seed + tid, 0, 0, &state); // Initialize the random number generator state for each thread
        int startCity = tid % numCities;
        paths[tid * numCities] = startCity; // Set the start city for each ant
        int currentCity = startCity;
        for (int i = 1; i < numCities; ++i) {
            float total = 0.0f;
            for (int j = 0; j < numCities; ++j) {

                if (j != currentCity) {

                    // Calculate the total probability
                    total += powf(pheromone[currentCity * numCities + j], 2.0f) * powf(1.0f / distances[currentCity * numCities + j], 5.0f);
                }
            }
            float randVal = curand_uniform(&state); // Generate a random number between 0 and 1
            float threshold = 0.0f;
            for (int j = 0; j < numCities; ++j) {
                if (j != currentCity) {
                    
                    // Calculate the probability of selecting each city
                    threshold += powf(pheromone[currentCity * numCities + j], 2.0f) * powf(1.0f / distances[currentCity * numCities + j], 5.0f) / total;
                    if (randVal <= threshold) {
                        currentCity = j; // Move to the next city based on the probability
                        break;
                    }
                }
            }
            paths[tid * numCities + i] = currentCity; // Assign the selected city to the path
        }
    }
}



// CUDA kernel to calculate the distances of all paths
__global__ void calculateDistances(float* distances, int* paths, float* pathDistances, int numAnts, int numCities) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numAnts) {
        float pathDist = 0.0f;
        for (int i = 0; i < numCities - 1; ++i) {
            int city1 = paths[tid * numCities + i];
            int city2 = paths[tid * numCities + i + 1];
            pathDist += distances[city1 * numCities + city2]; // Accumulate the distance between consecutive cities
        }
        int startCity = paths[tid * numCities];
        int endCity = paths[tid * numCities + numCities - 1];
        pathDist += distances[endCity * numCities + startCity]; // Add the distance from the last city back to the start city
        pathDistances[tid] = pathDist; // Save the total distance of the path
    }
}



// CUDA kernel to update pheromone levels
__global__ void updatePheromone(float* pheromone, int* bestPath, float bestDistance, int numCities, float evaporationRate, float q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCities) {
        int city1 = bestPath[idx];
        int city2 = bestPath[(idx + 1) % numCities];
        // Update the pheromone level on the edge between city1 and city2
        pheromone[city1 * numCities + city2] *= (1.0f - evaporationRate);
        pheromone[city1 * numCities + city2] += (q / bestDistance);
        pheromone[city2 * numCities + city1] = pheromone[city1 * numCities + city2]; // Update the symmetric edge
    }
}


int main() {
    std::ifstream file("cities1000.txt");
    if (!file) {
        std::cerr << "Error: Unable to open file.\n";
        return 1;
    }

    const int numCities = 1000;
    const int numAnts = 10;
    const int numIterations = 100;
    const float evaporationRate = 0.95f;
    const float q = 1.0f;

    float* cities = new float[numCities * 2];
    for (int i = 0; i < numCities * 2; ++i) {
        if (!(file >> cities[i])) {
            std::cerr << "Error reading input.\n";
            delete[] cities;
            return 1;
        }
    }

    float* distances = new float[numCities * numCities];
    for (int i = 0; i < numCities; ++i) {
        for (int j = 0; j < numCities; ++j) {
            float dx = cities[i * 2] - cities[j * 2];
            float dy = cities[i * 2 + 1] - cities[j * 2 + 1];
            distances[i * numCities + j] = sqrt(dx * dx + dy * dy);
        }
    }

    float* devCities;
    cudaMalloc(&devCities, numCities * 2 * sizeof(float));
    cudaMemcpy(devCities, cities, numCities * 2 * sizeof(float), cudaMemcpyHostToDevice);

    float* devDistances;
    cudaMalloc(&devDistances, numCities * numCities * sizeof(float));
    cudaMemcpy(devDistances, distances, numCities * numCities * sizeof(float), cudaMemcpyHostToDevice);

    float* devPheromone;
    cudaMalloc(&devPheromone, numCities * numCities * sizeof(float));
    cudaMemset(devPheromone, 1.0f, numCities * numCities * sizeof(float));

    int* devPaths;
    cudaMalloc(&devPaths, numAnts * numCities * sizeof(int));

    float* devPathDistances;
    cudaMalloc(&devPathDistances, numAnts * sizeof(float));

    dim3 blockSize(256);
    dim3 gridSize((numAnts + blockSize.x - 1) / blockSize.x);

    unsigned int seed = time(NULL);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Start of execution\n";

    for (int iter = 0; iter < numIterations; ++iter) {
        std::cout << "Iteration " << iter + 1 << " of " << numIterations << "\n";

        generatePaths<<<gridSize, blockSize>>>(devPheromone, devDistances, devPaths, numAnts, numCities, seed);
        cudaDeviceSynchronize();
        std::cout << "Paths generated\n";

        calculateDistances<<<gridSize, blockSize>>>(devDistances, devPaths, devPathDistances, numAnts, numCities);
        cudaDeviceSynchronize();
        std::cout << "Distances calculated\n";

        // Find best path
        int bestIndex = 0;
        float bestDistance;
        cudaMemcpy(&bestDistance, devPathDistances, sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 1; i < numAnts; ++i) {
            float dist;
            cudaMemcpy(&dist, &devPathDistances[i], sizeof(float), cudaMemcpyDeviceToHost);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestIndex = i;
            }
        }
        std::cout << "Best path found\n";

        // Update pheromone
        updatePheromone<<<gridSize, blockSize>>>(devPheromone, &devPaths[bestIndex * numCities], bestDistance, numCities, evaporationRate, q);
        cudaDeviceSynchronize();
        std::cout << "Pheromone updated\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";
    std::cout << "End of execution\n";

    // Clean up
    delete[] cities;
    delete[] distances;
    cudaFree(devCities);
    cudaFree(devDistances);
    cudaFree(devPheromone);
    cudaFree(devPaths);
    cudaFree(devPathDistances);

    return 0;
}