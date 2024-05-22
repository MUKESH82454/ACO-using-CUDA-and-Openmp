#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdlib> 
#include <algorithm> 

// Function to generate all paths for each ant
void generatePaths(float* pheromone, float* distances, int* paths, int numAnts, int numCities, unsigned int seed) {
    #pragma omp parallel for
    for (int tid = 0; tid < numAnts; ++tid) {
        // Initialize random number generator state for each thread
        unsigned int threadSeed = seed + tid;
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
            float randVal = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
 // Generate a random number between 0 and 1
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

// Function to calculate the distances of all paths
void calculateDistances(float* distances, int* paths, float* pathDistances, int numAnts, int numCities) {
    #pragma omp parallel for
    for (int tid = 0; tid < numAnts; ++tid) {
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

// Function to update pheromone levels
void updatePheromone(float* pheromone, int* bestPath, float bestDistance, int numCities, float evaporationRate, float q) {
    #pragma omp parallel for
    for (int idx = 0; idx < numCities; ++idx) {
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

    float* pheromone = new float[numCities * numCities];


    // Initialize pheromone

    for (int i = 0; i < numCities * numCities; ++i) {
        pheromone[i] = 1.0f;
    }

    int* paths = new int[numAnts * numCities];
    float* pathDistances = new float[numAnts];

    unsigned int seed = time(NULL);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Start of execution\n";

    for (int iter = 0; iter < numIterations; ++iter) {
        std::cout << "Iteration " << iter + 1 << " of " << numIterations << "\n";

        generatePaths(pheromone, distances, paths, numAnts, numCities, seed);
        std::cout << "Paths generated\n";

        calculateDistances(distances, paths, pathDistances, numAnts, numCities);
        std::cout << "Distances calculated\n";

        // Find best path

        int bestIndex = 0;
        float bestDistance = pathDistances[0];
        for (int i = 1; i < numAnts; ++i) {
            if (pathDistances[i] < bestDistance) {
                bestDistance = pathDistances[i];
                bestIndex = i;
            }
        }
        std::cout << "Best path found\n";

        // Update pheromone
        updatePheromone(pheromone, &paths[bestIndex * numCities], bestDistance, numCities, evaporationRate, q);
        std::cout << "Pheromone updated\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";
    std::cout << "End of execution\n";

    // Clean up
    delete[] cities;
    delete[] distances;
    delete[] pheromone;
    delete[] paths;
    delete[] pathDistances;

    return 0;
}
