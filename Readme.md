## Implementing the Travel Salesman Problem using CUDA and Openmp in GPU

## Usage Instructions:

This repository contains implementations of the Ant Colony Optimization (ACO) algorithm in serial, parallel (CUDA), 
and parallel (OpenMP) versions. These implementations aim to solve the Traveling Salesman Problem (TSP) by finding 
the shortest path through a given set of cities.


## Installation Steps (if any):
For the serial version (Python), ensure you have `Python 3` installed along with NumPy library.
For the parallel version (CUDA), ensure you have ``CUDA Toolkit installed`` and configured properly.
For the parallel version (OpenMP), ensure you have a ``C++ compiler that supports OpenMP directives``.


## Running the Program and Seeing Output:

## Serial Version (Python):
1.Install Python 3 and NumPy if not already installed.
2.Download the Python script `(serial.py)` and the dataset file `(cities1000.txt)`.
3.Run the Python script using the command: `python serial.py`
4.The program will output the shortest path found and the execution time.

## Parallel Version (CUDA):
1.Install CUDA Toolkit if not already installed.
2.Download the CUDA C++ source file `(parallel.cu)` and the dataset file `(cities1000.txt)`.
3.Compile the CUDA source file using the command:` nvcc -o parallel parallel.cu`
4.Run the compiled executable using the command: `./parallel`
5.The program will output the shortest path found and the execution time.

## Parallel Version (OpenMP):
1.Ensure you have a C++ compiler that supports OpenMP directives (e.g., GCC).
2.Download the C++ source file `(openmp_parallel.cpp)` and the dataset file `(cities1000.txt)`.
3.Compile the C++ source file with OpenMP support using the command:`` g++ -fopenmp -o openmp_parallel openmp_parallel.cpp``
4.Run the compiled executable using the command:`` ./openmp_parallel``
5.The program will output the shortest path found and the execution time.


## Demo Scripts:
Below are the demo scripts for running the programs:

# Serial Version (Python):

``python serial.py``

# Parallel Version (CUDA):

``nvcc -o parallel parallel.cu``
``./parallel``

# Parallel Version (OpenMP):

``g++ -fopenmp -o openmp_parallel openmp_parallel.cpp``
``./openmp_parallel``


## What Does the Program Do?
The program implements the Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP). 
It finds the shortest path through a given set of cities by simulating the foraging behavior of ants.
The algorithm iteratively improves the solution by adjusting pheromone levels on the edges of the graph based 
on the quality of the paths discovered by the ants.