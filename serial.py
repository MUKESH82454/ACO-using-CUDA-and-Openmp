import numpy as np
import time

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iterations (int): Number of iterations
            decay (float): Rate at which pheromone decays. The pheromone value is multiplied by decay.
            alpha (int or float): Exponent on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): Exponent on distance, higher beta gives distance more weight. Default=1
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone *= self.decay
            print("Iteration:", i+1, "Shortest Path:", all_time_shortest_path)
            time.sleep(0.1)  # Add a delay of 0.1 seconds for clarity
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = np.zeros_like(pheromone)
        non_zero_dist_indices = np.nonzero(dist)
        row[non_zero_dist_indices] = pheromone[non_zero_dist_indices] ** self.alpha * (( 1.0 / dist[non_zero_dist_indices]) ** self.beta)

        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

def create_data(n_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    points = np.random.rand(n_cities, 2)
    return points

def calculate_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(points[i] - points[j])
            distances[i][j] = distance
            distances[j][i] = distance
    return distances

def save_dataset(points, filename):
    np.savetxt(filename, points)

def load_dataset(filename):
    return np.loadtxt(filename)

if __name__ == '__main__':
    # Load dataset from file
    points = load_dataset("cities1000.txt")
    n_cities = len(points)
    distances = calculate_distances(points)
    
    # Start timing
    start_time = time.time()

    # Run the Ant Colony Optimization algorithm
    ant_colony = AntColony(distances, n_ants=10, n_best=5, n_iterations=100, decay=0.95, alpha=1, beta=2)
    shortest_path = ant_colony.run()

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the total distance of the shortest path
    print("Total distance of the shortest path:", shortest_path[1])

    # Print the execution time
    print("Execution Time:", execution_time)