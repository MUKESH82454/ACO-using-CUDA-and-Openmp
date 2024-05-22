import numpy as np

def create_data(n_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    points = np.random.uniform(1, 10, size=(n_cities, 2))  # Generating coordinates between 1 and 10
    return points

def save_dataset(points, filename):
    np.savetxt(filename, points, fmt='%.8f')  # Save coordinates with 8 decimal places

def print_formatted_data(points, num_points, filename):
    with open(filename, 'w') as file:
        for i in range(num_points):
            file.write("{:.8f} {:.8f}\n".format(points[i, 0], points[i, 1]))

if __name__ == '__main__':
    np.random.seed(0)  # For reproducibility
    points = create_data(3000)  # Creating 100 random cities
    save_dataset(points, "cities3000.txt")  # Saving dataset

    # Saving the first 5 points to "cities.txt" in the desired format
    print_formatted_data(points, 3000, "cities3000.txt")
