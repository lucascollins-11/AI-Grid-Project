import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from collections import defaultdict

class Grid:
    def __init__(self, size=10):
        self.size = size
        self.edges = {}  # Dictionary to store edge weights
        self.randomize_weights()
    
    def randomize_weights(self, min_weight=1, max_weight=10):
        """Assign random weights to all edges in the grid"""
        self.edges = {}
        for i in range(self.size):
            for j in range(self.size):
                # For each cell, add edges to neighbors (right and down only, to avoid duplicates)
                if i < self.size - 1:  # Edge to the cell below
                    weight = random.randint(min_weight, max_weight)
                    self.edges[((i, j), (i+1, j))] = weight
                    self.edges[((i+1, j), (i, j))] = weight  # Bidirectional
                
                if j < self.size - 1:  # Edge to the cell to the right
                    weight = random.randint(min_weight, max_weight)
                    self.edges[((i, j), (i, j+1))] = weight
                    self.edges[((i, j+1), (i, j))] = weight  # Bidirectional
    
    def get_neighbors(self, cell):
        """Get all neighboring cells"""
        i, j = cell
        neighbors = []
        # Check all four directions
        for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if 0 <= ni < self.size and 0 <= nj < self.size:
                neighbors.append((ni, nj))
        return neighbors
    
    def get_edge_weight(self, cell1, cell2):
        """Get the weight of the edge between cell1 and cell2"""
        if (cell1, cell2) in self.edges:
            return self.edges[(cell1, cell2)]
        elif (cell2, cell1) in self.edges:
            return self.edges[(cell2, cell1)]
        else:
            raise ValueError(f"No edge between {cell1} and {cell2}")
    
    def heuristic(self, cell, goal):
        """Manhattan distance heuristic for A* and Greedy algorithms"""
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
    
    def uniform_cost_search(self, start, goal):
        """Uniform Cost Search algorithm"""
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        cells_explored = 0
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            cells_explored += 1
            
            if current == goal:
                break
                
            for next_cell in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.get_edge_weight(current, next_cell)
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    heapq.heappush(frontier, (new_cost, next_cell))
                    came_from[next_cell] = current
        
        # Reconstruct path
        path = []
        if goal in came_from:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
        return path, cost_so_far.get(goal, float('inf')), cells_explored
    
    def a_star_search(self, start, goal):
        """A* Search algorithm"""
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        cells_explored = 0
        
        while frontier:
            _, current = heapq.heappop(frontier)
            cells_explored += 1
            
            if current == goal:
                break
                
            for next_cell in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.get_edge_weight(current, next_cell)
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + self.heuristic(next_cell, goal)
                    heapq.heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current
        
        # Reconstruct path
        path = []
        if goal in came_from:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
        return path, cost_so_far.get(goal, float('inf')), cells_explored
    
    def greedy_best_first_search(self, start, goal):
        """Greedy Best-First Search algorithm"""
        frontier = [(0, start)]
        came_from = {start: None}
        cells_explored = 0
        
        while frontier:
            _, current = heapq.heappop(frontier)
            cells_explored += 1
            
            if current == goal:
                break
                
            for next_cell in self.get_neighbors(current):
                if next_cell not in came_from:
                    priority = self.heuristic(next_cell, goal)
                    heapq.heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current
        
        # Reconstruct path
        path = []
        if goal in came_from:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
        # Calculate total cost of the path
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += self.get_edge_weight(path[i], path[i+1])
            
        return path, total_cost, cells_explored
    
    def visualize_paths(self, start, goal, ucs_path, astar_path, greedy_path):
        """Visualize the grid and paths"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        algorithm_names = ["Uniform Cost Search", "A* Search", "Greedy Best-First Search"]
        paths = [ucs_path, astar_path, greedy_path]
        
        for i, (ax, path, name) in enumerate(zip(axes, paths, algorithm_names)):
            # Create a grid
            ax.set_xlim(-0.5, self.size - 0.5)
            ax.set_ylim(-0.5, self.size - 0.5)
            
            # Draw grid lines
            for x in range(self.size):
                ax.axhline(y=x - 0.5, color='black', linestyle='-', alpha=0.2)
                ax.axvline(x=x - 0.5, color='black', linestyle='-', alpha=0.2)
            
            # Draw edge weights (only show a subset for clarity)
            if i == 0:  # Only show weights on the first plot
                for edge, weight in self.edges.items():
                    cell1, cell2 = edge
                    # Only show weights for horizontal edges to avoid clutter
                    if cell1[0] == cell2[0] and abs(cell1[1] - cell2[1]) == 1:
                        mid_x = (cell1[1] + cell2[1]) / 2
                        mid_y = cell1[0]
                        ax.text(mid_x, mid_y, str(weight), ha='center', va='center', color='blue', fontsize=8)
            
            # Draw path
            if path:
                path_x = [cell[1] for cell in path]
                path_y = [cell[0] for cell in path]
                ax.plot(path_x, path_y, 'r-', linewidth=2)
                
                # Draw start and goal
                ax.plot(start[1], start[0], 'go', markersize=10)
                ax.plot(goal[1], goal[0], 'bo', markersize=10)
            
            # Set title and labels
            ax.set_title(f"{name}\nPath Length: {len(path) - 1}, Cost: {self.calculate_path_cost(path)}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.grid(True)
            
            # Add row and column indices
            ax.set_xticks(range(self.size))
            ax.set_yticks(range(self.size))
        
        plt.tight_layout()
        plt.show()
    
    def calculate_path_cost(self, path):
        """Calculate the total cost of a path"""
        if not path or len(path) < 2:
            return 0
        
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += self.get_edge_weight(path[i], path[i+1])
        return total_cost

# Initialize the grid
grid_size = 10
grid = Grid(grid_size)

# Test the grid
def test_grid():
    # Verify that all cells have appropriate connections
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = grid.get_neighbors((i, j))
            # Check that each cell has the correct number of neighbors
            expected_neighbors = 4
            if i == 0 or i == grid_size - 1:
                expected_neighbors -= 1
            if j == 0 or j == grid_size - 1:
                expected_neighbors -= 1
            assert len(neighbors) == expected_neighbors, f"Cell ({i}, {j}) has {len(neighbors)} neighbors, expected {expected_neighbors}"
            
            # Check that edge weights exist for all neighbors
            for neighbor in neighbors:
                assert ((i, j), neighbor) in grid.edges or (neighbor, (i, j)) in grid.edges, \
                    f"Edge between ({i}, {j}) and {neighbor} doesn't have a weight"
    
    print("Grid test passed successfully!")

# Run the test
test_grid()

# Run simulations
def run_simulation(num_trials=5):
    results = defaultdict(lambda: {"total_cost": 0, "total_cells_explored": 0, "avg_path_length": 0})
    
    for _ in range(num_trials):
        # Generate random start and goal positions
        start = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        goal = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        
        # Make sure start and goal are different
        while start == goal:
            goal = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        
        # Run the algorithms
        ucs_path, ucs_cost, ucs_explored = grid.uniform_cost_search(start, goal)
        astar_path, astar_cost, astar_explored = grid.a_star_search(start, goal)
        greedy_path, greedy_cost, greedy_explored = grid.greedy_best_first_search(start, goal)
        
        # Update results
        results["UCS"]["total_cost"] += ucs_cost
        results["UCS"]["total_cells_explored"] += ucs_explored
        results["UCS"]["avg_path_length"] += len(ucs_path)
        
        results["A*"]["total_cost"] += astar_cost
        results["A*"]["total_cells_explored"] += astar_explored
        results["A*"]["avg_path_length"] += len(astar_path)
        
        results["Greedy"]["total_cost"] += greedy_cost
        results["Greedy"]["total_cells_explored"] += greedy_explored
        results["Greedy"]["avg_path_length"] += len(greedy_path)
        
        # Visualize the last trial
        if _ == num_trials - 1:
            print(f"Visualizing paths from {start} to {goal}")
            grid.visualize_paths(start, goal, ucs_path, astar_path, greedy_path)
    
    # Calculate averages
    for algo in results:
        results[algo]["avg_cost"] = results[algo]["total_cost"] / num_trials
        results[algo]["avg_cells_explored"] = results[algo]["total_cells_explored"] / num_trials
        results[algo]["avg_path_length"] = results[algo]["avg_path_length"] / num_trials - 1  # Subtract 1 because we count edges not nodes
    
    # Print results
    print("\nAlgorithm Performance (averaged over {} trials):".format(num_trials))
    print("-" * 80)
    print(f"{'Algorithm':<15} {'Avg Path Cost':<15} {'Avg Cells Explored':<20} {'Avg Path Length':<15}")
    print("-" * 80)
    for algo, stats in results.items():
        print(f"{algo:<15} {stats['avg_cost']:<15.2f} {stats['avg_cells_explored']:<20.2f} {stats['avg_path_length']:<15.2f}")

# Run the simulation
if __name__ == "__main__":
    run_simulation(num_trials=5)