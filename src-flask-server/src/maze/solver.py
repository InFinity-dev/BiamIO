import time
import random
import logging

logging.basicConfig(level=logging.DEBUG)


class Solver(object):
    """Base class for solution methods.
    Every new solution method should override the solve method.

    Attributes:
        maze (list): The maze which is being solved.
        neighbor_method:
        quiet_mode: When enabled, information is not outputted to the console

    """

    def __init__(self, maze, quiet_mode, neighbor_method):
        logging.debug("Class Solver ctor called")

        self.maze = maze
        self.neighbor_method = neighbor_method
        self.name = ""
        self.quiet_mode = quiet_mode

    def solve(self):
        logging.debug('Class: Solver solve called')
        raise NotImplementedError

    def get_name(self):
        logging.debug('Class Solver get_name called')
        raise self.name

    def get_path(self):
        logging.debug('Class Solver get_path called')
        return self.path

class DepthFirstBacktracker(Solver):
    """A solver that implements the depth-first recursive backtracker algorithm.
    """

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class DepthFirstBacktracker ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Depth First Backtracker"

    def solve(self):
        logging.debug("Class DepthFirstBacktracker solve called")
        k_curr, l_curr = self.maze.entry_coor  # Where to start searching
        self.maze.grid[k_curr][l_curr].visited = True  # Set initial cell to visited
        visited_cells = list()  # Stack of visited cells for backtracking
        path = list()  # To track path of solution and backtracking cells
        if not self.quiet_mode:
            print("\nSolving the maze with depth-first search...")

        time_start = time.time()

        while (k_curr, l_curr) != self.maze.exit_coor:  # While the exit cell has not been encountered
            neighbour_indices = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indices
            neighbour_indices = self.maze.validate_neighbours_solve(neighbour_indices, k_curr,
                                                                    l_curr, self.maze.exit_coor[0],
                                                                    self.maze.exit_coor[1], self.neighbor_method)

            if neighbour_indices is not None:  # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))  # Add current cell to stack
                path.append(((k_curr, l_curr), False))  # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbour_indices)  # Choose random neighbour
                self.maze.grid[k_next][l_next].visited = True  # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(visited_cells) > 0:  # If there are no unvisited neighbour cells
                path.append(((k_curr, l_curr), True))  # Add coordinates to part of search path
                k_curr, l_curr = visited_cells.pop()  # Pop previous visited cell (backtracking)

        path.append(((k_curr, l_curr), False))  # Append final location to path
        if not self.quiet_mode:
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        logging.debug('Class DepthFirstBacktracker leaving solve')
        return path
