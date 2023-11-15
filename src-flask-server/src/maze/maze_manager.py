from .maze import Maze
from .solver import DepthFirstBacktracker

class MazeManager(object):
    """A manager that abstracts the interaction with the library's components. The graphs, animations, maze creation,
    and solutions are all handled through the manager.

    Attributes:
        mazes (list): It is possible to have more than one maze. They are stored inside this variable.
        media_name (string): The filename for animations and images
        quiet_mode (bool): When true, information is not shown on the console
    """

    def __init__(self):
        self.mazes = []
        self.media_name = ""
        self.quiet_mode = False

    def add_maze(self, row, col, id=0):
        """Add a maze to the manager. We give the maze an index of
        the total number of mazes in the manager. As long as we don't
        add functionality to delete mazes from the manager, the ids will
        always be unique. Note that the id will always be greater than 0 because
        we add 1 to the length of self.mazes, which is set after the id assignment

        Args:
            row (int): The height of the maze
            col (int): The width of the maze
            id (int):  The optional unique id of the maze.

        Returns
            Maze: The newly created maze
        """

        if id is not 0:
            self.mazes.append(Maze(row, col, id))
        else:
            if len(self.mazes) < 1:
                self.mazes.append(Maze(row, col, 0))
            else:
                self.mazes.append(Maze(row, col, len(self.mazes) + 1))

        return self.mazes[-1]

    def solve_maze(self, maze_id, method, neighbor_method="fancy"):
        """ Called to solve a maze by a particular method. The method
        is specified by a string. The options are
            1. DepthFirstBacktracker
            2.
            3.
        Args:
            maze_id (int): The id of the maze that will be solved
            method (string): The name of the method (see above)
            neighbor_method:

        """
        maze = self.get_maze(maze_id)
        if maze is None:
            print("Unable to locate maze. Exiting solver.")
            return None

        """DEVNOTE: When adding a new solution method, call it from here.
            Also update the list of names in the documentation above"""
        solver = DepthFirstBacktracker(maze, neighbor_method, self.quiet_mode)
        maze.solution_path = solver.solve()
