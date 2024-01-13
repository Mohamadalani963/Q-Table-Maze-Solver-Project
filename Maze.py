import numpy as np

class Maze:
    def __init__(self, maze_data):
        self.maze = np.array(maze_data)
        self.rows, self.columns = self.maze.shape
        self.start = (0, 0)
        self.end = (self.rows - 1, self.columns - 1)
        self.walls_reward = -10
        self.empty_cell_reward = -1
        self.end_reward = 100
        self.current_position = self.start


    def get_state(self, row, col):
        self.current_position = row,col
        return row * self.columns + col

    def is_valid_move(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.columns 
    
    # and self.maze[row, col] != 1

    def is_end_state(self, row, col):
        return (row, col) == self.end

    def get_neighbors(self, row, col):
        neighbors = []

        for move in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + move[0], col + move[1]
            if self.is_valid_move(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def get_reward(self, row, col):
        if (row, col) == self.end:
            return self.end_reward
        elif self.maze[row, col] == 1:
            return self.walls_reward
        else:
            return self.empty_cell_reward