import pygame
import numpy as np

class MazeVisualizer:
    def __init__(self, maze):
        self.maze = maze
        self.cell_size = 50  # Adjust the cell size based on your preference
        self.colors = {
            0: (255, 255, 255),  # Empty cell color
            1: (0, 0, 0),        # Wall color
            2: (0, 255, 0),      # Agent color
            'text': (0, 0, 0),   # episode color
        }

        pygame.init()
        screen_size = (self.maze.columns * self.cell_size , self.maze.rows * self.cell_size)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Maze Visualization')
        self.font = pygame.font.SysFont(None, 36)  # Font for displaying text

    def draw_maze(self):
        for row in range(self.maze.rows):
            for col in range(self.maze.columns):
                pygame.draw.rect(self.screen, self.colors[self.maze.maze[row, col]],
                                 (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def draw_agent(self, current_position):
        pygame.draw.rect(self.screen, self.colors[2],
                         (current_position[1] * self.cell_size, current_position[0] * self.cell_size,
                          self.cell_size, self.cell_size))

    def draw_episode_number(self, episode_number):
        text = self.font.render(f'Episode: {episode_number}', True, self.colors['text'])
        self.screen.blit(text, (10, 10))  # Adjust the position as needed