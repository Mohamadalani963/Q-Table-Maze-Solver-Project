from Maze import Maze
from QLearner import QLearner
import matplotlib.pyplot as plt
import pygame
import numpy as np
from MazeVisualizer import MazeVisualizer
import csv


# Example maze where cell with value 0 for the empty cell and 1 for the wall
maze_data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
]

def plot_training_progress(reward_history):
    plt.plot(reward_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
maze = Maze(maze_data)

learner = QLearner(num_states=maze.rows * maze.columns, num_actions=4)

num_episodes = 100

reward_history = []

mazeVisualizer = MazeVisualizer(maze)
for episode in range(num_episodes):
    learner.dizziness_steps =0
    row, col = maze.start
    total_reward = 0
    max_steps = 100
    with open('q_table_values.csv', mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(f"#{episode}#")
        writer.writerows(learner.q_values)

    while not maze.is_end_state(row, col) and max_steps > 0:
        for event in pygame.event.get():
             if event.type == pygame.QUIT :
                 pygame.quit()
                 quit()

        mazeVisualizer.screen.fill((255, 255, 255))
        mazeVisualizer.draw_maze()
        mazeVisualizer.draw_agent((row, col))
        mazeVisualizer.draw_episode_number(episode)
        pygame.display.flip()
        pygame.time.delay(1)  # Adjust the speed of visualization

        state = maze.get_state(row, col)
        action = learner.select_action(state)
        reward = maze.get_reward(row, col)
        total_reward += reward
        available_moves = maze.get_neighbors(row, col)
        if(maze_data[row][col]==1):
                learner.dizziness_steps +=5
        if(len(available_moves) > action):
            next_row, next_col = available_moves[action]
        elif(len(available_moves)!=0):
            next_row,next_col = available_moves[0]
        else:
            next_row,next_col = maze.current_position
        
        next_state = maze.get_state(next_row, next_col)

        learner.update_q_values(state, action, reward, next_state, episode)
        row, col = next_row, next_col
        max_steps -= 1

    if maze.is_end_state(row, col):
        total_reward += maze.end_reward

    reward_history.append(total_reward)

plot_training_progress(reward_history)


while(True):
    # For testing the final result of training  
    final_q_values = learner.get_q_values()
    row, col = maze.start
    max_steps = 100
    while not maze.is_end_state(row, col) and max_steps > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.quit()
                quit()
        mazeVisualizer.screen.fill((255, 255, 255))
        mazeVisualizer.draw_maze()
        mazeVisualizer.draw_agent((row, col))
        pygame.display.flip()
        pygame.time.delay(100)  # Adjust the speed of visualization
        state = maze.get_state(row, col)
        action = np.argmax(final_q_values[state])
        reward = maze.get_reward(row, col)
        available_moves = maze.get_neighbors(row, col)
        if(len(available_moves) > action):
            next_row, next_col = available_moves[action]
        elif(len(available_moves)!=0):
            next_row,next_col = available_moves[0]
        else:
            next_row,next_col = maze.current_position
        next_state = maze.get_state(next_row, next_col)
        row, col = next_row, next_col
        max_steps -= 1
    # TODO check out how to keep the screen active after finishing the maze

