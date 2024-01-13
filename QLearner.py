import numpy as np
import csv 
class QLearner:
    #optimal when the robot is not blind learning_rate=0.6, discount_factor=0.998, exploration_prob=0.9
    #optimal when the robot is blind learning_rate=0.4 || 0.3, discount_factor=0.998, exploration_prob=0.9

    def __init__(self, num_states, num_actions, learning_rate=0.535, discount_factor=0.99, exploration_prob=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_values = np.zeros((num_states, num_actions))
        self.dizziness_steps =0

    def select_action(self , state):
        # print(self.exploration_prob)
        if np.random.rand() < self.exploration_prob or self.dizziness_steps > 0:
            
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value                
            return np.argmax(self.q_values[state, :])

    def update_q_values(self, state, action, reward, next_state,episode):
        if(self.dizziness_steps>0):
            self.dizziness_steps -=1
        # Q-value update using the Bellman equation
        current_q_value = self.q_values[state, action]
        max_next_q_value = np.max(self.q_values[next_state, :])
        new_q_value = (1 - self.learning_rate)*current_q_value  + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_values[state, action] = new_q_value
        self.exploration_prob = np.exp(-self.discount_factor*episode)
    def get_q_values(self):
        return self.q_values
