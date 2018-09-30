#!/usr/bin/env python3
import sys
import pygame
import time
import random
import Snake
import numpy as np


class Env():
    """class for moving objects
    """
    def __init__(self):
        """Initialize the environment, set the visualisation boolean, set the size of the field,
        set the block size and screen resolution"""
        self.visualize = False
        self.field_size = 22
        if self.visualize:
            pygame.init()
            self.block_size = 20
            self.res = [440, 440]
            self.screen = pygame.display.set_mode(self.res)

        self.reset_env()

    def matrix_to_game_dims(self, x):
        """Convert a matrix position to a position on the game screen"""
        return 10 + x * self.block_size

    def take_action(self, direction):
        """Take an action and move the snake on the matrix and on the game screen"""

        if self.visualize:
            pygame.draw.circle(self.screen, (0, 0, 0), (self.matrix_to_game_dims(self.snake.elements[-1][0]), self.matrix_to_game_dims(self.snake.elements[-1][1])), int(self.block_size / 2))
        self.update_matrix(self.snake.elements[-1], 0)

        self.snake.move(direction)

        if self.check_dead():
            return self.reward, True

        self.check_bug()
        self.draw_snake()
        if self.visualize:
            pygame.display.flip()

        return self.reward, False

    def draw_snake(self):
        """Draw the snake with all it's elements on the matrix and the game screen"""
        if self.check_matrix(self.snake.get_pos()) == 1:
            raise ValueError("Can't build snake in the wall")
        self.update_matrix(self.snake.get_pos(), 3)
        if self.visualize:
            pygame.draw.circle(self.screen, (0, 0, 255), (self.matrix_to_game_dims(self.snake.get_pos()[0]), self.matrix_to_game_dims(self.snake.get_pos()[1])), int(self.block_size / 2))
        for element in self.snake.elements[1:]:
            self.update_matrix(element, 2)
            if self.visualize:
                pygame.draw.circle(self.screen, (199, 21, 133), (self.matrix_to_game_dims(element[0]), self.matrix_to_game_dims(element[1])),
                               int(self.block_size / 2))

    def update_matrix(self, pos, value):
        """Update  a specific value of the matrix"""
        if self.check_matrix(pos) != 1:
            self.matrix[pos[1], pos[0]] = value

    def check_matrix(self, pos):
        """Check and return the  value on a specific value of the matrix"""
        if pos[0] > self.field_size - 1 or pos[1] > self.field_size - 1 or pos[0] < 0 or pos[1] < 0:
            return 1
        return self.matrix[pos[1], pos[0]]

    def check_dead(self):
        """Check if the snakes hits itself or a wall, if so,
        reduce reward and die"""
        if (self.check_matrix(self.snake.get_pos()) == 1) or (self.check_matrix(self.snake.get_pos()) == 2):
            self.reward = self.reward - 10
            return True
        return False

    def check_bug(self):
        """Check if the snake hits a bug, if so, increase the lenght of the snake,
        add a reward """
        if self.check_matrix(self.snake.get_pos()) == 5:
            self.score += 1
            self.reward = self.reward + 10
            self.snake.elements.append(self.snake.elements[-1])
            self.create_bug(self.get_free_pos())

    def draw_map(self):
        """Draw the walls and open spaces on the game screen"""
        for r_index, row in enumerate(self.matrix):
            for c_index, item in enumerate(row):
                if(item == 1):
                    pygame.draw.rect(self.screen, (250, 215, 0), (r_index * 20, c_index * 20, self.block_size, self.block_size), 0)
                elif(item == 0):
                    pygame.draw.circle(self.screen, (0, 0, 0), (self.matrix_to_game_dims(r_index), self.matrix_to_game_dims(c_index)), int(self.block_size / 2))

        pygame.display.flip()

    def is_danger(self, pos):
        """Check if there is danger located on a position in the matrix"""
        if(self.check_matrix(pos) == 1 or self.check_matrix(pos) == 2):
            return 1
        return 0

    def get_features(self):
        """Create a feature array where the snake will learn from,
        First 4 digits are the direction the snake is moving,
        next 3 digits are to check if there is danger in one of his next moves,
        last 4 digits indicate in which direction the bug is located"""
        features = np.array([])
        pos = self.snake.get_pos()
        if self.snake.current_dir == 'north':
            features = np.append(features, [1, 0, 0, 0])
            features = np.append(features, self.is_danger((pos[0] - 1, pos[1])))
            features = np.append(features, self.is_danger((pos[0], pos[1] - 1)))
            features = np.append(features, self.is_danger((pos[0] + 1, pos[1])))
        elif self.snake.current_dir == 'east':
            features = np.append(features, [0, 0, 1, 0])
            features = np.append(features, self.is_danger((pos[0], pos[1] - 1)))
            features = np.append(features, self.is_danger((pos[0] + 1, pos[1])))
            features = np.append(features, self.is_danger((pos[0], pos[1] + 1)))
        elif self.snake.current_dir == 'south':
            features = np.append(features, [0, 1, 0, 0])
            features = np.append(features, self.is_danger((pos[0] + 1, pos[1])))
            features = np.append(features, self.is_danger((pos[0], pos[1] + 1)))
            features = np.append(features, self.is_danger((pos[0] - 1, pos[1])))
        elif self.snake.current_dir == 'west':
            features = np.append(features, [0, 0, 0, 1])
            features = np.append(features, self.is_danger((pos[0], pos[1] + 1)))
            features = np.append(features, self.is_danger((pos[0] - 1, pos[1])))
            features = np.append(features, self.is_danger((pos[0], pos[1] - 1)))

        pos_x = self.candy_pos[0] - pos[0]
        pos_y = self.candy_pos[1] - pos[1]

        if pos_y > 0:
            features = np.append(features, [0, 1])
        elif pos_y < 0:
            features = np.append(features, [1, 0])
        else:
            features = np.append(features, [0, 0])

        if pos_x > 0:
            features = np.append(features, [1, 0])
        elif pos_x < 0:
            features = np.append(features, [0, 1])
        else:
            features = np.append(features, [0, 0])

        return features


    def create_bug(self, pos):
        """Creaate the bug on a free location, update the matrix and draw it"""
        if self.check_matrix(pos) != 0:
            ValueError("Invalid position for bug")
        self.candy_pos = pos
        self.update_matrix(self.candy_pos, 5)
        if self.visualize:
            pygame.draw.circle(self.screen, (255, 0, 0), (self.matrix_to_game_dims(self.candy_pos[0]), self.matrix_to_game_dims(self.candy_pos[1])), int(self.block_size / 2))
            pygame.display.flip()

    def create_map(self):
        """Create the matrix / map, 1's for wall the rest 0"""
        self.matrix = np.ones((self.field_size ,self.field_size))
        self.matrix[1:-1,1:-1] = 0 # Fill walls with 1's
        if self.visualize:
            self.draw_map()

    def get_free_pos(self):
        """Get a position in the game matrix where there is a 0 (free spot)"""
        pos = (random.randint(0, self.field_size - 1), random.randint(0, self.field_size - 1))
        while self.check_matrix(pos) != 0:
            pos = (random.randint(0, self.field_size - 1), random.randint(0, self.field_size - 1))
        return pos

    def create_snake(self, pos):
        """Create the snake and draw it"""
        self.snake = Snake.Snake(pos)
        self.draw_snake()

    def reset_env(self):
        """Resets the game environment"""
        self.create_map()
        self.create_snake((self.rand_num_matrix(), self.rand_num_matrix()))

        self.create_bug(self.get_free_pos())
        self.reward = 0
        self.score = 0
        if self.visualize:
            pygame.display.flip()

    def rand_num_matrix(self):
        """Get a random starting position for the snake 2 blocks from the wall"""
        return random.randint(5, self.field_size - 5)
