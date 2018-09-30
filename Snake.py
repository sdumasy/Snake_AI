#!/usr/bin/env python3

class Snake():
    """
    This class represents the Snake
    """

    def __init__(self, pos):
        """
        Initialise a snake object with position, add initial direction south and all elements to the list
        :param pos: The position of the head of the snake
        """
        self.headx, self.heady = pos
        self.length = 3
        self.elements = [[self.headx, self.heady]]
        self.current_dir = 'south'

        for x in range(1, self.length):
            self.elements.append((self.get_pos()[0], self.get_pos()[1] - x))



    def move(self, direction):
        """
        Move the snake taking into account it's current direction,
        add new position as the head of the snake, pop the tail,
        going south increases the Y
        :param direction: direction in which to head to (0: left, 1: straight, 2: right)
        """
        if((self.current_dir == "east" and direction == 0) or
                (self.current_dir == "west" and direction == 2) or
                (self.current_dir == "north" and direction == 1)):
            self.heady -=  1
            self.current_dir = 'north'
        elif((self.current_dir == "east" and direction == 1) or
                (self.current_dir == "south" and direction == 0) or
                (self.current_dir == "north" and direction == 2)):
            self.current_dir = 'east'
            self.headx +=  1
        elif((self.current_dir == "west" and direction == 1) or
                (self.current_dir == "south" and direction == 2) or
                (self.current_dir == "north" and direction == 0)):
            self.current_dir = 'west'
            self.headx -= 1
        elif((self.current_dir == "west" and direction == 0) or
                (self.current_dir == "south" and direction == 1) or
                (self.current_dir == "east" and direction == 2)):
            self.current_dir = 'south'
            self.heady += 1
        else:
            raise ValueError("Chosen action is not allowed")


        self.elements.pop()
        self.elements = [(self.headx, self.heady)] + self.elements[0:]

    def get_pos(self):
        """
        Get the position of the snake (x, y)
        :return: the x,y position
        """
        return self.headx, self.heady

