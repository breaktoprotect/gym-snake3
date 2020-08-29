import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Import and supress pygame banner

import pygame
import random
import numpy as np
import math

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False,width=20, height=20,segment_width=50, apple_body_distance=False):
        self.ENV_HEIGHT = height 
        self.ENV_WIDTH = width
        self.action_space = spaces.Discrete(4)
        self.snake_previous = None
        self.SEGMENT_WIDTH = segment_width # For rendering only
        self.RENDER = render
        self.INITIAL_HEALTH = width*2 + height*2
        self.APPLE_BODY_DISTANCE = apple_body_distance # If True, apple and walls will be detected as a value before 0 and 1. Else, boolean.

        # Render using pygame

        self.window = None

    def step(self, action):
        reward = 0 # initialize per step
        opp_facing = (self.facing + 2) % 4 # Calculate opposite of facing
        # If direction and action is NOT the same and not opposite of current direction facing
        if action != self.facing and action != opp_facing:
            # Update the new snake direction
            self.facing = action   
 
        #* Advancing the snake
        self._move_snake()
        self.done = self._check_collision()
        if self.done:
            reward += 0 # assigning negative reward not useful

        #* Consuming apple + growing
        #Simple Reward: Eating apple (1 point)
        if self._check_eaten():
            reward += 1	#+(0.25*self.snake_health/MAX_SNAKE_HEALTH) # 1 + 0.25 x early explorer's reward
            self.snake_health = self.INITIAL_HEALTH + len(self.snake_segments) # Health restored to max 
        #* Starving the snake
        else:
            self.snake_health -= 1 # reduce hitpoint simulate hunger.

        if self.snake_health == 0: 
            self.done = True
            #print("debug: Snake starved to death. :(")

        #* Additional reward for moving closer towards the red apple
        if self.snake_previous:
            # Distance between 
            if all(abs(np.subtract(self.apple_pos, self.snake_segments[0])) <= abs(np.subtract(self.apple_pos, self.snake_previous))):
                reward += 0.00 # Survival reward (was 0.01)

        #* Observations
        if self.done:
            observations = np.array(32*[0]) # dead snake sense nothing
        else:
            eight_direction_sensors = self._sense_all_directions(self.snake_segments[0])
            facing_sensors = self._sense_facing_direction()
            tail_direction = self._sense_tail_direction()

            observations = np.append(eight_direction_sensors, facing_sensors) 
            observations = np.append(observations, tail_direction)

            #distance_to_apple = self._distance_to_apple() #!experimental gym-snakev1 hybrid
            #observations = np.append(observations, distance_to_apple)

        #* Final adjustments
        self.snake_previous = self.snake_segments[0]
        
        # Each step should return: observation, reward, done, info (other information, optional)
        return tuple(observations), reward, self.done, {} #TODO turn others back to tuples, not np.array

    def reset(self):
        # Snake starting position
        head_x = random.randint(3,self.ENV_WIDTH-3)
        head_y = random.randint(3,self.ENV_HEIGHT-3)

        self.snake_segments = [(head_x,head_y),(head_x,head_y+1),(head_x,head_y+2)]
        self.snake_health = self.INITIAL_HEALTH + len(self.snake_segments) # Hitpoints
        self.facing = 0 # 0 is up, 1 is right, 2 is down, 3 is left #TODO randomize
        # Apple starting position
        self.apple_pos = self._spawn_apple()

        # 'Done' state
        self.done = False
  
        return #? need to return state?

    def init_window(self): 
        self.window = pygame.display.set_mode((self.ENV_HEIGHT * self.SEGMENT_WIDTH, self.ENV_WIDTH * self.SEGMENT_WIDTH))
        self.render()
        return

    def close_window(self):
        self.window = pygame.display.quit()
        return
        
    def render(self, mode='human', close=False):
        if not self.render:
            print("[!] Error rendering. Disable during object instantiation.")
            return
        #* Draw & Display
        self.window.fill((30,30,30)) # Clear screen to color black again

        # Walls
        walls_color = (16,5,110)
        for j in range(self.ENV_WIDTH):
            pygame.draw.rect(self.window, walls_color, (0, j*self.SEGMENT_WIDTH , self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))
            pygame.draw.rect(self.window, walls_color, ((self.ENV_HEIGHT-1)*self.SEGMENT_WIDTH, j*self.SEGMENT_WIDTH , self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))

        for i in range(self.ENV_HEIGHT):
            pygame.draw.rect(self.window, walls_color, (i*self.SEGMENT_WIDTH, 0 , self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))
            pygame.draw.rect(self.window, walls_color, (i*self.SEGMENT_WIDTH, (self.ENV_WIDTH-1)*self.SEGMENT_WIDTH , self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))

        # Snake head
        pygame.draw.rect(self.window, (25,165,0), (self.snake_segments[0][0]*self.SEGMENT_WIDTH, self.snake_segments[0][1]*self.SEGMENT_WIDTH, self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))

        # Snake body
        for segment in self.snake_segments[1:]:
            pygame.draw.rect(self.window, (0,205,75), (segment[0]*self.SEGMENT_WIDTH, segment[1]*self.SEGMENT_WIDTH, self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))
        
        # Apple
        pygame.draw.rect(self.window, (205,25,25), (self.apple_pos[0]*self.SEGMENT_WIDTH, self.apple_pos[1]*self.SEGMENT_WIDTH, self.SEGMENT_WIDTH, self.SEGMENT_WIDTH))
        pygame.display.update()
            
        return
    
    # Move the snake by 1 box / square
    def _move_snake(self):
        snake_head = (0,0)
        if self.facing == 0:
            snake_head = np.subtract(self.snake_segments[0], (0, 1))
        elif self.facing == 1:
            snake_head = np.add(self.snake_segments[0], (1, 0))
        elif self.facing == 2:
            snake_head = np.add(self.snake_segments[0], (0, 1))
        else:
            snake_head = np.subtract(self.snake_segments[0], (1, 0))

        leftovers = self.snake_segments[:-1]
        self.snake_segments = [] #reset
        self.snake_segments.append(tuple(snake_head))
        for segment in leftovers:
            self.snake_segments.append(segment)

    def _check_collision(self, snake_head=None):
        if np.all(snake_head) == None: 
            snake_head = self.snake_segments[0]

        # Borders
        if snake_head[0] == self.ENV_HEIGHT-1 or snake_head[1] == self.ENV_WIDTH-1 or \
            snake_head[0] == 0 or snake_head[1] == 0:
            return 1

        # Snake body itself
        for segment in self.snake_segments[1:]:
            if self.snake_segments[0] == segment:
                return 1

        return 0

    def _check_eaten(self):
        if not self.snake_segments[0] == self.apple_pos:
            return 0 

        #* Growing the snake
        additional_segment = self.snake_segments[len(self.snake_segments)-1:] #select the last segment

        if self.facing == 0: 
            additional_segment = np.add(additional_segment, (0, 1))
        if self.facing == 1:
            additional_segment = np.subtract(additional_segment, (1, 0))
        if self.facing == 2:
            additional_segment = np.subtract(additional_segment, (0, 1))
        if self.facing == 3:
            additional_segment = np.add(additional_segment, (1, 0))
        
        self.snake_segments.append(tuple(additional_segment[0]))

        #* Respawn Apple at random location
        self.apple_pos = self._spawn_apple()

        return 1
   
    def _spawn_apple(self):
        position_set = False
        while not position_set:
            apple_position = (random.randrange(1,self.ENV_WIDTH-2), random.randrange(1,self.ENV_HEIGHT-2)) # (0,0) is wall; (19,19) is wall when WIDTH/HEIGHT is 20.

            # Ensure it does not spawn in any parts of the snake
            for index, segment in enumerate(self.snake_segments):
                if np.all(segment == apple_position):
                    break 
                if index == len(self.snake_segments)-1:
                    position_set = True

        return apple_position

    #*** New set of functions for Observations ***
    def _sense_all_directions(self, epicenter):
        # Up sensor
        up_sensor = self._sweep_direction_for_object(epicenter, (0,-1))

        # Top-right sensor
        top_right_sensor = self._sweep_direction_for_object(epicenter, (1,-1))

        # Right sensor
        right_sensor = self._sweep_direction_for_object(epicenter, (1,0))

        # Bottom-right sensor
        bottom_right_sensor = self._sweep_direction_for_object(epicenter, (1,1))

        # Down sensor
        down_sensor = self._sweep_direction_for_object(epicenter, (0,1))

        # Bottom-left sensor
        bottom_left_sensor = self._sweep_direction_for_object(epicenter, (-1,1))

        # Left sensor
        left_sensor = self._sweep_direction_for_object(epicenter, (-1,0))

        # Top-left sensor
        top_left_sensor = self._sweep_direction_for_object(epicenter, (-1,-1))

        return np.array(up_sensor + top_right_sensor + right_sensor + bottom_right_sensor + down_sensor + bottom_left_sensor + left_sensor + top_left_sensor)

    def _sweep_direction_for_object(self, epicenter, vector):
        '''
        From epicenter, sweep towards a direction based on a vector 1 unit at a time.
        E.g. If vector is (0,-1), it's sweeping upwards. If (-1, 0), it's left.
        If (1,1), is sweeping bottom-right. 
        '''
        cur_point = epicenter
        wall_dist = body_dist = apple_dist = 0

        while cur_point[0] > -1 and cur_point[0] < self.ENV_WIDTH and \
            cur_point[1] > -1 and cur_point[1] < self.ENV_HEIGHT: # while within boundaries
            cur_point = np.add(cur_point, vector)

            object_type = self._detect_object_type_at_a_point(cur_point)

            if object_type:
                # Get the normalized distance
                if self.APPLE_BODY_DISTANCE:
                    norm_distance = self._normalized_distance_two_points(epicenter, cur_point)
                else:
                    norm_distance = 1 # Boolean True

                if object_type == 'wall':
                    #wall_dist = norm_distance
                    wall_dist = self._normalized_distance_two_points(epicenter, cur_point) # always distance-based, not boolean
                    return wall_dist, body_dist, apple_dist 

                if object_type == 'body':
                    body_dist = norm_distance
                    return wall_dist, body_dist, apple_dist 

                if object_type == 'apple':
                    apple_dist = norm_distance
                    return wall_dist, body_dist, apple_dist 

        return wall_dist, body_dist, apple_dist # detect only once and return nearest object

    def _detect_object_type_at_a_point(self, point):
        object_type = None

        # Wall 
        if point[0] == self.ENV_WIDTH-1 or point[1] == self.ENV_HEIGHT-1 or \
            point[0] == 0 or point[1] == 0:
            object_type = 'wall'

        # Snake body itself
        for segment in self.snake_segments[1:]:
            if np.all(point == segment):
                object_type = 'body'

        # Apple
        if np.all(point == self.apple_pos):
            object_type = 'apple'

        return object_type # If point has an object, return either 'wall', 'apple' or 'body'. Else, return 'None'

    def _normalized_distance_two_points(self, point_1, point_2):
        '''
        For horizontal and vertical lines, calculate the difference on X components or Y components respectively.
        Note: If diagonal, calculate both x and y components differences and add them together. 
        Cannot use 'math.sqrt((x2 - x1)**2 + (y2 - y1)**2)' formula as snake cannot move diagonally but only 4 directions.
        To move 1 square diagonally, needs two moves.
        '''        
        # Calculate distance (steps) formula
        x1 = point_1[0]
        y1 = point_1[1]        
        x2 = point_2[0]
        y2 = point_2[1]

        x_difference = abs(x2 - x1)
        y_difference = abs(y2 - y1)

        # Determining the max distance (denominator for normalization)
        # Max distance in the environment e.g. total steps required from lower-left corner to top-right corner
        max_distance = self.ENV_HEIGHT + self.ENV_WIDTH

        # Calculate distance between two points
        distance = x_difference + y_difference
        
        # Normalize the distance
        norm_distance = 1 - (distance / max_distance) # Higher value for closer distance

        return norm_distance

    # Convert single facing figure (0 to 3) to a list of 4 x boolean
    def _sense_facing_direction(self):
        face_up = face_right = face_down = face_left = 0

        if self.facing == 0:
            face_up = 1
        if self.facing == 1:
            face_right = 1
        if self.facing == 2:
            face_down = 1
        if self.facing == 3:
            face_left = 1

        return np.array([face_up, face_right, face_down, face_left]) 

    # Detect tail direction
    def _sense_tail_direction(self):

        # Determine vector of tail
        vector_of_tail = np.subtract(self.snake_segments[-1], self.snake_segments[-2])

        # Up
        if np.all(vector_of_tail == (0,-1)):
            return np.array([0, 0, 1, 0])
        # Right
        elif np.all(vector_of_tail == (1,0)):
            return np.array([0, 0, 0, 1])
        # Down
        elif np.all(vector_of_tail == (0,1)):
            return np.array([1, 0, 0, 0])
        # Left
        else:
            return np.array([0, 1, 0, 0])

    #! Experimental hybrid from gym-snakev1
    # Calculate scalar distance between snake head's and apple
    def _distance_to_apple(self, snake=None):
        if not snake:
            snake = self.snake_segments[0] # snake's head

        snake_x = snake[0]
        snake_y = snake[1]        
        apple_x = self.apple_pos[0]
        apple_y = self.apple_pos[1]


        # Calculate distance formula
        max_distance = math.sqrt(self.ENV_WIDTH**2 + self.ENV_HEIGHT**2)
        distance = math.sqrt((apple_x - snake_x)**2 + (apple_y - snake_y)**2)

        # Normalized distance
        norm_distance = 1- (distance / max_distance) # Higher value for closer distance (increase neuron stimulation?)

        return norm_distance

#? Test bed only
if __name__ == "__main__":
    import gym
    import gym_snake3
    env = gym.make('snake3-v0', render=True, segment_width=25, width=12, height=12)
    env.reset()

    observation, reward, done, info = env.step(1)
    observation, reward, done, info = env.step(2)
    observation, reward, done, info = env.step(2)

    print("observation:", observation)
    print("")
    print("sensed_direction:", env._sense_tail_direction())

    env.init_window()
    env.render()
    input()