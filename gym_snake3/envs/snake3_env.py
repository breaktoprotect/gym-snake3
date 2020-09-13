import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Import and supress pygame banner

import pygame
import random
import numpy as np
import math

from numba import jit # optimization/speed-up

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False,width=20, height=20,segment_width=50, apple_body_distance=False, randomness_seed=11111):
        self.ENV_HEIGHT = height 
        self.ENV_WIDTH = width
        self.action_space = spaces.Discrete(4)
        self.snake_previous = None
        self.SEGMENT_WIDTH = segment_width # For rendering only
        self.RENDER = render
        self.INITIAL_HEALTH = width * height # More liberal to allow "long route" snakes
        self.APPLE_BODY_DISTANCE = apple_body_distance # If True, apple and walls will be detected as a value before 0 and 1. Else, boolean.
        self.END_GAME_SCORE = ((width-2)*(height-2)) - 3 # Walls not counted

        self.window = None # Pygame window

        # Randomness seeding (for better training? #!exprimental)
        self.apple_rng_seed = self.start_pos_rng_seed = randomness_seed        # np.random.RandomState(seed) #TODO un-hardcode it 

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
            self.done = 1
            #print("debug: Snake starved to death. :(")

        # Additional reward for moving closer towards the red apple
        '''
        if self.snake_previous:
            # Distance between 
            if all(abs(np.subtract(self.apple_pos, self.snake_segments[0])) <= abs(np.subtract(self.apple_pos, self.snake_previous))):
                reward += 0.00 # Survival reward (was 0.01)
        ''' #!Obsoleted

        #* Observations
        if self.done:
            observations = np.array(32*[0]) # dead snake sense nothing
        else:
            #eight_direction_sensors = self._sense_all_directions(self.snake_segments[0]) #! to obsolete
            eight_direction_sensors = self._new_sense_all_directions()
            
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
        # Randomness seeding (for better training? #!exprimental)
        self.start_pos_rng = np.random.RandomState(self.apple_rng_seed)
        self.apple_rng = np.random.RandomState(self.start_pos_rng_seed) 

        # Snake starting position
        head_x = self.start_pos_rng.randint(3,self.ENV_WIDTH-3)
        head_y = self.start_pos_rng.randint(3,self.ENV_HEIGHT-3)
        self.facing = np.random.randint(0,3) # 0 is up, 1 is right, 2 is down, 3 is left
        if self.facing == 0:
            self.snake_segments = [(head_x,head_y),(head_x,head_y+1),(head_x,head_y+2)]
        elif self.facing == 1:
            self.snake_segments = [(head_x,head_y),(head_x-1,head_y),(head_x-2,head_y)]
        elif self.facing == 2:
            self.snake_segments = [(head_x,head_y),(head_x,head_y-1),(head_x,head_y-2)]
        else:
            self.snake_segments = [(head_x,head_y),(head_x+1,head_y),(head_x+2,head_y)]

        self.snake_health = self.INITIAL_HEALTH + len(self.snake_segments) # Hitpoints
        
        # Apple starting position
        self.apple_pos = self._spawn_apple()

        # Current game score
        self.current_game_score = 0

        # 'Done' state
        self.done = 0
  
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
        else:
            self.current_game_score += 1

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

        #* Check end game condition
        if self.current_game_score == self.END_GAME_SCORE:
            self.done = 2

            #debug
            print("[!] Game completed! HURRAY!~!~!~!~!~!~!~!")

            return 1

        #* Respawn Apple at random location
        self.apple_pos = self._spawn_apple()

        return 1
   
    def _spawn_apple(self):
        position_set = False
        while not position_set:
            apple_position = (self.apple_rng.randint(1,self.ENV_WIDTH-1), self.apple_rng.randint(1,self.ENV_HEIGHT-1)) # (0,0) is wall; (19,19) is wall when WIDTH/HEIGHT is 20.

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
                    #wall_dist = self._normalized_distance_two_points(epicenter, cur_point) # always distance-based, not boolean
                    wall_dist = self._simple_distance_two_points(epicenter, cur_point)
                    #return wall_dist, body_dist, apple_dist 

                if object_type == 'body':
                    body_dist = norm_distance
                    #return wall_dist, body_dist, apple_dist 

                if object_type == 'apple':
                    apple_dist = norm_distance
                    #return wall_dist, body_dist, apple_dist 

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
    #* ------------------------------------------------
    #* Newly Optimized All-Directions Object Detections
    #* ------------------------------------------------
    def _new_sense_all_directions(self):
        epicenter = self.snake_segments[:1][0] # Get list, then select position 0
        apple_pos = self.apple_pos
        body_segments = self.snake_segments[1:] #excluding the head
        env_width = self.ENV_WIDTH
        env_height = self.ENV_HEIGHT

        # Calculate apple
        apple_observation = self._sense_object_all_directions(epicenter, apple_pos)
        
        # calculate snake body
        body_observation = np.zeros(8)
        for segment in body_segments:
            cur_observation = self._sense_object_all_directions(epicenter, segment)
            
            body_observation = np.add(body_observation, cur_observation)

        body_observation = np.clip(body_observation, 0, 1)
        
        # Calculate all 8 direction of distances of walls 
        wall_observation = np.zeros(8)
        wall_pt_1, wall_pt_5 = self._locate_diagonal_walls(epicenter, -1, env_width, env_height) # / line
        wall_pt_7, wall_pt_3 = self._locate_diagonal_walls(epicenter, 1, env_width, env_height)  # \ line
        
        wall_observation[0] = self._simple_distance_two_points(epicenter, (epicenter[0], 0))
        wall_observation[1] = self._simple_distance_two_points(epicenter, wall_pt_1)
        wall_observation[2] = self._simple_distance_two_points(epicenter, (env_width-1, epicenter[1]))
        wall_observation[3] = self._simple_distance_two_points(epicenter, wall_pt_3)
        wall_observation[4] = self._simple_distance_two_points(epicenter, (epicenter[0], env_height-1))
        wall_observation[5] = self._simple_distance_two_points(epicenter, wall_pt_5)
        wall_observation[6] = self._simple_distance_two_points(epicenter, (0, epicenter[1]))
        wall_observation[7] = self._simple_distance_two_points(epicenter, wall_pt_7)
        
        # wall, apple, body
        return wall_observation[0], body_observation[0], apple_observation[0], wall_observation[1], body_observation[1], apple_observation[1], wall_observation[2], body_observation[2], apple_observation[2], wall_observation[3], body_observation[3], apple_observation[3], wall_observation[4], body_observation[4], apple_observation[4], wall_observation[5], body_observation[5], apple_observation[5], wall_observation[6], body_observation[6], apple_observation[6], wall_observation[7], body_observation[7], apple_observation[7]

    @staticmethod
    @jit(nopython=True)
    def _locate_diagonal_walls(epicenter, gradient, env_width, env_height):
        if gradient == -1: # / line
            # Get line equation: y = mx + c
            m = -1 #gradient
            c = epicenter[1] - (m*epicenter[0]) # +c component
            # y = 0, x = (y - c)/m
            top_wall_y = 0
            top_wall_x = (top_wall_y - c)/m
            if top_wall_x > (env_width-1):
                top_wall_x = (env_width-1)
                top_wall_y = m*top_wall_x + c
                
                bottom_wall_y = (env_height-1)
                bottom_wall_x = (bottom_wall_y - c) / m
            else:
                bottom_wall_x = 0
                bottom_wall_y = m*bottom_wall_x + c
                
            return (top_wall_x, top_wall_y), (bottom_wall_x, bottom_wall_y)
                
        else: #gradient == 1
            m = 1 #gradient
            c = epicenter[1] - (m*epicenter[0]) # c component
            
            top_wall_y = 0
            top_wall_x = (top_wall_y - c)/m
            if top_wall_x < 0:
                top_wall_x = 0
                top_wall_y = m*top_wall_x + c
                
                bottom_wall_y = (env_height -1)
                bottom_wall_x = (bottom_wall_y - c)/m
            else:
                bottom_wall_x = (env_width - 1)
                bottom_wall_y = m*bottom_wall_x + c
                
            return (top_wall_x, top_wall_y), (bottom_wall_x, bottom_wall_y) 
            

    # Boolean object; Boolean body (for speed optimization)
    def _sense_object_all_directions(self, epicenter, object_pos):
        object_direction_bool_list = np.zeros(8) # direction 0 to 7, value 0 means not found, 1 means found
        
        ### Detect object position ###
        # Detect vertical line
        if epicenter[0] == object_pos[0]: 
            if epicenter[1] > object_pos[1]:
                object_direction_bool_list[0] = 1
            else:
                object_direction_bool_list[4] = 1
            return object_direction_bool_list
                    
        # Calculate Gradient
        gradient = self._calculate_gradient(epicenter, object_pos)
        
        # Positive gradient \ slope
        if gradient == 1.0: 
            if epicenter[1] > object_pos[1]: 
                object_direction_bool_list[7] = 1
            else:
                object_direction_bool_list[3] = 1
            return object_direction_bool_list
                
        # Negative gradient / slope
        if gradient == -1.0: # 
            if epicenter[0] > object_pos[0]: # 
                object_direction_bool_list[5] = 1
            else:
                object_direction_bool_list[1] = 1
            return object_direction_bool_list
                
        # Horizontal line
        if gradient == 0.0: 
            if epicenter[0] > object_pos[0]:
                object_direction_bool_list[6] = 1
            else:
                object_direction_bool_list[2] = 1
            return object_direction_bool_list

        # If nothing found, still return 8 zeros    
        return object_direction_bool_list
    
    # Implemented closely to match Chrispresso's snake
    @staticmethod
    @jit(nopython=True)
    def _simple_distance_two_points(point_1, point_2):
        difference = (point_1[0] - point_2[0], point_1[1] - point_2[1])
        distance = 1.0 / max(abs(difference[0]), abs(difference[1]))

        return distance

    #@staticmethod 
    #@jit(nopython=True) #?for some reason with _calculate_gradient it's slower with @jit
    def _calculate_gradient(self, vector_1, vector_2):
        return (vector_1[1] - vector_2[1]) / (vector_1[0] - vector_2[0])  # (0,0) starts from top left, not bottom left.
        

#? Optimization Statistics (12 Sep 2020)
# [5000 steps trial]
# Old sensing mechanism time taken: 18.92, 19.05, 19.17
# New sensing mechanism time taken: 2.45, 2.50, 2.35

# [20000 steps trial]
# New sensing mechanism time taken: 9.00, 9.51, 8.79
# Numba optimized (_simple_distance_two_points): 7.46, 7.52, 7.53
# Numba optimized (_simple_distance_two_points, _calculate_gradient): 7.84, 7.95, 7.71 #?for some reason with _calculate_gradient it's slower
# Numba optimized (_simple_distance_two_points, _locate_diagonal_walls): 7.30, 7.29, 7.61


#? Human player Test bed
def test_main():
    import keyboard
    import time
    env = SnakeEnv(render=True,segment_width=20, width=8, height=8)
    env.reset()
    env.init_window()
    env.render()
    done = False
    fps = 5

    while not done:
        time.sleep(1/fps)
        env.render()
        action = -1
        while action < 0:
            if keyboard.is_pressed('up'):
                action = 0
            elif keyboard.is_pressed('right'):
                action = 1
            elif keyboard.is_pressed('down'):
                action = 2
            elif keyboard.is_pressed('left'):
                action = 3
            env.render()

        #print("action taken:", action)


        observations, rewards, done, _ = env.step(action)
        env.render()

        #print("observations:",observations)         # Observations: Wall, body, apple
        print("done state:", done)

        if done:
            return

#? Test bed only
if __name__ == "__main__":
    import sys
    test_main()
    sys.exit(1)
    import gym
    import gym_snake3
    import time

    env = gym.make('snake3-v0', render=True, segment_width=25, width=12, height=12, randomness_seed=1)
    env.reset()

    # Performance testing
    '''
    start_time = time.time()
    for _ in range(0,20000):
        observation, reward, done, info = env.step(0)
        observation, reward, done, info = env.step(1)
        observation, reward, done, info = env.step(2)
        observation, reward, done, info = env.step(3)
    elapsed_time = time.time() - start_time
    print("[%] Total elapsed time:", elapsed_time)
    '''
    #print("observation:", observation)

    #env.init_window()
    #env.render()
    #input()