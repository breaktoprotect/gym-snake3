# Description
My very own Snake game implemented in OpenAI Gym. There are many out there, but there's none like this. :D
I'm a control freak so I like to have control over the environment itself, thus created this for education and flexibility.

This is the version v3. Snake v1 returns a set of observables and Snake v2 returns the entire environment in list of matrices.
For Snake v3, it will return distances between the Snake head and the relevant objects such as walls, apple or itself in 8 directions so a total of 24 inputs. 

# Updates
## Optimization Statistics (12 Sep 2020)
### 5000 steps trial:
- Old sensing mechanism time taken: 18.92, 19.05, 19.17
- New sensing mechanism time taken: 2.45, 2.50, 2.35

### 20000 steps trial:
- New sensing mechanism time taken: 9.00, 9.51, 8.79
- Numba optimized (_simple_distance_two_points): 7.46, 7.52, 7.53
- Numba optimized (_simple_distance_two_points, _calculate_gradient): 7.84, 7.95, 7.71 (for some reason with _calculate_gradient it's slower)
- Numba optimized (_simple_distance_two_points, _locate_diagonal_walls): 7.30, 7.29, 7.61

# Details
## How does the game works?
Standard Snake game. If you owned a Nokia phone ever, you would have played this game. 

## Demo: Randomized agent to show environment and 'matrices' returned 
![Simple demo of v3](INSERT_LINK_HERE)

## Observations
8 directions - Up, Top-Right, Right, Bottom-Right, Down, Bottom-Left, Left, Top-Left
For each directions, 3 indicators for: wall, apple and Snake's body (itself)

## Actions
0 - Up
1 - Right
2 - Down
3 - Left

Pressing the movement while in the same direction yields no action. 
For example, if the snake is facing North, press Up will effectively do nothing.

## Reward System
* +1 point for consuming apple

## Starting State:
Snake starts at a random position within a given boundary
Apple appears at a random position throughout accessible positions in the environment. Changes position whenever it gets eaten.

## Episode Termination:
* Death by starvation - A total of N health. To refill full health, consume apple. When health is 0, game ends
* Crashes into a wall or its own body
