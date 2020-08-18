from gym.envs.registration import register

register(id='snake3-v0',
         entry_point='gym_snake3.envs:SnakeEnv'
         )
