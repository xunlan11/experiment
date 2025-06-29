import time
import random
import numpy as np
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0",render_mode="human")
env.reset()
while(1):
    env.render()
    env.step(env.action_space.sample())