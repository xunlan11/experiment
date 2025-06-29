import gymnasium as gym
import numpy as np

# 帧堆叠
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self._obs_shape = env.observation_space.shape  # 记录原始观测形状
        self.frames = np.zeros((self.num_stack, *self._obs_shape), dtype=np.uint8)  # 预分配固定内存数组
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.num_stack, *self._obs_shape), dtype=np.uint8)  # 调整观测空间

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = obs  # 用初始观测填充所有帧
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 滚动更新帧
        self.frames[:-1] = self.frames[1:]  # 左移旧帧
        self.frames[-1] = obs  # 插入新帧
        return self.frames.copy(), reward, terminated, truncated, info