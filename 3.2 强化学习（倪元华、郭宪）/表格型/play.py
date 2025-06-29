import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="4x4", render_mode='human')
observation, info = env.reset()
env.render()
done = False
while not done:
    # 随机采样
    action = env.action_space.sample()
    # 执行动作并获取结果
    observation, reward, terminated, truncated, info = env.step(action)
    # 渲染新状态
    env.render()
    # 检查结束
    done = terminated or truncated
env.close()