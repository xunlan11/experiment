import gymnasium as gym
import numpy as np

GAMMA = 0.9   # 折扣率
THETA = 1e-6  # 迭代终止阈值

# "FrozenLake-v1"环境
env = gym.make("FrozenLake-v1", map_name="8x8", render_mode='ansi')
env_unwrapped = env.unwrapped
obs_n = env_unwrapped.observation_space.n  # 状态空间元素数
act_n = env_unwrapped.action_space.n       # 动作空间元素数
P = env_unwrapped.P                        # 转移概率矩阵

# 值迭代
def value_iteration():
    V = np.zeros(obs_n)
    iters = 0
    while True:
        delta = 0
        # 遍历所有状态
        for s in range(obs_n):
            v = V[s]
            action_values = []
            # 计算动作价值
            for a in range(act_n):
                action_values.append(sum(p * (r + GAMMA * V[s_]) for p, s_, r, _ in P[s][a]))
            # 更新值函数
            V[s] = max(action_values) if action_values else 0
            # 更新本轮最大变化量
            delta = max(delta, abs(v - V[s]))
        iters += 1
        if delta < THETA:
            break
    # 最优策略
    policy = np.zeros(obs_n)
    for s in range(obs_n):
        action_values = []
        for a in range(act_n):
            action_values.append(sum(p * (r + GAMMA * V[s_]) for p, s_, r, _ in P[s][a]))
        best_action = np.argmax(action_values) if action_values else 0
        policy[s] = best_action
    return policy, V, iters

# 策略评估
def policy_evaluation(policy):
    V = np.zeros(obs_n)
    eval_iters = 0
    while True:
        delta = 0
        # 遍历所有状态
        for s in range(obs_n):
            v = V[s]
            # 计算值函数
            V[s] = sum(p * (r + GAMMA * V[s_]) for p, s_, r, _ in P[s][policy[s]])
            # 更新本轮最大变化量
            delta = max(delta, abs(v - V[s]))
        eval_iters += 1
        if delta < THETA:
            break
    return V, eval_iters

# 策略改进
def policy_improvement(V):
    policy = np.zeros(obs_n)
    for s in range(obs_n):
        action_values = []
        # 计算动作价值
        for a in range(act_n):
            action_values.append(sum(p * (r + GAMMA * V[s_]) for p, s_, r, _ in P[s][a]))
        # 贪婪策略
        best_action = np.argmax(action_values) if action_values else 0
        policy[s] = best_action
    return policy

# 策略迭代
def policy_iteration():
    iters = 0
    total_eval_iters = 0
    # 初始化策略为随机动作
    policy = np.zeros(obs_n)
    for s in range(obs_n):
        policy[s] = env_unwrapped.action_space.sample()
    while True:
        # 策略评估
        V, eval_iters = policy_evaluation(policy)
        total_eval_iters += eval_iters
        # 策略改进
        new_policy = policy_improvement(V)
        iters += 1
        # 不再更新则停止迭代
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    return policy, V, iters, total_eval_iters

if __name__ == "__main__":
    # 值迭代
    print("===值迭代===")
    vi_policy, vi_value, vi_iters = value_iteration()
    print(f"最优策略：{vi_policy}")
    print(f"最优值函数：{vi_value}")
    print(f"迭代轮数：{vi_iters}")
    # 策略迭代
    print("===策略迭代===")
    pi_policy, pi_value, pi_iters, total_pi_eval_iters = policy_iteration()
    print(f"最优策略：{pi_policy}")
    print(f"最优值函数：{pi_value}")
    print(f"迭代轮数：{pi_iters}")
    print(f"总策略评估次数：{total_pi_eval_iters}")