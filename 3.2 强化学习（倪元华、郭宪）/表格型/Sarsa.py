import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class FrozenLake:
    def __init__(self, env):
        self.env = env
        self.obs_n = env.observation_space.n  # 状态空间大小
        self.act_n = env.action_space.n  # 动作空间大小
        self.qvalue = np.zeros((self.obs_n, self.act_n))  # 状态-动作价值函数
        self.gamma = 0.9  # 折扣率
        self.alpha = 0.1  # 学习率
        self.epsilon = 0.1  # 探索概率

    # epsilon-greedy（多最优随机选择）
    def sample_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.act_n)
        else:
            max_actions = np.where(self.qvalue[state] == np.max(self.qvalue[state]))[0]
            action = np.random.choice(max_actions)
        return action

    # 贪婪决策（多最优随机选择）
    def get_greedy_policy(self):
        greedy_policy = np.zeros(self.obs_n)
        for i in range(self.obs_n):
            max_actions = np.where(self.qvalue[i] == np.max(self.qvalue[i]))[0]
            greedy_policy[i] = np.random.choice(max_actions)
        return greedy_policy

    # SARSA
    def sarsa(self, episodes):
        start_time = time.time()
        for episode in range(episodes):
            state = self.env.reset()[0]
            action = self.sample_action(state)
            flag = False  # 终止或截断标签
            while not flag:
                # 选取动作
                next_state, reward, flag, _, _ = self.env.step(action)
                next_action = self.sample_action(next_state)
                # q值更新
                td_target = reward + self.gamma * self.qvalue[next_state, next_action]
                td_error = td_target - self.qvalue[state, action]
                self.qvalue[state, action] += self.alpha * td_error
                state = next_state
                action = next_action
        end_time = time.time()
        print(f"总运行时间:{end_time - start_time:.2f}秒")

    # 价值函数热力图
    def visualize_q(self):
        plt.figure(figsize=(10, 6))
        plt.title("Final Q-values")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.imshow(self.qvalue, cmap='viridis', aspect='auto')
        plt.colorbar(label='Q-value')
        plt.show()

    # 策略可视化
    def visualize_policy(self, policy):
        # 环境信息
        desc = self.env.unwrapped.desc.astype(str)
        n_rows = desc.shape[0]
        n_cols = desc.shape[1]
        # 绘图
        fig, ax = plt.subplots(figsize=(n_cols, n_rows))
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_xticks(range(n_cols + 1))
        ax.set_yticks(range(n_rows + 1))
        ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=1)
        # 绘制策略箭头
        for row in range(n_rows):
            for col in range(n_cols):
                state = row * n_cols + col
                policy_action = policy[state]
                x_center = col + 0.5
                y_center = row + 0.5
                if policy_action == 0:  # left
                    ax.arrow(x_center + 0.25, y_center, -0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                elif policy_action == 1:  # down
                    ax.arrow(x_center, y_center - 0.25, 0, 0.3, head_width=0.1, head_length=0.1, fc='green', ec='green')
                elif policy_action == 2:  # right
                    ax.arrow(x_center - 0.25, y_center, 0.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
                elif policy_action == 3:  # up
                    ax.arrow(x_center, y_center + 0.25, 0, -0.3, head_width=0.1, head_length=0.1, fc='yellow', ec='yellow')
        # 标注陷阱 (红色背景)、目标 (绿色背景)
        holes = [(r, c) for r in range(desc.shape[0]) for c in range(desc.shape[1]) if desc[r, c] == 'H']
        goal = [(r, c) for r in range(desc.shape[0]) for c in range(desc.shape[1]) if desc[r, c] == 'G']
        for hole in holes:
            row, col = hole
            rect = Rectangle((col, row), 1, 1, facecolor="red", alpha=0.5, edgecolor="black")
            ax.add_patch(rect)
        for g in goal:
            row, col = g
            rect = Rectangle((col, row), 1, 1, facecolor="green", alpha=0.5, edgecolor="black")
            ax.add_patch(rect)
        plt.title("Policy Visualization")
        plt.show()

    # 评估策略
    def evaluate_policy(self, policy, num_episodes):
        success_count = 0
        for _ in range(num_episodes):
            state = self.env.reset()[0]
            flag = False
            while not flag:
                action = policy[state]
                next_state, reward, flag, _, _ = self.env.step(action)
                state = next_state
                if reward == 1:
                    success_count += 1
                    break
        success_rate = success_count / num_episodes
        print(f"策略成功率: {success_rate * 100:.2f}%")
        return success_rate

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", map_name="4x4", render_mode='ansi', is_slippery=True)
    frozen_lake = FrozenLake(env)
    frozen_lake.sarsa(10000)
    greedy_policy = frozen_lake.get_greedy_policy()
    frozen_lake.visualize_q()
    frozen_lake.visualize_policy(greedy_policy)
    frozen_lake.evaluate_policy(greedy_policy, 1000)
'''
if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", map_name="4x4", render_mode='ansi', is_slippery=True)
    success_rates = []
    run_times = []
    num_experiments = 10
    for experiment in range(num_experiments):
        print(f"\nExperiment {experiment + 1}/{num_experiments}")
        frozen_lake = FrozenLake(env)
        start_time = time.time()
        frozen_lake.sarsa(10000)
        end_time = time.time()
        success_rate = frozen_lake.evaluate_policy(frozen_lake.get_greedy_policy(), 1000)
        if success_rate > 0:
            success_rates.append(success_rate)
            run_times.append(end_time - start_time)
    avg_success_rate = sum(success_rates) / len(success_rates)
    avg_run_time = sum(run_times) / len(run_times)
    print(f"\n平均策略成功率: {avg_success_rate * 100:.2f}%")
    print(f"平均耗时: {avg_run_time:.2f}秒")
'''