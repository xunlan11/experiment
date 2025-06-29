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
        self.n = np.zeros((self.obs_n, self.act_n)) # 状态-动作计数器
        self.gamma = 0.9  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.9  # 探索率衰减速度
        self.epsilon_min = 0.01  # 探索率最小值
        self.Pi = np.full((self.obs_n, self.act_n), 1 / self.act_n) # 均匀初始化策略

    # 重置环境
    def reset(self):
        self.qvalue = np.zeros((self.obs_n, self.act_n))
        self.n = np.zeros((self.obs_n, self.act_n))
        self.epsilon = 1.0

    # 策略更新
    def update_epsilon_greedy(self):
        for i in range(self.obs_n):
            self.Pi[i, :] = self.epsilon / self.act_n
            max_num = np.argmax(self.qvalue[i, :])
            self.Pi[i, max_num] = self.epsilon / self.act_n + (1 - self.epsilon)

    # 贪婪决策（多最优随机选择）
    def get_greedy_policy(self):
        greedy_policy = np.zeros(self.obs_n)
        for i in range(self.obs_n):
            max_actions = np.where(self.qvalue[i] == np.max(self.qvalue[i]))[0]
            greedy_policy[i] = np.random.choice(max_actions)
        return greedy_policy

    # On_policy
    def On_policy(self, num_episodes):
        start_time = time.time()
        for episode in range(num_episodes):
            flag = False # 终止或截断标签
            # 采样一条轨迹
            state_traj = []
            action_traj = []
            reward_traj = []
            g = 0
            self.env.reset()
            cur_state = 0
            while not flag:
                # 根据策略pi采样一个动作
                cur_action = np.random.choice(self.act_n, p=self.Pi[cur_state, :])
                state_traj.append(cur_state)
                action_traj.append(cur_action)
                # 更新状态
                next_state, reward, flag, _, _ = self.env.step(cur_action)
                cur_state = next_state
                reward_traj.append(reward)
            # 利用轨迹更新行为值函数
            for i in reversed(range(len(state_traj))):
                # 计算状态-动作对(s,a)的访问频次
                self.n[state_traj[i], action_traj[i]] += 1.0
                # 增量更新当前状态动作值
                g = reward_traj[i] + self.gamma * g
                # 新Q值=[当前Q值*(访问次数-1)（代表累积经验）+新折扣累积奖励g]/访问次数
                self.qvalue[state_traj[i], action_traj[i]] = (
                    self.qvalue[state_traj[i], action_traj[i]] * (self.n[state_traj[i], action_traj[i]] - 1) + g
                ) / self.n[state_traj[i], action_traj[i]]
            # 更新策略
            if episode % 100 == 0:
                self.update_epsilon_greedy()
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
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
        desc = env.unwrapped.desc.astype(str)
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
                if policy_action == 0:    # left
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
    env = gym.make('FrozenLake-v1', map_name="4x4", render_mode='ansi')
    frozen_lake = FrozenLake(env)
    frozen_lake.On_policy(10000)
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
        frozen_lake.On_policy(10000)
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