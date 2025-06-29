import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import ale_py
import numpy as np
import os
from collections import deque
from frame_stack_wrapper import FrameStackWrapper
from tqdm import tqdm
import matplotlib.pyplot as plt

# 参数配置
GAME = 'AirRaid'
GAMMA = 0.99  # 折扣因子
OBSERVE = 50000  # 观察步数，用于填充经验池
EXPLORE = 1000000  # 探索步数，用于逐步减少ε
INITIAL_EPSILON = 1.0  # 初始ε
FINAL_EPSILON = 0.01  # 最小ε
REPLAY_MEMORY = 1000000  # 经验回放缓冲区大小
BATCH = 64  # 批量大小
FRAME_SKIP = 2  # 帧跳过数
SAVE_INTERVAL = 10000  # 每轮步数
POLICY_FILE = None  # "policy\policy_20000.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络层（He初始化）
def layer_init(layer, bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# DQN
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions))
        )
    def forward(self, x):
        x = x.float() / 255.0
        return self.net(x)

# 经验回放缓冲区
class ExperienceBuffer:
    def __init__(self, capacity=REPLAY_MEMORY):
        self.buffer = deque(maxlen=capacity)
    # 放入经验
    def add(self, experience):
        self.buffer.append(experience)
    # 采样
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(DEVICE)
        )

# 智能体
class Agent:
    def __init__(self, policy_file=None):
        self.n_actions = gym.make("ALE/AirRaid-v5").action_space.n
        self.policy_net = DQN(self.n_actions).to(DEVICE)
        self.target_net = DQN(self.n_actions).to(DEVICE)
        self.steps = 0
        self.epsilon = INITIAL_EPSILON
        self.target_update = SAVE_INTERVAL  # 目标网络更新间隔
        self.loss_history = []  # 损失值记录
        self.steps_history = []  # 步数记录
        if policy_file:  # 加载策略
            self.load_policy(policy_file)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = RMSprop(self.policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)  # RMSprop优化器
        self.buffer = ExperienceBuffer()
    # 加载策略（不会加载经验池）
    def load_policy(self, file_path):
        if os.path.exists(file_path):
            self.policy_net.load_state_dict(torch.load(file_path, map_location=DEVICE))
            self.steps = int(file_path.split('_')[-1].split('.')[0])
            print(f"加载{file_path}")
        else:
            print(f"没有{file_path}")
    # epsilon衰减
    def decay_epsilon(self):
        if self.steps > OBSERVE:
            decay_steps = min(self.steps - OBSERVE, EXPLORE)
            self.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * decay_steps / EXPLORE)
    # 更新网络
    def update_network(self):
        if len(self.buffer.buffer) < BATCH:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH)
        # 双Q
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * GAMMA * next_q
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(-1))
        loss = F.smooth_l1_loss(current_q, target_q)  # Huber损失
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.optimizer.step()
        # 定期更新目标网络
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # 记录损失和步数
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.steps_history.append(self.steps)
        return loss_value
    # 保存网络
    def save_model(self, progress_bar):
        filepath = f"policy/policy_{self.steps}.pth"
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"\n已保存{filepath}")
        progress_bar.reset()

if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make("ALE/AirRaid-v5", render_mode=None, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=FRAME_SKIP, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = FrameStackWrapper(env, num_stack=4)  # 堆叠4帧
    # 初始化智能体
    agent = Agent(POLICY_FILE)
    state, _ = env.reset()
    current_episode_reward = 0
    progress_bar = tqdm(total=SAVE_INTERVAL, unit="steps")
    try:
        while True:
            # ε-greedy
            if np.random.rand() < agent.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    q_values = agent.policy_net(state_t)
                    action = q_values.argmax().item()
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 存储经验
            agent.buffer.add((state, action, reward, next_state, done))
            state = next_state
            current_episode_reward += reward  # 累计当前回合奖励
            # 游戏结束时输出分数
            if done:
                print(f"\n本局得分：{current_episode_reward}")
                current_episode_reward = 0
                state, _ = env.reset()
            else:
                state = next_state
            agent.steps += 1
            progress_bar.update(1)
            # 开始训练
            if agent.steps > OBSERVE:
                agent.update_network()
                agent.decay_epsilon()
            # 保存模型
            if agent.steps % SAVE_INTERVAL == 0:
                agent.save_model(progress_bar)
                progress_bar.reset()
    except KeyboardInterrupt:
        print("中断")
        # 绘制损失曲线
        if len(agent.loss_history) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(agent.steps_history, agent.loss_history, alpha=0.6)
            plt.xlabel('步数')
            plt.ylabel('损失值')
            plt.title('DQN损失变化曲线')
            plt.grid(True)
            plt.savefig('training_loss_curve.png')
    finally:
        env.close()
        progress_bar.close()