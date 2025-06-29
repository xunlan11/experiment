import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import os
import time
import random


# 神经网络层初始化
def layer_init(layer, std=np.sqrt(2), gain=1, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std * gain) # 正交初始化权重
    torch.nn.init.constant_(layer.bias, bias_const) # 常数初始化偏置
    return layer


# Actor网络：连续动作策略网络
class Actor_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super(Actor_Net, self).__init__()
        self.act_limit = act_limit
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], act_dim), gain=0.01),
            nn.Tanh()
        )
    # 前向传播
    def forward(self, obs):
        return self.act_limit * self.actor_net(obs)
    # 获取动作
    def get_a(self, obs):
        with torch.no_grad():
            return self.forward(obs).cpu().numpy()


# Critic网络：动作价值网络
class Critic_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super(Critic_Net, self).__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim + act_dim, hidden_sizes[0]))
        self.fc2 = layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.fc3 = layer_init(nn.Linear(hidden_sizes[1], 1))
    # 输出层不使用激活函数，直接输出Q值
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    # 添加经验到缓冲区
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 从缓冲区采样
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done).astype(float)).unsqueeze(1)
        return state, action, reward, next_state, done
    # 缓冲区长度
    def __len__(self):
        return len(self.buffer)


# OU噪声，用于探索
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    # 重置噪声状态
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    # 采样噪声
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# DDPG
class DDPG:
    def __init__(self, env, epochs):
        # 环境
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0] # 动作限制
        # 网络
        hidden_sizes = [256, 256] # 隐藏层大小
        self.actor = Actor_Net(self.obs_dim, self.act_dim, hidden_sizes, self.act_limit)
        self.critic = Critic_Net(self.obs_dim, self.act_dim, hidden_sizes)
        self.actor_target = Actor_Net(self.obs_dim, self.act_dim, hidden_sizes, self.act_limit)
        self.critic_target = Critic_Net(self.obs_dim, self.act_dim, hidden_sizes)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 学习率与优化器
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.lr_decay_rate = 0.995 # 衰减率
        self.min_lr = 1e-6 # 最小学习率
        # 其他超参数
        self.epochs = epochs # 训练轮数
        self.save_freq = 100 # 模型保存频率
        self.gamma = 0.99 # 折扣因子
        self.tau = 0.005 # 软更新参数
        self.batch_size = 64 # 批次大小
        self.steps_per_epoch = 1000 # 每个epoch的步数
        self.max_ep_len = 1000 # 每个回合最大步数
        self.start_steps = 10000 # 开始学习前的随机动作步数
        self.update_after = 1000 # 开始更新的步数
        self.update_every = 50 # 更新频率
        self.noise_sigma = 0.1 # 噪声标准差
        self.buffer_size = 100000 # 经验回放缓冲区大小
        self.noise = OUNoise(self.act_dim, sigma=self.noise_sigma) # 噪声
        self.replay_buffer = ReplayBuffer(self.buffer_size) # 经验回放
        self.return_traj = []
        self.critic_losses = []
        self.actor_losses = []
        self.best_return = -float('inf')
        self.total_steps = 0
        # 模型保存
        self.training_path = r'lunarlander_ddpg'
        os.makedirs(self.training_path, exist_ok=True)
        self.best_model_path = None
    # 更新目标网络参数
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    # 获取带噪声的动作
    def get_action(self, obs, noise=True):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs).numpy()
        # 添加噪声用于探索
        if noise:
            action += self.noise.sample() * self.act_limit
            action = np.clip(action, -self.act_limit, self.act_limit)
        return action
    # 更新网络
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从经验回放中采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        # 更新Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 记录损失
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    # 学习率衰减
    def decay_learning_rate(self):
        for param_group in self.actor_optimizer.param_groups:
            new_lr = max(self.min_lr, param_group['lr'] * self.lr_decay_rate)
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            new_lr = max(self.min_lr, param_group['lr'] * self.lr_decay_rate)
            param_group['lr'] = new_lr
    # 训练
    def ddpg_train(self):
        start_time = time.time()
        best_avg_return = -float('inf')
        state, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        for epoch in range(self.epochs):
            print(f"训练{epoch + 1}轮")
            epoch_rewards = []
            for t in range(self.steps_per_epoch):
                self.total_steps += 1
                # 选择动作，最开始使用随机动作
                if self.total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state)
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                self.replay_buffer.push(state, action, reward, next_state, done) # 存储经验
                state = next_state
                # 回合结束或达到最大步数
                if done or episode_steps >= self.max_ep_len:
                    epoch_rewards.append(episode_reward)
                    self.return_traj.append(episode_reward)
                    # 重置
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    self.noise.reset()
                # 更新网络
                if self.total_steps >= self.update_after and self.total_steps % self.update_every == 0:
                    for _ in range(self.update_every):
                        self.update()
            # 计算当前epoch的平均回报
            avg_return = np.mean(epoch_rewards) if epoch_rewards else 0
            print(f"平均回报: {avg_return:.2f}")
            # 更新最佳回报并保存
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                self.best_return = avg_return
                best_actor_path = os.path.join(self.training_path, 'best_actor.pth')
                best_critic_path = os.path.join(self.training_path, 'best_critic.pth')
                torch.save(self.actor.state_dict(), best_actor_path)
                torch.save(self.critic.state_dict(), best_critic_path)
                self.best_model_path = best_actor_path
                print(f"已保存更好模型，平均回报: {avg_return:.2f}")
            # 定期保存
            if (epoch + 1) % self.save_freq == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.training_path, f'{epoch + 1}_actor.pth'))
            # 学习率衰减
            self.decay_learning_rate()
        end_time = time.time()
        print(f"\n训练耗时: {end_time - start_time:.2f}秒")
        self.plot_curves()
    # 绘制曲线
    def plot_curves(self):
        # 回报曲线
        _, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.plot(self.return_traj, 'b-', alpha=0.7, label='Episode Reward')
        # 滑动平均
        if len(self.return_traj) >= 20:
            window = min(50, len(self.return_traj)//10)
            moving_avg = np.convolve(self.return_traj, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.return_traj)), moving_avg, 'r-', linewidth=2, label=f'Moving Average({window})')
        ax1.axhline(y=self.best_return, color='orange', linestyle='--', label=f'Best Reward: {self.best_return:.2f}') # 最佳回报水平线
        ax1.set_title('Training Performance')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.training_path, 'reward_curve.png'), dpi=300)
        # 损失曲线
        _, axes = plt.subplots(2, 1, figsize=(12, 8))
        # Actor损失
        if self.actor_losses:
            axes[0].plot(self.actor_losses, 'r-')
            axes[0].set_title('Actor Loss')
            axes[0].set_xlabel('Updates')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
        # Critic损失
        if self.critic_losses:
            axes[1].plot(self.critic_losses, 'b-')
            axes[1].set_title('Critic Loss')
            axes[1].set_xlabel('Updates')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.training_path, 'loss_curves.png'), dpi=300)


if __name__ == '__main__':
    env = gym.make('LunarLander-v3', continuous=True)
    ddpg = DDPG(env, epochs=1000)
    ddpg.ddpg_train()
    env.close()