import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import os
import time


# 采样
class Sample():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99 # 折扣因子
        self.no_penalty_steps = 200 # 步数惩罚阈值，只作用于训练
        self.batch_state = None
        self.batch_act = None
        self.batch_val_target = None
        self.index = None
        self.sum_return = 0
    # 采样1条轨迹
    def sample_one_episode(self, actor_net):
        val_target = 0.0
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_val_target = []
        cur_obs, _ = self.env.reset() # 重置环境并获取初始观测
        done = False
        episode_sum = 0
        step_count = 0
        while not done:
            episode_obs.append(cur_obs)
            action, _ = actor_net.get_a(torch.as_tensor(cur_obs, dtype=torch.float32))
            episode_actions.append(action)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            # 动态时间惩罚，超过阈值步数后，惩罚逐步增加
            if step_count > self.no_penalty_steps:
                time_penalty = -0.01 * (1 + (step_count - self.no_penalty_steps) * 0.01)
                reward += time_penalty
            cur_obs = next_obs
            episode_rewards.append(reward)
            step_count += 1
            # 回报
        val_target = 0.0
        for t in reversed(range(0, len(episode_rewards))):
            val_target = episode_rewards[t] + val_target * self.gamma
            episode_val_target.insert(0, val_target)  # 插入表头
        episode_sum = sum(episode_rewards)
        episode_obs = np.array(episode_obs)
        episode_actions = np.array(episode_actions).reshape(-1, 1)
        episode_val_target = np.array(episode_val_target).reshape(-1, 1)
        return episode_obs, episode_actions, episode_val_target, episode_sum
    # 采样多条轨迹，并批次化数据
    def sample_many_episodes(self, actor_net, num):
        self.sum_return = 0
        all_states, all_acts, all_val_targets = [], [], []
        for i in range(num):
            episode_state, episode_act, episode_val_target, episode_sum = self.sample_one_episode(actor_net)
            all_states.append(episode_state)
            all_acts.append(episode_act)
            all_val_targets.append(episode_val_target)
            self.sum_return += episode_sum
        self.batch_state = np.concatenate(all_states, 0)
        self.batch_act = np.concatenate(all_acts, 0)
        self.batch_val_target = np.concatenate(all_val_targets, 0)
    # 获取mini-batc数据
    def get_data(self, start_index, sgd_num):
        sgd_batch_state = self.batch_state[self.index[start_index:start_index + sgd_num]]
        sgd_batch_act = self.batch_act[self.index[start_index:start_index + sgd_num]]
        sgd_batch_val_target = self.batch_val_target[self.index[start_index:start_index + sgd_num]]
        # 归一化优势
        if sgd_batch_val_target.std() > 1e-8:
            sgd_batch_val = (sgd_batch_val_target - sgd_batch_val_target.mean()) / (sgd_batch_val_target.std() + 1e-8)
        else:
            sgd_batch_val = sgd_batch_val_target - sgd_batch_val_target.mean()
        sgd_batch_state = torch.as_tensor(sgd_batch_state, dtype=torch.float32)
        sgd_batch_act = torch.as_tensor(sgd_batch_act, dtype=torch.long)
        sgd_batch_val = torch.as_tensor(sgd_batch_val, dtype=torch.float32)
        return sgd_batch_state, sgd_batch_act, sgd_batch_val


# 神经网络层初始化
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std) # 正交初始化权重
    torch.nn.init.constant_(layer.bias, bias_const) # 常数初始化偏置
    return layer


# 策略网络：全连接网络
class Actor_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super(Actor_Net, self).__init__()
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], act_dim), std=0.01)
        )
    # 计算分布
    def distribution(self, obs):
        logits = self.actor_net(obs)
        return Categorical(logits=logits)
    # 前向传播
    def forward(self, obs, act):
        dist = self.distribution(obs)
        logp_a = dist.log_prob(act.squeeze(-1)) # 批状态策略对数概率
        return dist, logp_a, dist.entropy()
    # 采样动作，不计算梯度
    def get_a(self, obs):
        with torch.no_grad():
            dist = self.distribution(obs)
            act = dist.sample()
            log_a = dist.log_prob(act) # 单状态对数概率
        return act.item(), log_a.item()


# ReinForce
class ReinForce():
    def __init__(self, env, epochs=5000):
        # 环境
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        # 网络
        self.hidden = [128, 128]
        self.actor = Actor_Net(self.obs_dim, self.act_dim, self.hidden)
        # 学习率与优化器
        self.pi_lr = 0.0003
        self.lr_decay_rate = 0.98
        self.min_lr = 0.00003
        self.pi_optimizer = Adam(self.actor.parameters(), self.pi_lr)
        # 其他超参数
        self.epochs = epochs # 训练轮数
        self.train_pi_iters = 5 # 网络训练迭代次数
        self.entropy_coeff = 0.01 # 熵正则化系数
        self.sgd_num = 256 # 批大小
        self.save_freq = 100 # 模型保存频率
        self.episodes_num = 10 # 每轮采样的轨迹数
        self.max_grad_norm = 0.5 # 梯度裁剪阈值
        self.sampler = Sample(env)
        self.return_traj = []
        self.policy_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.best_return = -float('inf')
        # 模型保存
        self.training_path = 'lunarlander_RF'
        os.makedirs(self.training_path, exist_ok=True)
        self.best_model_path = None
    # 策略损失
    def compute_loss_pi(self, obs, act, value):
        _, logp, entropy = self.actor(obs, act)
        num = obs.size()[0]
        logp = logp.reshape(num, 1)
        loss_pi = -(logp * value).mean() # 策略梯度损失
        loss_entropy = entropy.mean() # 熵正则化
        return loss_pi, loss_entropy
    # 指数衰减学习率
    def update_learning_rate(self, epoch):
        new_lr = max(self.pi_lr * (self.lr_decay_rate ** (epoch // 50)), self.min_lr)
        for param_group in self.pi_optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    # 采集数据并更新
    def update(self):
        # 采集数据
        self.sampler.sample_many_episodes(self.actor, self.episodes_num)
        avg_return = self.sampler.sum_return / self.episodes_num
        self.return_traj.append(avg_return)
        print(f"平均回报{avg_return:.2f}")
        batch_size = self.sampler.batch_state.shape[0]
        self.sampler.index = np.arange(batch_size)
        sum_pi_loss = 0.0
        sum_entropy_loss = 0.0
        sum_total_loss = 0.0
        update_count = 0
        # 训练
        for i in range(self.train_pi_iters):
            np.random.shuffle(self.sampler.index)
            for start_index in range(0, batch_size - self.sgd_num, self.sgd_num):
                batch_state, batch_act, batch_val_target = self.sampler.get_data(start_index, self.sgd_num) # 采集mini批数据
                self.pi_optimizer.zero_grad() # 清除梯度值
                loss_pi, loss_entropy = self.compute_loss_pi(batch_state, batch_act, batch_val_target) # 损失
                loss = loss_pi - self.entropy_coeff * loss_entropy # 熵正则化以鼓励探索
                loss.backward() # 反向传播
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # 梯度裁剪
                self.pi_optimizer.step() # 并更新参数
                # 累加损失
                sum_pi_loss += loss_pi.item()
                sum_entropy_loss += loss_entropy.item()
                sum_total_loss += loss.item()
                update_count += 1
        # 平均损失
        avg_pi_loss = sum_pi_loss / update_count if update_count > 0 else 0
        avg_entropy_loss = sum_entropy_loss / update_count if update_count > 0 else 0
        avg_total_loss = sum_total_loss / update_count if update_count > 0 else 0
        self.policy_losses.append(avg_pi_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.total_losses.append(avg_total_loss)
    # 训练
    def rf_train(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            current_lr = self.update_learning_rate(epoch) # 更新学习率
            print(f"训练{epoch+1}轮，学习率{current_lr:.6f}")
            self.update()
            # 获取平均回报，保存（最优/一定轮数）
            current_return = self.return_traj[-1]
            if current_return > self.best_return:
                self.best_return = current_return
                best_model_path = os.path.join(self.training_path, 'best_actor.pth')
                torch.save(self.actor.state_dict(), best_model_path)
                self.best_model_path = best_model_path
                print(f"已保存更好模型")
            if (epoch + 1) % self.save_freq == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.training_path, str(epoch + 1) + 'actor.pth'))
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\n训练耗时{training_duration:.2f}秒")
        # 回报曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.return_traj)
        plt.axhline(y=self.best_return, color='r', linestyle='--', label=f'Best Reward: {self.best_return:.2f}')
        plt.title('Training Performance')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.training_path, 'reward_curve.png'))
        # 损失曲线
        plt.figure(figsize=(12, 8))
        # 策略损失
        plt.subplot(3, 1, 1)
        plt.plot(self.policy_losses)
        plt.title('Policy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        # 熵
        plt.subplot(3, 1, 2)
        plt.plot(self.entropy_losses, color='green')
        plt.title('Entropy Value')
        plt.xlabel('Epochs')
        plt.ylabel('Entropy')
        plt.grid(True)
        # 总损失
        plt.subplot(3, 1, 3)
        plt.plot(self.total_losses, color='orange')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.training_path, 'loss_curves.png'))


if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    lunarlander_pg = ReinForce(env, epochs=5000)
    lunarlander_pg.rf_train()
    env.close()