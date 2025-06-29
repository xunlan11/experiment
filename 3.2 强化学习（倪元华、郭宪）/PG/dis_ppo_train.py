import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
from torch.optim import Adam 
from torch.distributions.categorical import Categorical 
from collections import deque 
import os  
import time 


# 采样类
class Sample:  
    def __init__(self, env):  
        self.env = env 
        self.gamma = 0.99 # 折扣因子
        self.lamda = 0.95 # GAE参数
        self.batch_state = None 
        self.batch_act = None 
        self.batch_logp = None # 动作对数概率
        self.batch_val_target = None 
        self.batch_adv = None # 优势值
        self.index = None 
        self.sum_return = 0   
    # 采样1条轨迹    
    def sample_one_episode(self, actor_net, critic_net): 
        episode_obs = [] 
        episode_act = [] 
        episode_logp = [] 
        episode_rewards = []  
        episode_val = []  
        val_target = []
        episode_adv = [] 
        cur_obs, _ = self.env.reset() # 重置环境并获取初始观测
        done = False 
        episode_sum = 0  
        while not done:
            episode_obs.append(cur_obs)  
            obs_tensor = torch.as_tensor(cur_obs, dtype=torch.float32)
            action, logp = actor_net.get_a(obs_tensor)
            value = critic_net.get_v(obs_tensor)
            episode_act.append(action)  
            episode_logp.append(logp) 
            episode_val.append(value) 
            next_obs, reward, terminated, truncated, _ = self.env.step(action) 
            done = terminated or truncated 
            cur_obs = next_obs
            episode_rewards.append(reward) 
            episode_sum += reward  
        # GAE优势
        vals = episode_val + [0] # 末状态价值设为0
        adv = 0 
        for t in reversed(range(len(episode_rewards))):
            delta = episode_rewards[t] + self.gamma * vals[t+1] - vals[t] # TD误差
            adv = delta + self.gamma * self.lamda * adv # GAE优势
            episode_adv.insert(0, adv) # 插入表头
        # 目标值函数
        ret = 0 
        for r in reversed(episode_rewards): 
            ret = r + self.gamma * ret # 折扣回报
            val_target.insert(0, ret) # 插入表头
        return (np.array(episode_obs), np.array(episode_act).reshape(-1, 1), 
                np.array(episode_logp).reshape(-1, 1), np.array(episode_adv).reshape(-1, 1),
                np.array(val_target).reshape(-1, 1), episode_sum) 
    # 采样多条轨迹，并批次化数据
    def sample_many_episodes(self, actor_net, critic_net, num): 
        self.sum_return = 0 
        all_states, all_acts, all_logps, all_advs, all_val_targets = [], [], [], [], [] 
        for i in range(num): 
            episode_state, episode_act, episode_logp, episode_adv, episode_val_target, episode_sum = self.sample_one_episode(actor_net, critic_net)
            all_states.append(episode_state) 
            all_acts.append(episode_act) 
            all_logps.append(episode_logp)  
            all_advs.append(episode_adv)  
            all_val_targets.append(episode_val_target)  
            self.sum_return += episode_sum  
        self.batch_state = np.concatenate(all_states, 0) 
        self.batch_act = np.concatenate(all_acts, 0)  
        self.batch_logp = np.concatenate(all_logps, 0) 
        self.batch_adv = np.concatenate(all_advs, 0)  
        self.batch_val_target = np.concatenate(all_val_targets, 0)   
    # 获取mini-batch数据
    def get_data(self, start_index, sgd_num): 
        idx = self.index[start_index:start_index+sgd_num]  
        sgd_batch_state = self.batch_state[idx]  
        sgd_batch_act = self.batch_act[idx] 
        sgd_batch_logp = self.batch_logp[idx] 
        sgd_batch_adv = self.batch_adv[idx]  
        sgd_batch_val_target = self.batch_val_target[idx] 
        # 归一化优势
        if sgd_batch_adv.std() > 1e-8: 
            sgd_batch_adv = (sgd_batch_adv - sgd_batch_adv.mean()) / (sgd_batch_adv.std() + 1e-8)  
        else:
            sgd_batch_adv = sgd_batch_adv - sgd_batch_adv.mean() 
        sgd_batch_state = torch.as_tensor(sgd_batch_state, dtype=torch.float32) 
        sgd_batch_act = torch.as_tensor(sgd_batch_act, dtype=torch.long)
        sgd_batch_logp = torch.as_tensor(sgd_batch_logp, dtype=torch.float32) 
        sgd_batch_adv = torch.as_tensor(sgd_batch_adv, dtype=torch.float32) 
        sgd_batch_val_target = torch.as_tensor(sgd_batch_val_target, dtype=torch.float32) 
        return sgd_batch_state, sgd_batch_act, sgd_batch_logp, sgd_batch_adv, sgd_batch_val_target 


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
            nn.Tanh(),  
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),  
            nn.Tanh(),         
            layer_init(nn.Linear(hidden_sizes[1], act_dim), std=0.01)  
        )
    # 前向传播
    def forward(self, obs, act=None):  
        logits = self.actor_net(obs) # 动作未归一化概率
        dist = Categorical(logits=logits) # 创建分类分布
        # 动作对数概率
        if act is not None: 
            logp = dist.log_prob(act.squeeze())  
        else:  
            logp = None  
        entropy = dist.entropy() # 分布熵
        return dist, logp, entropy  
    # 采样动作，不计算梯度
    def get_a(self, obs): 
        with torch.no_grad(): 
            dist, _, _ = self.forward(obs) 
            action = dist.sample()  
            logp = dist.log_prob(action) # 单状态对数概率
        return action.item(), logp.item() 


# Critic网络：全连接网络
class Critic_Net(nn.Module): 
    def __init__(self, obs_dim, hidden_sizes):  
        super(Critic_Net, self).__init__()  
        self.critic_net = nn.Sequential( 
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])), 
            nn.Tanh(), 
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])), 
            nn.Tanh(),  
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0) 
        )    
    def forward(self, obs):  
        return self.critic_net(obs).squeeze() 
    def get_v(self, obs):
        with torch.no_grad(): 
            return self.forward(obs).item() 


# PPO
class PPO:
    def __init__(self, env, epochs):
        # 环境
        self.env = env 
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n 
        # 网络
        hidden_sizes = [64, 64] # 隐藏层大小
        self.actor = Actor_Net(self.obs_dim, self.act_dim, hidden_sizes) 
        self.critic = Critic_Net(self.obs_dim, hidden_sizes) 
        # 学习率与优化器
        self.pi_lr = 1e-4  
        self.critic_lr = 5e-4 
        self.pi_optimizer = Adam(self.actor.parameters(), lr=self.pi_lr)  
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr) 
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(self.pi_optimizer, step_size=50, gamma=0.9)  
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=50, gamma=0.9)  
        # 其他超参数
        self.epochs = epochs # 训练轮数
        self.save_freq = 100 # 模型保存频率
        self.clip_ratio = 0.1 # PPO裁剪比例
        self.episodes_num = 8 # 每次更新采样回合数
        self.sgd_num = 64 # 批大小
        self.train_pi_iters = 10 # 策略网络训练迭代次数
        self.entropy_coeff = 0.01 # 熵正则化系数
        self.max_grad_norm = 0.3 # 梯度裁剪阈值
        self.kl = 0.02 # KL散度阈值
        self.patience = 50 # 早停容忍轮数
        self.sampler = Sample(env) 
        self.return_traj = [] 
        self.policy_losses = [] 
        self.entropy_losses = []  
        self.critic_losses = []  
        self.kl_divergences = [] 
        self.recent_returns = deque(maxlen=20) # 性能监控
        self.best_return = -float('inf') 
        # 模型保存
        self.training_path = r'lunarlander_dis_ppo'
        os.makedirs(self.training_path, exist_ok=True)      
        self.best_model_path = None
    # 策略损失函数
    def compute_loss_pi(self, obs, act, logp_old, adv): 
        _, logp, entropy = self.actor(obs, act)  
        ratio = torch.exp(logp - logp_old.squeeze()) # 重要性采样比率
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv.squeeze() # 裁剪优势函数
        loss_pi = -torch.min(ratio * adv.squeeze(), clip_adv).mean() # PPO策略损失
        loss_entropy = entropy.mean() # 熵正则化项
        approx_kl = torch.abs(logp_old.squeeze() - logp).mean().item() # KL散度
        return loss_pi, loss_entropy, approx_kl 
    # 价值函数损失
    def compute_loss_critic(self, obs, val_target): 
        return ((self.critic(obs) - val_target.squeeze()) ** 2).mean() # 均方误差损失    
    # 更新
    def update(self):  
        # 采集数据
        self.sampler.sample_many_episodes(self.actor, self.critic, self.episodes_num)
        avg_return = self.sampler.sum_return / self.episodes_num  
        self.return_traj.append(avg_return)  
        self.recent_returns.append(avg_return)  
        batch_size = self.sampler.batch_state.shape[0] 
        self.sampler.index = np.arange(batch_size) 
        sum_pi_loss = 0.0  
        sum_entropy_loss = 0.0  
        sum_critic_loss = 0.0 
        sum_kl = 0.0  
        update_count = 0     
        # 训练
        for i in range(self.train_pi_iters): 
            np.random.shuffle(self.sampler.index) # 随机打乱批次索引
            epoch_kl = 0.0  
            epoch_updates = 0 
            for start_index in range(0, batch_size - self.sgd_num, self.sgd_num): 
                batch_state, batch_act, batch_logp, batch_adv, batch_val_target = self.sampler.get_data(start_index, self.sgd_num)  
                # 策略网络
                self.pi_optimizer.zero_grad() # 清零梯度
                loss_pi, loss_entropy, approx_kl = self.compute_loss_pi(batch_state, batch_act, batch_logp, batch_adv) # 策略损失
                total_pi_loss = loss_pi - self.entropy_coeff * loss_entropy # 总策略损失（包含熵正则化）
                total_pi_loss.backward() # 反向传播
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # 梯度裁剪
                self.pi_optimizer.step() # 更新参数
                epoch_kl += approx_kl # 累积KL散度
                epoch_updates += 1 
                # 价值网络
                self.critic_optimizer.zero_grad() # 清零梯度
                loss_critic = self.compute_loss_critic(batch_state, batch_val_target) # 评价损失
                loss_critic.backward() # 反向传播
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # 梯度裁剪
                self.critic_optimizer.step() # 更新参数
                sum_pi_loss += loss_pi.item() # 累积策略损失
                sum_entropy_loss += loss_entropy.item() # 累积熵损失
                sum_critic_loss += loss_critic.item() # 累积评价损失
                sum_kl += approx_kl # 累积KL散度
                update_count += 1 
            # KL散度早停（内层早停）
            avg_kl = epoch_kl / epoch_updates if epoch_updates > 0 else 0 # 平均KL散度
            if avg_kl > self.kl: 
                break       
        # 更新学习率
        self.pi_scheduler.step()  
        self.critic_scheduler.step()  
        if update_count > 0: 
            self.policy_losses.append(sum_pi_loss / update_count)  
            self.entropy_losses.append(sum_entropy_loss / update_count) 
            self.critic_losses.append(sum_critic_loss / update_count) 
            self.kl_divergences.append(sum_kl / update_count) 
        print(f"平均回报{avg_return:.2f}")  
    # 训练
    def ppo_train(self):  
        start_time = time.time() 
        best_avg_return = -float('inf') 
        no_improvement_count = 0 
        for epoch in range(self.epochs): 
            print(f"训练{epoch+1}轮")  
            self.update()  
            # 最近平均回报
            if len(self.recent_returns) >= 10:  
                avg_recent_return = np.mean(list(self.recent_returns))  
            else: 
                avg_recent_return = self.return_traj[-1]  
            # 更新最佳回报并保存
            if avg_recent_return > best_avg_return: 
                best_avg_return = avg_recent_return  
                no_improvement_count = 0 
                self.best_return = avg_recent_return 
                best_model_path = os.path.join(self.training_path, 'best_actor.pth') 
                best_critic_path = os.path.join(self.training_path, 'best_critic.pth')              
                torch.save(self.actor.state_dict(), best_model_path) 
                torch.save(self.critic.state_dict(), best_critic_path) 
                self.best_model_path = best_model_path 
                print(f"已保存更好模型，平均回报: {avg_recent_return:.2f}") 
            # 早停
            else: 
                no_improvement_count += 1 
            if no_improvement_count >= self.patience and len(self.recent_returns) >= 10: 
                print(f"性能在{self.patience}轮内无改进，提前停止训练")  
                break 
            # 定期保存
            if (epoch + 1) % self.save_freq == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.training_path, f'{epoch + 1}_actor.pth')) 
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
        plt.savefig(os.path.join(self.training_path, 'reward_curves.png'), dpi=300)
        # 损失曲线
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
        # 策略损失
        if self.policy_losses:
            axes[0, 0].plot(self.policy_losses, 'r-')
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Epochs')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        # 熵
        if self.entropy_losses:
            axes[0, 1].plot(self.entropy_losses, 'm-')
            axes[0, 1].set_title('Entropy Loss')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        # 价值损失
        if self.critic_losses:
            axes[1, 0].plot(self.critic_losses, 'b-')
            axes[1, 0].set_title('Critic Loss')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        # KL散度
        if self.kl_divergences:
            axes[1, 1].plot(self.kl_divergences, 'g-')
            axes[1, 1].set_title('KL Divergence')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('KL Divergence')
            axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.training_path, 'loss_curves.png'), dpi=300)


if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    ppo = PPO(env, epochs=1000)     
    ppo.ppo_train()
    env.close()