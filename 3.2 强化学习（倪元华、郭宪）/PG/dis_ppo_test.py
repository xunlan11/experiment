import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import os
import pygame


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    def forward(self, obs, act=None):
        logits = self.actor_net(obs)
        dist = Categorical(logits=logits)
        if act is not None:
            logp = dist.log_prob(act.squeeze())
        else:
            logp = None
        entropy = dist.entropy()
        return dist, logp, entropy
    def get_a(self, obs):
        with torch.no_grad():
            dist, _, _ = self.forward(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), logp.item()
    def get_deterministic_a(self, obs):
        with torch.no_grad():
            logits = self.actor_net(obs)
            action = torch.argmax(logits).item()
        return action


class ModelTester:
    def __init__(self, model_path):
        self.env = gym.make('LunarLander-v3', render_mode="human")
        self.test_env = gym.make('LunarLander-v3')
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.hidden = [64, 64] 
        self.action_names = ["无动作", "左侧引擎", "主引擎", "右侧引擎"]
        self.actor = Actor_Net(self.obs_dim, self.act_dim, self.hidden)
        self.load_model(model_path)
        self.rewards_history = []
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path))
            print(f"成功加载模型: {model_path}")
        else:
            print(f"找不到模型: {model_path}")
            exit(1)
    def visual_test(self, stochastic=False):
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        actions = []
        while not done:
            self.env.render()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if stochastic:
                action, _ = self.actor.get_a(obs_tensor)
            else:
                action = self.actor.get_deterministic_a(obs_tensor)
            actions.append(action)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            step_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        print(f"\n演示结束! 总步数{step_count}，总奖励{episode_reward:.2f}")
        return episode_reward
    def test_episode(self, stochastic=False):
        obs, _ = self.test_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if stochastic:
                action, _ = self.actor.get_a(obs_tensor)
            else:
                action = self.actor.get_deterministic_a(obs_tensor)
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            step_count += 1
        self.rewards_history.append(episode_reward)
        return episode_reward, step_count
    def run_tests(self, num_tests=10, stochastic=False):
        print(f"\n开始进行{num_tests}次评估测试")
        self.rewards_history = []
        steps_history = []
        for i in range(num_tests):
            reward, steps = self.test_episode(stochastic)
            steps_history.append(steps)
            print(f"测试{i + 1}/{num_tests}: 奖励{reward:.2f}, 步数{steps}")
        rewards = np.array(self.rewards_history)
        steps = np.array(steps_history)
        print("\n===== 测试统计结果 =====")
        print(f"平均奖励: {rewards.mean():.2f}")
        print(f"最高奖励: {rewards.max():.2f}")
        print(f"平均步数: {steps.mean():.2f}")
        print("=====================\n")


if __name__ == "__main__":
    model_path = "lunarlander_dis_ppo/best_actor.pth"
    tester = ModelTester(model_path)
    tester.visual_test(stochastic=False) 
    tester.run_tests(num_tests=10, stochastic=False)
    tester.env.close()