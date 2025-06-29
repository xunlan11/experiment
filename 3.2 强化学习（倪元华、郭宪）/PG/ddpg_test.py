import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import os
import pygame


def layer_init(layer, std=np.sqrt(2), gain=1, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std * gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    def forward(self, obs):
        return self.act_limit * self.actor_net(obs)
    def get_a(self, obs, noise=False):
        with torch.no_grad():
            action = self.forward(obs).cpu().numpy()
        return action


class ModelTester:
    def __init__(self, model_path):
        self.env = gym.make('LunarLander-v3', continuous=True, render_mode="human")
        self.test_env = gym.make('LunarLander-v3', continuous=True)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.hidden = [256, 256]
        self.actor = Actor_Net(self.obs_dim, self.act_dim, self.hidden, self.act_limit)
        self.action_names = ["主引擎(上/下)力度", "左右方向控制力度"]
        self.load_model(model_path)
        self.rewards_history = []
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path))
            print(f"成功加载模型: {model_path}")
        else:
            print(f"找不到模型: {model_path}")
            exit(1)
    def visual_test(self):
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        actions_history = []
        while not done:
            self.env.render()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = self.actor.get_a(obs_tensor)
            actions_history.append(action)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            step_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        print(f"\n演示结束! 总步数{step_count}，总奖励{episode_reward:.2f}")
        actions_array = np.array(actions_history)
        if len(actions_array) > 0:
            print("\n动作统计:")
            mean_actions = actions_array.mean(axis=0)
            min_actions = actions_array.min(axis=0)
            max_actions = actions_array.max(axis=0)
            for i, (name, mean, min_val, max_val) in enumerate(zip(self.action_names, mean_actions, min_actions, max_actions)):
                print(f"  {name}: 平均 {mean:.3f}, 范围 [{min_val:.3f}, {max_val:.3f}]")
        return episode_reward

    def test_episode(self):
        obs, _ = self.test_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = self.actor.get_a(obs_tensor)
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            step_count += 1
        self.rewards_history.append(episode_reward)
        return episode_reward, step_count
    def run_tests(self, num_tests=10):
        print(f"\n开始进行{num_tests}次评估测试")
        self.rewards_history = []
        steps_history = []
        for i in range(num_tests):
            reward, steps = self.test_episode()
            steps_history.append(steps)
            print(f"测试{i + 1}/{num_tests}: 奖励{reward:.2f}, 步数{steps}")
        rewards = np.array(self.rewards_history)
        steps = np.array(steps_history)
        print("\n===== 测试统计结果 =====")
        print(f"平均奖励: {rewards.mean():.2f}")
        print(f"最高奖励: {rewards.max():.2f}")
        print(f"最低奖励: {rewards.min():.2f}")
        print(f"奖励标准差: {rewards.std():.2f}")
        print(f"平均步数: {steps.mean():.2f}")
        print("=====================\n")


if __name__ == "__main__":
    model_path = "lunarlander_ddpg/best_actor.pth"
    tester = ModelTester(model_path)
    tester.visual_test()
    tester.run_tests(num_tests=10)
    tester.env.close()