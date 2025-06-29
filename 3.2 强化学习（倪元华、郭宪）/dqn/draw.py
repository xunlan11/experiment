import torch
from torch.multiprocessing import Process, Queue
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import ale_py
import numpy as np
from frame_stack_wrapper import FrameStackWrapper
import matplotlib.pyplot as plt

# 参数配置
GAME = 'AirRaid'
FRAME_SKIP = 2
CONCURRENT_PROCESSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DQN(torch.nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            layer_init(torch.nn.Conv2d(4, 32, 8, stride=4)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Conv2d(32, 64, 4, stride=2)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Conv2d(64, 64, 3, stride=1)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            layer_init(torch.nn.Linear(64 * 7 * 7, 512)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(512, n_actions))
        )
    def forward(self, x):
        x = x.float() / 255.0
        return self.net(x)

def run(queue, shared_policy_net_state, render):
    try:
        with torch.cuda.device(DEVICE):
            gym.register_envs(ale_py)
            env = gym.make(f"ALE/{GAME}-v5", render_mode="human" if render else None, frameskip=1)
            env = AtariPreprocessing(env, frame_skip=FRAME_SKIP, screen_size=84, grayscale_obs=True, scale_obs=False)
            env = FrameStackWrapper(env, num_stack=4)
            n_actions = env.action_space.n
            policy_net = DQN(n_actions).to(DEVICE)
            policy_net.load_state_dict(shared_policy_net_state)
            policy_net.eval()
            state, _ = env.reset()
            total_reward = 0
            with torch.no_grad():
                while True:
                    state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(DEVICE)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    state = next_state
                    if terminated or truncated:
                        break
            queue.put(total_reward)
    except Exception as e:
        print(f"进程出错: {str(e)}")
        queue.put(0)
    finally:
        if 'env' in locals():
            env.close()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    TEST_PER_MODEL = 10  # 每个模型测试次数
    all_averages = []
    for model_num in range(1, 91):
        policy_file = f"policy/policy_{model_num * 10000}.pth"
        shared_policy = torch.load(policy_file, map_location='cpu')
        rewards = []
        for batch_start in range(0, TEST_PER_MODEL, CONCURRENT_PROCESSES):
            batch_size = min(CONCURRENT_PROCESSES, TEST_PER_MODEL - batch_start)
            print(f"模型{model_num} - 批次{batch_start // CONCURRENT_PROCESSES + 1}/{(TEST_PER_MODEL - 1) // CONCURRENT_PROCESSES + 1}")
            queue = Queue()
            processes = []
            for _ in range(batch_size):
                p = Process(target=run, args=(queue, shared_policy, False))
                p.start()
                processes.append(p)
            try:
                for p in processes:
                    p.join(timeout=120)
                while not queue.empty():
                    reward = queue.get()
                    rewards.append(reward)
            except KeyboardInterrupt:
                print("用户中断测试")
                break
        avg_reward = sum(rewards) / TEST_PER_MODEL if rewards else 0.0
        all_averages.append(avg_reward)
        print(f"模型{model_num}平均分：{avg_reward:.1f}")
    # 绘制结果
    plt.figure(figsize=(15, 7))
    plt.plot(range(1, 91), all_averages, marker='o', linestyle='-', color='b')
    plt.title('Average Reward vs Training Step')
    plt.xlabel('Model Step(w)')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()