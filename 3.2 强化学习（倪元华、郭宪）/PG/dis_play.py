import gymnasium as gym
import pygame
import numpy as np


class LunarLanderGame:
    def __init__(self):
        self.env = gym.make("LunarLander-v3", render_mode="human")
        self.action_names = ["无动作", "左侧引擎", "主引擎", "右侧引擎"]
        self.rewards_history = []
        self.current_episode = 0
        self.print_controls()
    def print_controls(self):
        print("\n=== 游戏说明 ===")
        print("空格键: 无动作")
        print("A 或 左箭头: 左侧引擎")
        print("W 或 上箭头: 主引擎")
        print("D 或 右箭头: 右侧引擎")
        print("R: 重置环境")
        print("Q 或 ESC: 退出游戏")
        print("================\n")
    def get_action_from_key(self, key):
        if key in [pygame.K_SPACE]:
            return 0
        elif key in [pygame.K_a, pygame.K_LEFT]:
            return 1
        elif key in [pygame.K_w, pygame.K_UP]:
            return 2
        elif key in [pygame.K_d, pygame.K_RIGHT]:
            return 3
        return None
    def play_episode(self):
        self.current_episode += 1
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\n开始第{self.current_episode}回合...")
        pygame.init()
        clock = pygame.time.Clock()
        while not done:
            self.env.render()
            action = None
            # 处理键盘事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return -1 # 退出游戏信号
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        return -1 # 退出游戏信号
                    elif event.key == pygame.K_r:
                        return 0 # 重置当前回合
                    else:
                        action = self.get_action_from_key(event.key)
            # 处理按键
            keys = pygame.key.get_pressed()
            if action is None:
                if keys[pygame.K_SPACE]:
                    action = 0
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = 1
                elif keys[pygame.K_w] or keys[pygame.K_UP]:
                    action = 2
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    action = 3
                else:
                    action = 0 # 默认无动作
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if steps % 20 == 0:
                print(f"{steps}步, 总奖励{total_reward:.2f}")
            clock.tick(30) # 30 FPS
        self.rewards_history.append(total_reward)
        print(f"\n回合结束! 总步数{steps}，总奖励{total_reward:.2f}")
        if len(self.rewards_history) % 5 == 0:
            self.show_statistics()
        return 1  # 正常完成
    def show_statistics(self):
        if len(self.rewards_history) > 0:
            print("\n===== 游戏统计 =====")
            print(f"{len(self.rewards_history)}回合，平均奖励{np.mean(self.rewards_history):.2f}，最高奖励{np.max(self.rewards_history):.2f}")
            print("====================\n")
    def run_game(self):
        try:
            while True:
                result = self.play_episode()
                if result < 0: # 退出信号
                    break
                # 询问玩家是否继续
                if len(self.rewards_history) % 5 == 0:
                    response = input("\n是否继续游戏? (y/n): ").lower()
                    if response != 'y' and response != '':
                        break
            self.show_statistics()
        finally:
            self.env.close()
            pygame.quit()


if __name__ == "__main__":
    game = LunarLanderGame()
    game.run_game()