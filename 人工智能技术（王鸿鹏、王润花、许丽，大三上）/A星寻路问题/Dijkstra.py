import sys
import time
import tkinter
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

# 节点类
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.G = 0          # 起点到当前节点的实际代价（欧式距离）
        self.parent = None  # 储存父节点
    # 重载==
    def __eq__(self, other):
        if other and self:
            return self.x == other.x and self.y == other.y
        else:
            return not (self or other)

# 地图类
class Map:
    def __init__(self, in_map, target_position):
        self.r = len(in_map)
        self.c = len(in_map[0])
        self.map = []
        for i in range(self.r):
            self.map.append([])
            for j in range(self.c):
                self.map[i].append(Node(i, j))
    def Getnode(self, node):
        return self.map[node.x][node.y]
    def in_open_list(self, pos):
        return any(open_list_pos == pos for open_list_pos in open_list)
    def in_close_list(self, pos):
        return any(close_list_pos == pos for close_list_pos in close_list)

# A*算法
def SearchPath(in_map, start_position, target_position):
    # 检查起点、终点
    def check():
        flag = 1
        if in_map[start_position.x][start_position.y] == 1:
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showwarning("输入不合法", "起点为障碍！")
            root.destroy()
            flag = 0
            return flag
        if in_map[target_position.x][target_position.y] == 1:
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showwarning("输入不合法", "终点为障碍！")
            root.destroy()
            flag = 0
            return flag
    # 迭代
    def update(frame):
        if not open_list:                          # 如果open_list空，则停止动画
            animate.event_source.stop()
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showwarning("没有路径", "没有从起点到终点的路径！")
            root.destroy()
            return lines
        current_position = min(open_list, key=lambda elem: map.Getnode(elem).G)
        if current_position == target_position:    # 如果到达终点，则停止动画
            animate.event_source.stop()
            path(current_position)
            return lines
        open_list.remove(current_position)
        close_list.append(current_position)
        # 遍历邻居
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = current_position.x + dx, current_position.y + dy
            # 在范围内并且不是障碍物
            if 0 <= ni < map.r and 0 <= nj < map.c and in_map[ni][nj] == 0:
                new_G = map.Getnode(current_position).G + (1.414 if (dx != 0 and dy != 0) else 1)
                # 不在两个表内，加入开表
                if not map.in_close_list(Node(ni, nj)) and not map.in_open_list(Node(ni, nj)):
                    open_list.append(Node(ni, nj))
                    map.map[ni][nj].parent = current_position
                    map.map[ni][nj].G = new_G
                # 在开表并且更近，更新
                elif map.in_open_list(Node(ni, nj)) and map.map[ni][nj].G > new_G:
                    map.map[ni][nj].parent = current_position
                    map.map[ni][nj].G = new_G
        path(current_position)
        return lines
    # 构建路径
    def path(current_position):
        # 回溯并反转
        path = []
        while current_position:
            path.append(current_position)
            current_position = map.Getnode(current_position).parent
        path.reverse()
        # 转存
        path_set = set()
        for pos in path:
            path_set.add((pos.x, pos.y))
        Draw(map, path_set)
        add_legend(ax)
    # 绘图
    def Draw(map, path_set):
        ax.clear()
        # 设置等比例x、y轴范围，关闭坐标轴，翻转图像
        ax.set_xlim(0, map.c)
        ax.set_ylim(0, map.r)
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.gca().invert_yaxis()
        # 设置字体
        font = FontProperties(size=10, family='Arial')
        # 绘制格子
        for i in range(map.r):
            for j in range(map.c):
                # 坐标轴
                ax.text(-0.5, i + 0.5, str(i + 1), ha='center', va='center')
                ax.text(j + 0.5, -0.5, str(j + 1), ha='center', va='center', rotation=0)
                if in_map[i][j] == 1:                                                               # 障碍物格子填黑
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                else:
                    if (i, j) in path_set:                                                          # 路径格子填红
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=2, edgecolor='black', facecolor='red')
                    elif map.in_open_list(Node(i, j)) and (i, j) not in path_set:                   # open_list格子填绿
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='green')
                    elif map.in_close_list(Node(i, j)) and (i, j) not in path_set:                  # close_list格子填蓝
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='blue')
                    else:                                                                           # 非路径非障碍物格子填白
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                node = map.map[i][j]
                # 显示代价
                ax.text(j, i + 0.15, round(node.G, 1), horizontalalignment='left', verticalalignment='center', fontproperties=font)
        # 起点和终点
        rect = patches.Rectangle((start_position.y, start_position.x), 1, 1, linewidth=2, edgecolor='black', facecolor='pink')
        ax.add_patch(rect)
        ax.text(start_position.y + 1 / 2, start_position.x + 1 / 2, 'start', horizontalalignment='center', verticalalignment='center', fontproperties=font)
        rect = patches.Rectangle((target_position.y, target_position.x), 1, 1, linewidth=2, edgecolor='black', facecolor='yellow')
        ax.add_patch(rect)
        ax.text(target_position.y + 1 / 2, target_position.x + 1 / 2, 'end', horizontalalignment='center', verticalalignment='center', fontproperties=font)
    # 图例
    def add_legend(ax):
        start_patch = patches.Patch(color='pink', label='Start')
        end_patch = patches.Patch(color='yellow', label='End')
        path_patch = patches.Patch(color='red', label='Path')
        open_patch = patches.Patch(color='green', label='Open List')
        closed_patch = patches.Patch(color='blue', label='Closed List')
        obstacle_patch = patches.Patch(color='black', label='Obstacle')
        # 添加到图像底部
        ax.legend(handles=[start_patch, end_patch, path_patch, open_patch, closed_patch, obstacle_patch],
                  loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
    # 主函数
    start_time = time.time()
    flag = check()
    if flag == 0:
        sys.exit()
    map = Map(in_map, target_position)
    open_list.append(start_position)
    fig, ax = plt.subplots(figsize=(map.c * 0.8, map.r * 0.8))
    lines = []
    animate = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200, blit=False)
    plt.show()
    end_time = time.time()
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showwarning("找到路径", f"成功找到解，用时{end_time - start_time}s")
    root.destroy()

# 主程序
if __name__ == "__main__":
    in_map = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]
    start_position = Node(0, 0)
    target_position = Node(13, 13)
    path_set = set()
    open_list = []
    close_list = []
    SearchPath(in_map, start_position, target_position)