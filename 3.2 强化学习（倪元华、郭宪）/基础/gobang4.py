import numpy as np
import pickle
import os

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

# 状态类
class State:
    def __init__(self):
        # 4*4棋盘数组，1、-1表示先后手标志，0表示空格
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None     # 胜利者标识
        self.hash_val = None   # 哈希值
        self.end = None        # 结束标识

    # 1、-1、0->2、0、1
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # 游戏结束（一方获胜或死局）检查
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # 计算行、列、正副对角线的和
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)
        # 和有±4则一方胜利
        for result in results:
            if result == 4:
                self.winner = 1
                self.end = True
                return self.end
            if result == -4:
                self.winner = -1
                self.end = True
                return self.end
        # 检验死局
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        # 都不是，游戏继续
        self.end = False
        return self.end

    # 下一步棋，更新状态
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # 绘制棋盘
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

# 迭代得到所有状态
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states: # 不在已知状态表中的状态加入该表
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states) # 迭代中，更换执棋者

# 初始化状态并调用上述函数得到所有状态
def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

# 评判类
class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1 # 先手
        self.p2 = player2 # 后手
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # 绘制棋局
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner

# 玩家类
class Player:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # 更新价值估算
    def backup(self):
        states = [state.hash() for state in self.states]
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # epsilon法选择贪婪/探索
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        # 遍历可行步
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())
        # 随机探索
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action
        # 贪婪（重排以保证等最大可能选项的公平性）
        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# 交互界面
# | 1 | 2 | 3 | 4 |
# | q | w | e | r |
# | a | s | d | f |
# | z | x | c | v |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['1', '2', '3', '4', 'q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol

# 训练
def train(n1, n2, print_every_n):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    total_epochs = max(n1, n2)
    for epoch in range(1, total_epochs + 1):
        winner = judger.play(print_state=False)
        # 根据当前epoch备份不同玩家的策略
        if epoch <= n1:
            player1.backup()
        if epoch <= n2:
            player2.backup()
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        if epoch % print_every_n == 0:
            print(f'Epoch {epoch}, '
                  f'Player 1 Win Rate: {player1_win / epoch:.02f}, '
                  f'Player 2 Win Rate: {player2_win / epoch:.02f}')
    judger.reset()
    player1.save_policy()
    player2.save_policy()

# 竞赛
def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# 对战
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")

def save_all_states(all_states, filename='all_states.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(all_states, f)

def load_all_states(filename='all_states.pkl'):
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    return None

if __name__ == '__main__':
    # 保留状态遍历结果，节省时间
    all_states = load_all_states()
    if not all_states:
        all_states = get_all_states()
        save_all_states(all_states)
    # 可训练不同次数
    train(n1=1000, n2=1000, print_every_n=100)
    compete(1000)
    # play()