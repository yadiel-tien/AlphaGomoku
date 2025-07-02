import re

import gymnasium as gym
from gymnasium import spaces
import pygame
import random

from config import CONFIG
from inference import InferenceEngine as IE
from player import Human, AIServer, RandomPlayer
from ui import Button
import numpy as np

from utils import Timer

settings = CONFIG['chess']


class ChineseChess(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = None
        self.board = None
        self.piece2id = None
        self.id2piece = None
        self.bishop_moves = None
        self.advisor_moves = None
        self.move2action = None
        self.action2move = None
        self.dest_func = None
        self.no_eat_count = 0
        self.winner = 2  # 0,1代表获胜玩家，-1代表平局，2代表未决胜负
        self._init_dicts()
        self.action_space = spaces.Discrete(2086)
        self.observation_space = spaces.Box(0, 1, shape=(10, 9, 20), dtype=np.float32)
        self.current_player = 0  # 当前走棋方，0代表红，1代表黑
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.history = []
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        重置游戏, 返回当前棋盘状态
        state.shape=(10,9,20),其中0-6红方，7-13黑方,14-18历史5步，19当前玩家
        """
        self.board = np.array([
            [7, 8, 9, 10, 11, 10, 9, 8, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 12, -1, -1, -1, -1, -1, 12, -1],
            [13, -1, 13, -1, 13, -1, 13, -1, 13],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, 6, -1, 6, -1, 6, -1, 6],
            [-1, 5, -1, -1, -1, -1, -1, 5, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
        ])
        self.current_player = 0
        self.no_eat_count = 0
        self.winner = 2
        self._init_state()
        self.terminated = False
        self.last_action = None
        return self.state, {}

    def _init_state(self):
        # 根据self.board设置self.state
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        # 当前盘面
        for r in range(10):
            for c in range(9):
                channel = self.board[r, c]
                if channel != -1:
                    self.state[r, c, channel] = 1

    def valid_actions(self):
        available_actions = []
        for r in range(10):
            for c in range(9):
                piece = self.board[r, c]
                if (self.current_player == 0 and 0 <= piece <= 6) or (self.current_player == 1 and piece >= 7):
                    destinations = self.dest_func[piece](r, c)
                    for to_r, to_c in destinations:
                        available_actions.append(self.move2action[(r, c, to_r, to_c)])
        return available_actions

    def get_rook_dest(self, r, c):
        destinations = []
        for dirct in (-1, 1):
            for add_r in range(1, 10):
                to_r = r + dirct * add_r
                if 0 <= to_r < 10:
                    if self.board[to_r, c] == -1:  # 空位
                        destinations.append((to_r, c))
                    elif (self.board[to_r, c] < 7) != (self.board[r, c] < 7):  # 对手棋子，可以吃
                        destinations.append((to_r, c))
                        break
                    else:
                        break

            for add_c in range(1, 9):
                to_c = c + dirct * add_c
                if 0 <= to_c < 9:
                    if self.board[r, to_c] == -1:  # 空位
                        destinations.append((r, to_c))
                    elif (self.board[r, to_c] < 7) != (self.board[r, c] < 7):  # 对手棋子，可以吃
                        destinations.append((r, to_c))
                        break
                    else:
                        break
        return destinations

    def get_horse_dest(self, r, c):
        destinations = []
        for add_r, add_c, in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            to_r, to_c = r + add_r, c + add_c
            obstacle_r, obstacle_c = int(add_r / 2) + r, int(add_c / 2) + c
            # 目标位置在棋盘上，且没有蹩脚
            if 0 <= to_r < 10 and 0 <= to_c < 9 and self.board[obstacle_r, obstacle_c] == -1:
                # 目标位置没有棋子或有对方棋子
                if self.board[to_r, to_c] == -1 or ((self.board[r, c] < 7) ^ (self.board[to_r, to_c] < 7)):
                    destinations.append((to_r, to_c))
        return destinations

    def get_bishop_dest(self, r, c):
        destinations = []
        for from_r, from_c, to_r, to_c in self.bishop_moves:
            obstacle_r, obstacle_c = (from_r + to_r) // 2, (from_c + to_c) // 2
            if from_r == r and from_c == c and self.board[obstacle_r, obstacle_c] == -1:
                destinations.append((to_r, to_c))
        return destinations

    def get_advisor_dest(self, r, c):
        destinations = []
        for from_r, from_c, to_r, to_c in self.advisor_moves:
            if from_r == r and from_c == c:
                if self.board[to_r, to_c] == -1 or (self.board[from_r, from_c] < 7) != (
                        self.board[to_r, to_c] < 7):  # 目标位置为空或为敌方棋子
                    destinations.append((to_r, to_c))
        return destinations

    def get_king_dest(self, r, c):
        destinations = []
        candidates = ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
        for to_r, to_c in candidates:
            if 3 <= to_c <= 5:
                # 红帅活动范围
                if self.board[r, c] == 4:
                    if 7 <= to_r <= 9 and (self.board[to_r, to_c] == -1 or self.board[to_r, to_c] > 6):
                        destinations.append((to_r, to_c))
                else:  # 黑帅活动范围
                    if 0 <= to_r <= 3 and (self.board[to_r, to_c] == -1 or self.board[to_r, to_c] <= 6):
                        destinations.append((to_r, to_c))
        # 两帅照面的情况
        if self.board[r, c] == 4:
            for i in range(1, 9):
                if self.board[r - i, c] == 11:
                    destinations.append((r - i, c))
                    break
                elif r - i < 0 or self.board[r - i, c] != -1:
                    break
        if self.board[r, c] == 11:
            for i in range(1, 9):
                if self.board[r + i, c] == 4:
                    destinations.append((r + i, c))
                    break
                elif r + i > 9 or self.board[r + i, c] != -1:
                    break
        return destinations

    def get_cannon_dest(self, r, c):
        destinations = []
        for dirct in (-1, 1):
            found_screen = False  # 是否遇到炮架
            for add_r in range(1, 10):
                to_r = r + dirct * add_r
                if 0 <= to_r < 10:  # 在棋盘上
                    if not found_screen and self.board[to_r, c] == -1:  # 空位直接移动
                        destinations.append((to_r, c))
                    elif not found_screen:  # 碰到第一个棋子作为炮架
                        found_screen = True
                    elif self.board[to_r, c] == -1:
                        pass
                    elif (self.board[to_r, c] < 7) == (self.board[r, c] < 7):  # 乙方棋子
                        break
                    elif (self.board[to_r, c] < 7) != (self.board[r, c] < 7):  # 炮架后对方棋子可以吃
                        destinations.append((to_r, c))
                        break
            found_screen = False
            for add_c in range(1, 9):
                to_c = c + dirct * add_c
                if 0 <= to_c < 9:
                    if not found_screen and self.board[r, to_c] == -1:
                        destinations.append((r, to_c))
                    elif not found_screen:
                        found_screen = True
                    elif self.board[r, to_c] == -1:
                        pass
                    elif (self.board[r, to_c] < 7) == (self.board[r, c] < 7):
                        break
                    elif (self.board[r, to_c] < 7) != (self.board[r, c] < 7):
                        destinations.append((r, to_c))
                        break

        return destinations

    def get_pawn_dest(self, r, c):
        destinations = []
        if self.board[r, c] == 6:  # 红兵
            if r > 4:  # 过河前
                if self.board[r - 1, c] > 6 or self.board[r - 1, c] == -1:  # 目标位置为空或对方棋子
                    destinations.append((r - 1, c))
            else:  # 过河后
                candidates = ((r - 1, c), (r, c + 1), (r, c - 1))  # 前右左
                for to_r, to_c in candidates:
                    if 0 <= to_r < 10 and 0 <= to_c < 9:  # 在棋盘上
                        if self.board[to_r, to_c] > 6 or self.board[to_r, to_c] == -1:  # 目标位置为空或对方棋子
                            destinations.append((to_r, to_c))
        else:  # 黑兵
            if r < 5:
                if self.board[r + 1, c] < 7:
                    destinations.append((r + 1, c))
            else:
                candidates = ((r + 1, c), (r, c - 1), (r, c + 1))
                for to_r, to_c in candidates:
                    if 0 <= to_r < 10 and 0 <= to_c < 9:
                        if self.board[to_r, to_c] < 7:
                            destinations.append((to_r, to_c))
        return destinations

    def _init_dicts(self):
        self.piece2id = {
            '红车': 0, '红马': 1, '红象': 2, '红士': 3, '红帅': 4, '红炮': 5, '红兵': 6,
            '黑车': 7, '黑马': 8, '黑象': 9, '黑士': 10, '黑帅': 11, '黑炮': 12, '黑兵': 13, '一一': -1
        }
        self.id2piece = {v: k for k, v in self.piece2id.items()}
        self.move2action = {}
        a = 0
        # 垂直水平移动
        for r in range(10):
            for c in range(9):
                for to_r in range(10):
                    if to_r != r:
                        move = (r, c, to_r, c)
                        self.move2action[move] = a
                        a += 1
                for to_c in range(9):
                    if to_c != c:
                        move = (r, c, r, to_c)
                        self.move2action[move] = a
                        a += 1
        # 士的动作
        self.advisor_moves = [
            (0, 3, 1, 4), (0, 5, 1, 4), (1, 4, 0, 3), (1, 4, 0, 5), (1, 4, 2, 3), (1, 4, 2, 5), (2, 3, 1, 4),
            (2, 5, 1, 4), (9, 3, 8, 4), (9, 5, 8, 4), (8, 4, 9, 3), (8, 4, 9, 5), (8, 4, 7, 3), (8, 4, 7, 5),
            (7, 3, 8, 4), (7, 5, 8, 4)
        ]
        for move in self.advisor_moves:
            self.move2action[move] = a
            a += 1
        # 象的动作
        self.bishop_moves = [
            (0, 2, 2, 0), (0, 2, 2, 4), (0, 6, 2, 4), (0, 6, 2, 8), (2, 0, 0, 2), (2, 0, 4, 2), (2, 4, 0, 2),
            (2, 4, 4, 2), (2, 4, 0, 6), (2, 4, 4, 6), (2, 8, 0, 6), (2, 8, 4, 6), (4, 2, 2, 0), (4, 2, 2, 4),
            (6, 2, 2, 4), (6, 2, 2, 8)
        ]
        rival_bishop_moves = [(9 - r, c, 9 - to_r, to_c) for r, c, to_r, to_c in self.bishop_moves]
        self.bishop_moves.extend(rival_bishop_moves)
        for move in self.bishop_moves:
            self.move2action[move] = a
            a += 1
        # 马的动作
        for r in range(10):
            for c in range(9):
                for add_r, add_c, in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
                    to_r, to_c = r + add_r, c + add_c
                    if 0 <= to_r < 10 and 0 <= to_c < 9:
                        move = (r, c, to_r, to_c)
                        self.move2action[move] = a
                        a += 1
        self.action2move = {v: k for k, v in self.move2action.items()}

        self.dest_func = {0: self.get_rook_dest, 1: self.get_horse_dest, 2: self.get_bishop_dest,
                          3: self.get_advisor_dest, 4: self.get_king_dest, 5: self.get_cannon_dest,
                          6: self.get_pawn_dest}
        for i in range(7, 14):
            self.dest_func[i] = self.dest_func[i - 7]

    def run(self, players, silent=False):
        """模拟玩家比赛，玩家1胜返回(1,0)，玩家2胜返回(0,1)，平局返回（0，0）"""
        self.reset()
        for player in players:
            player.silent = silent
            player.reset()

        while True:
            if not silent:
                print(f'-----player{self.current_player + 1}-----')

            action = players[self.current_player].get_action(self)
            _, reward, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                outcome = self.winner
                break
        return outcome

    def random_order_play(self, players, silent=False):
        """随机一局对弈，先手顺序随机"""
        n = random.randint(0, 1)
        p1, p2 = players
        current_players = [p1, p2] if n == 0 else [p2, p1]
        winner = self.run(current_players, silent=silent)
        if winner == -1:
            return -1
        if n == 0:
            return winner
        else:
            return 1 - winner

    def evaluate(self, players, n_rounds=100):
        """2玩家对弈，打印胜率"""
        outcomes = []
        for i in range(n_rounds):
            print(f'第{i + 1}局:', end=' ')
            # 改变先手顺序
            outcome = self.random_order_play(players)
            outcomes.append(outcome)
            print(f'比赛结果：{outcome}')

        print(f"Player 1 Win Percentage:{outcomes.count(0) / n_rounds:.2%}", )
        print(f"Player 2 Win Percentage:{outcomes.count(1) / n_rounds:.2%}")
        print(f"Draw Percentage:{outcomes.count(-1) / n_rounds:.2%}")
        return outcomes

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        # 记录棋谱
        self.history.append(np.copy(self.board))

        r, c, to_r, to_c = self.action2move[action]
        attack_piece = self.board[r, c]
        attacked_piece = self.board[to_r, to_c]
        # 统计双方未吃子回合，用于平局
        if attacked_piece == -1:
            self.no_eat_count += 1
        else:
            self.no_eat_count = 0

        # 执行落子

        self.board[r, c] = -1
        self.board[to_r, to_c] = attack_piece
        self.state[r, c, attack_piece] = 0
        self.state[to_r, to_c, attack_piece] = 1
        self.state[to_r, to_c, attacked_piece] = 0
        # 历史信息
        self.state[:, :, 15:19] = self.state[:, :, 14:18]
        self.state[:, :, 14] = (self.board == self.history[-1]).astype(np.float32)
        # 更改玩家
        self.current_player = 1 - self.current_player
        self.state[:, :, 19] = self.current_player
        self.last_action = action

        # 胜利奖励1，平局或未结束奖励0，失败奖励-1
        self.check_winner()
        reward = 0
        if self.winner == 1 - self.current_player:
            reward = 1
        elif self.winner == self.current_player:
            reward = -1

        return self.state, reward, self.terminated, self.truncated, {}

    def check_winner(self):
        if not np.isin(4, self.board):  # 红帅被杀
            self.winner = 1
            self.terminated = True
        elif not np.isin(11, self.board):  # 黑帅被杀
            self.winner = 0
            self.terminated = True
        elif (
                len(self.history) > 4 and
                np.all(self.board == self.history[-2]) and
                np.all(self.board == self.history[-4])
        ):  # 先简化长捉和长将逻辑
            self.winner = self.current_player
            self.terminated = True
        elif self.no_eat_count >= 100:  # 双方未吃子超过50回合
            self.winner = -1
            self.terminated = True

    def render(self) -> None:
        """打印棋盘"""
        board_str = ''
        head = ''
        board_str += ' ' + ' '.join([f'{i:>5}' for i in range(9)]) + '\n'
        for i, row in enumerate(self.board):
            row_str = f'{i}'
            for piece_id in row:
                if 0 <= piece_id <= 6:
                    row_str += f' \033[91m{self.id2piece[piece_id]:^4}\033[0m'
                else:
                    row_str += f' {self.id2piece[piece_id]:^4}'
            board_str += row_str + '\n'

        print(board_str)

    def handle_human_input(self):
        self.render()
        while True:
            while True:
                txt = input('输入一个4位数字，前两位代表当前棋子位置，后两位代表移动到的位置，例如红方炮7平4为7774。\n')
                if len(txt) == 4 and txt.isdigit():
                    break
                else:
                    print("输入格式有误，请输入行列编号，逗号隔开。")

            move = tuple(map(int, txt))
            if move in self.move2action:
                action = self.move2action[move]
                if action in self.valid_actions():
                    break

            print("输入位置不合法，请重新输入！")
        return action

    def describe_move(self, action):
        r, c, to_r, to_c = self.action2move[action]
        piece = self.id2piece[self.board[r, c]]
        eat_piece = self.id2piece[self.board[to_r, to_c]]
        result = '' if eat_piece == '一一' else '吃 ' + eat_piece
        print(f'{piece} ({r},{c}) -> ({to_r}, {to_c}) {result}')


class ChineseChessUI:
    def __init__(self, players=None):
        self.piece_pics = {}
        grid_size = settings['grid_size']
        for i in range(14):
            pic = pygame.image.load(f'graphics/chess/piece{i}.png')
            pic = pygame.transform.scale(pic, (grid_size * 1.2, grid_size * 1.2))
            self.piece_pics[i] = pic

        self.mark_pic = pygame.image.load('graphics/gomoku/circle.png')
        self.image = pygame.image.load('graphics/chess/board.jpeg')
        self.image = pygame.transform.scale(self.image, (540, 600))
        self.screen = pygame.display.get_surface()
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.env = ChineseChess()
        self.button = Button("Start", self.start, (200, 700), color='green')
        self.status = 'new'  # new playing finished
        # 0,1分别代表红方黑方
        self.timers = {0: Timer(limit=60000, func=self.time_up), 1: Timer(limit=60000, func=self.time_up)}
        self.players = players
        self.winner = None
        self.history = []

    def handle_input(self, event):
        if self.status == 'playing':
            self.players[self.env.current_player].handle_input(event)
        else:
            self.button.handle_input(event)

    def update(self):
        if self.status == 'playing':
            player = self.players[self.env.current_player]
            if player.pending_action is None:
                player.update(self.env)
            else:
                move = self.env.action2move[player.pending_action]
                self.history.append((move, self.env.current_player))
                new_state, reward, terminated, truncated, _ = self.env.step(player.pending_action)
                player.pending_action = None

                if terminated or truncated:
                    winner = self.env.winner
                    winner = 'red' if winner == (1, 0) else 'black' if winner == (0, 1) else 'draw'
                    self.set_winner(winner)
                else:
                    self.switch_side()

    def draw(self):
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_pieces()
        self.draw_last_mark()
        if self.status == 'finished':
            self.draw_step_mark()
            self.draw_victory()
            self.button.draw()
        elif self.status == 'new':
            self.draw_new_game_title()
            self.button.draw()
        else:
            self.draw_player()
            self.draw_cursor()

    def time_up(self):
        for side, timer in self.timers.items():
            if timer.remain == 0:
                winner = 'white' if side else 'black'
                self.set_winner(winner)

    def switch_side(self):
        current_player = self.board.current_player
        prev_player = 1 - current_player
        # 重设timer
        self.timers[prev_player].reset()
        self.timers[current_player].activate()
        # 修改玩家状态
        self.players[prev_player].is_active = False
        self.players[current_player].is_active = True

    def set_winner(self, winner):
        self.winner = winner
        self.status = 'finished'
        self.button = Button("Restart", self.start, (200, 700), color='green')

    def draw_pieces(self):
        for row in range(10):
            for col in range(9):
                piece = int(self.env.board[row, col])
                if piece != -1:
                    x, y = self._index2pos(row, col)
                    self.screen.blit(self.piece_pics[piece], (x, y))

    def _index2pos(self, row: int, col: int) -> tuple[float, float]:
        y = (row - 5) * settings['grid_size'] + self.rect.centery
        x = (col - 4) * settings['grid_size'] + self.rect.centerx - settings['grid_size'] * 0.6
        return x, y

    def draw_step_mark(self):
        font = pygame.font.Font(None, 16)
        for idx, (row, col, mark) in enumerate(self.history):
            color = 'black' if mark else 'white'  # 黑棋则白字，白棋则黑字
            text = font.render(str(idx + 1), True, color)
            x, y = self._index2pos(row, col)
            rect = text.get_rect(center=(x + 17, y + 17))
            self.screen.blit(text, rect.topleft)

    def draw_last_mark(self):
        if self.history:
            row, col, _ = self.history[-1]
            x, y = self._index2pos(row, col)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_victory(self):
        path = f'graphics/{self.winner}_win.png'
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (200, 200))
        x = self.rect.centerx
        y = 60
        rect = image.get_rect(center=(x, y))
        self.screen.blit(image, rect.topleft)

    def draw_new_game_title(self):
        font = pygame.font.Font(None, 70)
        text = font.render('New Game', True, 'orange')
        x = self.rect.centerx
        y = 60
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def start(self):
        if self.status == 'finished':
            self.board.reset()
            for timer in self.timers.values():
                timer.reset()
        self.history = []
        self.status = 'playing'
        # 交换玩家
        self.players.reverse()
        # 重设玩家
        for player in self.players:
            player.reset()
        # 当前玩家开始计时
        self.players[self.board.current_player].is_active = True
        self.timers[self.board.current_player].activate()

    def draw_player(self):
        self.timers[self.board.current_player].update()
        white_sec = self.timers[1].remain // 1000
        black_sec = self.timers[0].remain // 1000
        font = pygame.font.Font(None, 60)
        white = font.render(f'{self.players[1].description}  White:{white_sec:02}', True, 'orange')
        black = font.render(f'{self.players[0].description} Black:{black_sec:02}', True, 'orange')
        white_rect = white.get_rect(midleft=(60, 60))
        black_rect = black.get_rect(midleft=(60, 740))
        self.screen.blit(white, white_rect.topleft)
        self.screen.blit(black, black_rect.topleft)

    def draw_cursor(self):
        self.players[self.board.current_player].draw()


if __name__ == '__main__':
    env = ChineseChess()
    competences = [Human(), RandomPlayer()]
    env.run(competences)
    env.render()
