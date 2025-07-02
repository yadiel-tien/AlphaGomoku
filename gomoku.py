import re

import gymnasium as gym
from gymnasium import spaces
import pygame
import random

from chess import ChineseChess
from config import CONFIG
from inference import InferenceEngine as IE
from player import Human, AIServer
from ui import Button
import numpy as np

from utils import Timer

settings = CONFIG['gomoku']


class GomokuEnv(gym.Env):
    def __init__(self, rows: int = 15, columns: int = 15):
        super().__init__()
        self.shape = rows, columns, 2
        self.state = None
        self.action_space = spaces.Discrete(rows * columns)
        self.observation_space = spaces.Box(0, 1, shape=self.shape, dtype=np.float32)
        self.current_player = 0  # 当前走棋方，0代表黑，1代表白
        self.winner = 2  # 0,1代表获胜玩家，-1代表平局，2代表未决胜负
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        重置游戏, 返回当前棋盘状态
        state=[Xt, Yt]
        Xt 1代表有当前玩家棋子，0代表无当前玩家棋子，Yt 1代表有对手玩家棋子，0代表无对手玩家棋子
        """
        self.state = np.zeros(self.shape, dtype=np.float32)
        self.current_player = 0
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.winner = 2
        return self.state, {}

    def run(self, players, silent=False):
        """模拟玩家比赛，玩家1胜返回0，玩家2胜返回1，平局返回-1"""
        self.reset()
        for player in players:
            player.silent = silent
            player.reset()

        index = 0
        while True:
            if not silent:
                print(f'-----player{index + 1}-----')

            action = players[index].get_action(self)
            _, reward, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                outcome = self.winner
                break
            index = 1 - index
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

    def valid_actions(self):
        state = self.state[:, :, 0] + self.state[:, :, 1]
        return np.flatnonzero(state == 0)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        if self.terminated or self.truncated:
            raise ValueError('Game is already over.')
        if action not in self.valid_actions():
            raise ValueError('Invalid move: position already occupied or not in action space')

        row, col = self.action2coordinate(action)
        # 执行落子
        self.state[row, col, 0] = 1

        # 更改棋盘和当前玩家
        self.state[:, :, [0, 1]] = self.state[:, :, [1, 0]]
        self.current_player = 1 - self.current_player
        self.last_action = action

        self.check_win()
        # 胜利奖励1，平局或未结束奖励0，隐含失败后对手奖励1，通过min-max自己奖励-1
        reward = 0
        if self.winner == 1 - self.current_player:  # 落子导致胜利
            reward = 1

        return self.state, reward, self.terminated, self.truncated, {}

    def check_win(self):
        """对方落子后，检查胜利情况，并设置terminated，winner"""
        if self.last_action:
            # 和棋
            if self._is_draw():
                self.set_winner(-1)
            # 检查是否连成5子
            elif self.get_win_stones():
                self.set_winner(1 - self.current_player)

    def set_winner(self, winner):
        self.terminated = True
        self.winner = winner

    def coordinate2action(self, row, col) -> int:
        """从 (row, col) 坐标获取动作编号"""
        return row * self.shape[1] + col

    def action2coordinate(self, move) -> tuple[int, int]:
        """从动作编号获取坐标 (row, col)"""
        return divmod(move, self.shape[1])

    def describe_move(self, action):
        row, col = self.action2coordinate(action)
        print(f'选择落子：({row + 1},{col + 1})')

    def _is_draw(self):
        return np.all(np.logical_or(self.state[:, :, 0], self.state[:, :, 1]))

    def get_win_stones(self):
        """在获胜的情况下获取连成5子的棋子"""
        h, w, _ = self.shape
        h0, w0 = self.action2coordinate(self.last_action)
        for dh, dw in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
            stones = [(h0, w0)]
            for direction in (-1, 1):
                for step in range(1, 5):
                    i, j = h0 + step * dh * direction, w0 + step * dw * direction
                    if 0 <= i < h and 0 <= j < w and self.state[i, j, 1]:
                        stones.append((i, j))
                        if len(stones) == 5:
                            return stones
                    else:
                        break
        return None

    def render(self) -> None:
        """打印棋盘"""
        board_str = ''
        col_indices = [str(i + 1) for i in range(self.shape[1])]
        head = '  '
        for idx in col_indices:
            head += f'{idx:>3}'
        board_str += head + '\n'
        for i, row in enumerate(self.state):
            board_str += f'{i + 1:>2}  ' + '  '.join(
                ['X' if cell[self.current_player] else 'O' if cell[1 - self.current_player] else '.'
                 for cell in row]) + '\n'
            # 用红色显示玩家 1 的棋子 (1)
        board_str = re.sub(r'\bX\b', '\033[31mX\033[0m', board_str)
        # 用蓝色显示玩家 2 的棋子 (2)
        board_str = re.sub(r'\bO\b', '\033[34mO\033[0m', board_str)
        print(board_str)

    def handle_human_input(self):
        self.render()
        while True:
            while True:
                txt = input('输入落子位置坐标，示例"1,2"代表第1行第2列:')
                txt = txt.replace('，', ',')
                pos = txt.split(',')
                if len(pos) == 2 and type(pos) is list and pos[0].isdigit() and pos[1].isdigit():
                    break
                else:
                    print("输入格式有误，请输入行列编号，逗号隔开。")

            pos = tuple(int(i) - 1 for i in pos)
            action = env.coordinate2action(*pos)
            if action in env.valid_actions():
                break
            else:
                print("输入位置不合法，请重新输入！")
        return action


class GomokuUI:
    def __init__(self, rows=15, columns=15, players=None):
        self.black_piece = pygame.image.load('graphics/gomoku/black.png')
        self.white_piece = pygame.image.load('graphics/gomoku/white.png')
        self.mark_pic = pygame.image.load('graphics/gomoku/circle.png')
        self.image = pygame.image.load('graphics/gomoku/chessBoard.jpeg')
        self.place_sound = pygame.mixer.Sound('sound/place_stone.mp3')
        self.win_sound = pygame.mixer.Sound('sound/win.mp3')
        pygame.mixer.music.load('sound/bgm.mp3')
        pygame.mixer.music.set_volume(0.5)
        self.screen = pygame.display.get_surface()
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.env = GomokuEnv(rows, columns)
        self.players = players
        self.start_btn = Button("Start", self.start, (200, 680), color='green')
        self.reverse_player_btn = Button(f'First:{players[0].description}', self.reverse_player, pos=(200, 740),
                                         color='grey')
        self.status = 'new'  # new playing finished
        # 0,1分别代表黑方白方
        self.timers = {0: Timer(limit=60000, func=self.time_up), 1: Timer(limit=60000, func=self.time_up)}
        self.history = []

    def handle_input(self, event):
        if self.status == 'playing':
            self.players[self.env.current_player].handle_input(event)
        else:
            self.start_btn.handle_input(event)
            self.reverse_player_btn.handle_input(event)

    def update(self):
        if self.status == 'playing':
            player = self.players[self.env.current_player]
            if player.pending_action is None:
                player.update(self.env)
            else:
                row, col = self.env.action2coordinate(player.pending_action)
                self.history.append((row, col, self.env.current_player))
                self.env.step(player.pending_action)
                player.pending_action = None
                self.place_sound.play()
                if self.env.terminated or self.env.truncated:
                    self.set_win_status()
                else:
                    self.switch_side()

    def draw(self):
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_boundary()
        self.draw_pieces()
        self.draw_last_mark()

        if self.status == 'finished':
            self.draw_step_mark()
            self.draw_victory_badge()
            if self.timers[self.env.current_player].remain > 0:
                self.draw_victory_stones()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        elif self.status == 'new':
            self.draw_new_game_title()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        else:
            self.draw_player()
            self.draw_cursor()

    def time_up(self):
        for side, timer in self.timers.items():
            if timer.remain == 0:
                self.env.set_winner(1 - side)
                self.set_win_status()

    def switch_side(self):
        current_player = self.env.current_player
        prev_player = 1 - current_player
        # 重设timer
        self.timers[prev_player].reset()
        self.timers[current_player].activate()
        # 修改玩家状态
        self.players[prev_player].is_active = False
        self.players[current_player].is_active = True

    def set_win_status(self):
        self.status = 'finished'
        self.start_btn.text = "Restart"
        pygame.mixer.music.stop()
        self.win_sound.play()

    def draw_pieces(self):

        # 获取所有白棋和黑棋的位置
        white_positions = np.argwhere(self.env.state[:, :, 1 - self.env.current_player])
        black_positions = np.argwhere(self.env.state[:, :, self.env.current_player])

        # 绘制所有白棋
        for pos in white_positions:
            x, y = self._index2pos(row=pos[0], col=pos[1])
            self.screen.blit(self.white_piece, (x, y))

        # 绘制所有黑棋
        for pos in black_positions:
            x, y = self._index2pos(row=pos[0], col=pos[1])
            self.screen.blit(self.black_piece, (x, y))

    def _index2pos(self, row: int, col: int) -> tuple[float, float]:
        rows, columns, _ = self.env.shape
        y = (row - rows // 2) * settings['grid_size'] + self.rect.centery - 17
        x = (col - columns // 2) * settings['grid_size'] + self.rect.centerx - 17
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

    def draw_victory_badge(self):
        winner = 'black' if self.env.winner == 0 else 'white' if self.env.winner == 1 else 'draw'
        path = f'graphics/gomoku/{winner}_win.png'
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (200, 200))
        x = self.rect.centerx
        y = 60
        rect = image.get_rect(center=(x, y))
        self.screen.blit(image, rect.topleft)
        font = pygame.font.Font(None, 36)
        text = font.render(f'winner:{self.players[1 - self.env.current_player].description}', True, (255, 255, 30))
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def draw_victory_stones(self):
        for row, col in self.env.get_win_stones():
            x, y = self._index2pos(row, col)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_new_game_title(self):
        font = pygame.font.Font(None, 70)
        text = font.render('New Game', True, 'orange')
        x = self.rect.centerx
        y = 60
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def start(self):
        if self.status == 'finished':
            self.env.reset()
            for timer in self.timers.values():
                timer.reset()
        self.history = []
        self.status = 'playing'
        # 重设玩家
        for player in self.players:
            player.reset()
        # 当前玩家开始计时
        self.players[self.env.current_player].is_active = True
        self.timers[self.env.current_player].activate()
        pygame.mixer.music.play()

    def reverse_player(self):
        self.players.reverse()
        self.reverse_player_btn.text = f'First: {self.players[0].description}'

    def draw_player(self):
        self.timers[self.env.current_player].update()
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
        self.players[self.env.current_player].draw()

    def draw_boundary(self):
        if self.env.shape[:2] == (15, 15):
            return
        # 外部轮廓
        x0 = -7 * settings['grid_size'] + self.rect.centerx - 17
        y0 = -7 * settings['grid_size'] + self.rect.centery - 17
        w0 = settings['grid_size'] * 15
        h0 = settings['grid_size'] * 15
        # 可下棋区域
        x, y = self._index2pos(row=0, col=0)
        h, w, _ = self.env.shape
        w *= settings['grid_size']
        h *= settings['grid_size']
        overlay = pygame.Surface((w0, h0), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # 半透明黑色 (RGBA)
        pygame.draw.rect(overlay, 'green', (x - x0 - 1, y - y0 - 1, w + 2, h + 2), 2)
        # 在半透明表面上绘制一个完全透明的矩形
        pygame.draw.rect(overlay, (0, 0, 0, 0), (x - x0, y - y0, w, h))
        self.screen.blit(overlay, (x0, y0))


if __name__ == '__main__':
    env = ChineseChess()
    infer1, infer2 = IE.make_engine(209), IE.make_engine(120)
    competitors = [Human(), AIServer(infer2)]
    winner = env.run(competitors)
    env.render()
    if winner == -1:
        print('平局')
    else:
        print(f'玩家1:{competitors[winner].description} 获胜')
    infer1.shutdown()
    infer2.shutdown()
