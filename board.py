import re

import gymnasium as gym
from gymnasium import spaces
import pygame

from constant import BOARD_GRID_SIZE
from functions import is_win
from player import Human, AIServer
from ui import Button
import numpy as np

from utils import Timer


class GomokuEnv(gym.Env):
    def __init__(self, rows: int, columns: int):
        super().__init__()
        self.shape = rows, columns, 2
        self.state = None
        self.action_space = spaces.Discrete(rows * columns)
        self.observation_space = spaces.Box(0, 1, shape=self.shape, dtype=np.float32)
        self.current_player = 0  # 当前走棋方，0代表黑，1代表白
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
        self.last_action = None
        return self.state, {}

    def run(self, players, silent=False):
        """模拟玩家比赛，玩家1胜返回(1,0)，玩家2胜返回(0,1)，平局返回（0，0）"""
        index = 0
        for player in players:
            player.silent = silent

        while True:
            if not silent:
                print(f'-----player{index + 1}-----')

            action = players[index].get_action(self)

            _, reward, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                return self.get_winner(reward)

            index = 1 - index

    def get_winner(self, reward):
        """确保游戏结束后再调用。(1,0)玩家1胜利，(0,1)玩家2胜利,(0,0)平局"""
        winner = 1 - self.current_player if reward else -1
        winner = (1, 0) if winner == 0 else (0, 1) if winner == 1 else (0, 0)
        return winner

    def evaluate(self, players, n_rounds=100):
        """2玩家对弈，打印胜率"""
        outcomes = []

        for i in range(n_rounds):
            print(f'第{i + 1}局:', end=' ')
            self.reset()
            for player in players:
                player.reset()
            # 改变先手顺序
            current_players = reversed(players) if i % 2 == 1 else players
            outcome = self.run(current_players)
            # 还原对应胜利结果
            if i % 2 == 1:
                outcome = reversed(outcome)
            outcomes.append(outcome)
            print(outcome)

        print(f"Player 1 Win Percentage:{outcomes.count((1, 0)) / n_rounds:.2%}", )
        print(f"Player 2 Win Percentage:{outcomes.count((1, 0)) / n_rounds:.2%}")
        print(f"Draw Percentage:{outcomes.count((1, 0)) / n_rounds:.2%}")
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

        # 胜利奖励1，平局或未结束奖励0，隐含失败后对手奖励1，通过min-max自己奖励-1
        reward = 0
        if is_win(self.state, action):
            reward = 1
            self.terminated = True
        elif self._is_draw():
            self.terminated = True

        return self.state, reward, self.terminated, self.truncated, {}

    def coordinate2action(self, row, col) -> int:
        """从 (row, col) 坐标获取动作编号"""
        return row * self.shape[1] + col

    def action2coordinate(self, move) -> tuple[int, int]:
        """从动作编号获取坐标 (row, col)"""
        return divmod(move, self.shape[1])

    def _is_draw(self):
        return np.all(np.logical_or(self.state[:, :, 0], self.state[:, :, 1]))

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


class BoardUI:
    def __init__(self, rows=15, columns=15, players=None):
        self.black_piece = pygame.image.load('graphics/black.png')
        self.white_piece = pygame.image.load('graphics/white.png')
        self.mark_pic = pygame.image.load('graphics/circle.png')
        self.image = pygame.image.load('graphics/chessBoard.jpeg')
        self.screen = pygame.display.get_surface()
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.board = GomokuEnv(rows, columns)
        self.button = Button("Start", self.start, (200, 700), color='green')
        self.status = 'new'  # new playing finished
        # 0,1分别代表黑方白方
        self.timers = {0: Timer(limit=60000, func=self.time_up), 1: Timer(limit=60000, func=self.time_up)}
        self.players = players
        self.winner = None
        self.history = []

    def handle_input(self, event):
        if self.status == 'playing':
            self.players[self.board.current_player].handle_input(event)
        else:
            self.button.handle_input(event)

    def update(self):
        if self.status == 'playing':
            player = self.players[self.board.current_player]
            if player.pending_action is None:
                player.update(self.board)
            else:
                row, col = self.board.action2coordinate(player.pending_action)
                self.history.append((row, col, self.board.current_player))
                new_state, reward, terminated, truncated, _ = self.board.step(player.pending_action)
                player.pending_action = None

                if terminated or truncated:
                    winner = self.board.get_winner(reward)
                    winner = 'black' if winner == (1, 0) else 'white' if winner == (0, 1) else 'draw'
                    self.set_winner(winner)
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

        # 获取所有白棋和黑棋的位置
        white_positions = np.argwhere(self.board.state[:, :, 1 - self.board.current_player])
        black_positions = np.argwhere(self.board.state[:, :, self.board.current_player])

        # 绘制所有白棋
        for pos in white_positions:
            x, y = self._index2pos(row=pos[0], col=pos[1])
            self.screen.blit(self.white_piece, (x, y))

        # 绘制所有黑棋
        for pos in black_positions:
            x, y = self._index2pos(row=pos[0], col=pos[1])
            self.screen.blit(self.black_piece, (x, y))

    def _index2pos(self, row: int, col: int) -> tuple[float, float]:
        rows, columns, _ = self.board.shape
        y = (row - rows // 2) * BOARD_GRID_SIZE + self.rect.centery - 17
        x = (col - columns // 2) * BOARD_GRID_SIZE + self.rect.centerx - 17
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
        # 重设玩家
        for player in self.players.values():
            player.reset()
        # 当前玩家开始计时
        self.players[self.board.current_player].is_active = True
        self.timers[self.board.current_player].activate()

    def draw_player(self):
        self.timers[self.board.current_player].update()
        white_sec = self.timers[1].remain // 1000
        black_sec = self.timers[0].remain // 1000
        font = pygame.font.Font(None, 60)
        white = font.render(f'White:{white_sec:02}', True, 'orange')
        black = font.render(f'Black:{black_sec:02}', True, 'orange')
        white_rect = white.get_rect(midleft=(60, 60))
        black_rect = black.get_rect(midleft=(60, 740))
        self.screen.blit(white, white_rect.topleft)
        self.screen.blit(black, black_rect.topleft)

    def draw_cursor(self):
        self.players[self.board.current_player].draw()

    def draw_boundary(self):
        if self.board.shape[:2] == (15, 15):
            return
        # 外部轮廓
        x0 = -7 * BOARD_GRID_SIZE + self.rect.centerx - 17
        y0 = -7 * BOARD_GRID_SIZE + self.rect.centery - 17
        w0 = BOARD_GRID_SIZE * 15
        h0 = BOARD_GRID_SIZE * 15
        # 可下棋区域
        x, y = self._index2pos(row=0, col=0)
        h, w, _ = self.board.shape
        w *= BOARD_GRID_SIZE
        h *= BOARD_GRID_SIZE
        overlay = pygame.Surface((w0, h0), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # 半透明黑色 (RGBA)
        pygame.draw.rect(overlay, 'green', (x - x0 - 1, y - y0 - 1, w + 2, h + 2), 2)
        # 在半透明表面上绘制一个完全透明的矩形
        pygame.draw.rect(overlay, (0, 0, 0, 0), (x - x0, y - y0, w, h))
        self.screen.blit(overlay, (x0, y0))


if __name__ == '__main__':
    h, w = 9, 9
    env = GomokuEnv(h, w)
    competitors = [Human((h, w)), AIServer(model_id=291, iteration=1000)]
    result = env.run(competitors)
    env.render()
    if result == (1, 0):
        print(f'玩家1:{type(competitors[0])}获胜> ')
    elif result == (0, 1):
        print(f'玩家2:{type(competitors[1])}获胜')
    else:
        print('平局')
