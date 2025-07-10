import re

from gymnasium import spaces
import pygame

from numpy.typing import NDArray

from chess import ChineseChess
from config import CONFIG, GameConfig
from env import BaseEnv, GameUI
from inference import InferenceEngine as Engine
from player import Human, AIServer, Player
import numpy as np
from typing import cast, TypeAlias

settings: GameConfig = CONFIG['Gomoku']
GomokuMove: TypeAlias = tuple[int, int]


class Gomoku(BaseEnv):
    shape: tuple[int, int, int] = 15, 15, 2

    def __init__(self, rows: int = 15, columns: int = 15):
        super().__init__()
        self.shape: tuple[int, int, int] = rows, columns, 2
        self.action_space = spaces.Discrete(rows * columns)
        self.n_actions = rows * columns
        self.observation_space = spaces.Box(0, 1, shape=self.shape, dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[NDArray[np.int_], dict]:
        """
        重置游戏, 返回当前棋盘状态
        state=[Xt, Yt]
        Xt 1代表有当前玩家棋子，0代表无当前玩家棋子，Yt 1代表有对手玩家棋子，0代表无对手玩家棋子
        """
        self.state = np.zeros(self.shape, dtype=np.float32)
        self.reset_status()
        return self.state, {}

    @classmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        return state

    @classmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        state = state[:, :, 0] + state[:, :, 1]
        return np.flatnonzero(state == 0)

    @classmethod
    def virtual_step(cls, state: NDArray[np.float32], action: int) -> NDArray[np.float32]:
        """只改变state，不计算输赢和奖励"""
        new_state = np.copy(state)
        row, col = cls.action2move(action)
        # 执行落子
        new_state[row, col, 0] = 1

        # 更改棋盘和当前玩家
        new_state[:, :, [0, 1]] = new_state[:, :, [1, 0]]
        return new_state

    @classmethod
    def step_fn(cls, state: NDArray[np.float32], action: int, player_to_move: int) -> tuple[
        NDArray[np.float32], int, bool, bool, dict]:
        """脱离环境进行step"""
        new_state = cls.virtual_step(state, action)
        player_just_moved = player_to_move
        winner = cls.check_winner(new_state, player_just_moved, action)
        # 胜利奖励1，平局或未结束奖励0，隐含失败后对手奖励1，通过min-max自己奖励-1
        reward = 0
        terminated = False
        if winner == player_just_moved:  # 落子导致胜利
            reward = 1
        if winner != 2:
            terminated = True

        return new_state, reward, terminated, False, {}

    def step(self, action: int) -> tuple[NDArray[np.float32], int, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        # 执行落子
        self.state, reward, terminated, _, _ = self.step_fn(self.state, action, self.player_to_move)
        if terminated:
            winner = self.player_to_move if reward == 1 else -1
            self.set_winner(winner)

        # 更改玩家
        self.player_to_move = 1 - self.player_to_move
        self.last_action = action

        # 胜利奖励1，平局或未结束奖励0，隐含失败后对手奖励1，通过min-max自己奖励-1
        reward = 0
        if self.winner == 1 - self.player_to_move:  # 落子导致胜利
            reward = 1

        return self.state, reward, terminated, self.truncated, {}

    @classmethod
    def check_winner(cls, state: NDArray, player_just_moved: int, action_just_executed: int) -> int:
        """:return 0，1玩家，2未分胜负，-1和"""
        # 和棋
        if cls._is_draw(state):
            return -1
        # 检查是否连成5子
        if cls.get_win_stones(state, action_just_executed):
            return player_just_moved
        return 2

    @classmethod
    def move2action(cls, move: GomokuMove) -> int:
        """从 (row, col) 坐标获取动作编号"""
        row, col = move
        return row * cls.shape[1] + col

    @classmethod
    def action2move(cls, action: int) -> GomokuMove:
        """从动作编号获取坐标 (row, col)"""
        return divmod(action, cls.shape[1])

    @classmethod
    def describe_move(cls, state: NDArray, action: int) -> None:
        """无UI对弈时，打印描述行棋的说明"""
        row, col = cls.action2move(action)
        print(f'选择落子：({row + 1},{col + 1})')

    @staticmethod
    def _is_draw(state: NDArray) -> bool:
        return np.all(np.logical_or(state[:, :, 0], state[:, :, 1]))

    @classmethod
    def get_win_stones(cls, state: NDArray, action_just_executed: int) -> list[tuple[int, int]]:
        """落子后检查获胜的情况，获胜返回连成5子的棋子位置，未获胜返回空列表[]"""
        h, w, _ = cls.shape
        h0, w0 = cls.action2move(action_just_executed)
        for dh, dw in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
            stones = [(h0, w0)]
            for direction in (-1, 1):
                for step in range(1, 5):
                    i, j = h0 + step * dh * direction, w0 + step * dw * direction
                    if 0 <= i < h and 0 <= j < w and state[i, j, 1]:
                        stones.append((i, j))
                        if len(stones) == 5:
                            return stones
                    else:
                        break
        return []

    def render(self) -> None:
        self.render_fn(self.state, self.player_to_move)

    @classmethod
    def render_fn(cls, state: NDArray, player_to_move: int) -> None:
        """打印棋盘"""
        board_str = ''
        col_indices = [str(i + 1) for i in range(cls.shape[1])]
        head = '  '
        for idx in col_indices:
            head += f'{idx:>3}'
        board_str += head + '\n'
        for i, row in enumerate(state):
            board_str += f'{i + 1:>2}  ' + '  '.join(
                ['X' if cell[player_to_move] else 'O' if cell[1 - player_to_move] else '.'
                 for cell in row]) + '\n'
            # 用红色显示玩家 1 的棋子 (1)
        board_str = re.sub(r'\bX\b', '\033[31mX\033[0m', board_str)
        # 用蓝色显示玩家 2 的棋子 (2)
        board_str = re.sub(r'\bO\b', '\033[34mO\033[0m', board_str)
        print(board_str)

    @classmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        cls.render_fn(state, player_to_move)
        valid_actions = cls.get_valid_actions(state, player_to_move)
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
            action = cls.move2action(*pos)
            if action in valid_actions:
                break
            else:
                print("输入位置不合法，请重新输入！")
        return action


class GomokuUI(GameUI):
    def __init__(self, players: list[Player], rows=15, columns=15):
        super().__init__(Gomoku(rows, columns), players, settings['img_path'])
        self.black_piece = pygame.image.load('graphics/gomoku/black.png')
        self.white_piece = pygame.image.load('graphics/gomoku/white.png')
        self.mark_pic = pygame.image.load('graphics/gomoku/circle.png')
        self.cursor_pic = pygame.image.load('graphics/gomoku/cursor.png')
        self.env = cast(Gomoku, self.env)
        self.settings = settings

    def handle_human_input(self) -> None:
        player = cast(Human, self.players[self.env.player_to_move])
        if player.selected_grid is None:
            return
        action = self.env.move2action(player.selected_grid)
        if action in self.env.valid_actions:
            player.pending_action = action

    def play_place_sound(self, action: int) -> None:
        self.piece_sound.play()

    def draw(self):
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_boundary()
        self.draw_pieces()
        self.draw_last_mark()

        if self.status == 'finished':
            self.draw_step_mark()
            self.draw_victory_badge()
            if self.timers[self.env.player_to_move].remain > 0:
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

    def draw_pieces(self):

        # 获取所有白棋和黑棋的位置
        white_positions = np.argwhere(self.env.state[:, :, 1 - self.env.player_to_move])
        black_positions = np.argwhere(self.env.state[:, :, self.env.player_to_move])

        # 绘制所有白棋
        for pos in white_positions:
            x, y = self._grid2pos(pos)
            self.screen.blit(self.white_piece, (x, y))

        # 绘制所有黑棋
        for pos in black_positions:
            x, y = self._grid2pos(pos)
            self.screen.blit(self.black_piece, (x, y))

    def draw_step_mark(self):
        font = pygame.font.Font(None, 16)
        for idx, (action, mark) in enumerate(self.history):
            color = 'black' if mark else 'white'  # 黑棋则白字，白棋则黑字
            text = font.render(str(idx + 1), True, color)
            x, y = self._grid2pos(self.env.action2move(action))
            rect = text.get_rect(center=(x + 17, y + 17))
            self.screen.blit(text, rect.topleft)

    def draw_last_mark(self):
        if self.history:
            action, _ = self.history[-1]
            move = self.env.action2move(action)
            x, y = self._grid2pos(move)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_victory_badge(self):
        winner = 'black' if self.env.winner == 0 else 'white' if self.env.winner == 1 else 'draw'
        path = f'graphics/gomoku/{winner}_win.png'
        self.draw_victory(path)

    def draw_victory_stones(self):
        for grid in self.env.get_win_stones(self.env.state, self.env.last_action):
            x, y = self._grid2pos(grid)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_cursor(self):
        if self.cursor_grid is not None:
            x, y = self._grid2pos(self.cursor_grid)
            self.screen.blit(self.cursor_pic, (x, y))

    def draw_boundary(self):
        if self.env.shape[:2] == (15, 15):
            return
        # 外部轮廓
        x0 = -7 * settings['grid_size'] + self.rect.centerx - 17
        y0 = -7 * settings['grid_size'] + self.rect.centery - 17
        w0 = settings['grid_size'] * 15
        h0 = settings['grid_size'] * 15
        # 可下棋区域
        x, y = self._grid2pos((0, 0))
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
    infer1, infer2 = Engine.make_engine(209), Engine.make_engine(120)
    competitors = [Human(), AIServer(infer2)]
    victor = env.run(competitors)
    env.render()
    if victor == -1:
        print('平局')
    else:
        print(f'玩家1:{competitors[victor].description} 获胜')
    infer1.shutdown()
    infer2.shutdown()
