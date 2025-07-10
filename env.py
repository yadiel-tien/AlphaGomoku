import typing
from abc import ABC, abstractmethod
import random
from typing import TypeAlias, Literal

import gymnasium as gym
import numpy as np
import pygame
from numpy.typing import NDArray

from ui import Button
from utils import Timer

if typing.TYPE_CHECKING:
    from player import Player


class BaseEnv(gym.Env, ABC):
    n_actions: int

    def __init__(self):
        self.state: NDArray[np.float32] | None = None
        self.shape: tuple[int, int, int] = (0, 0, 0)
        self.last_action: int | None = None
        self.player_to_move: int = 0  # 当前走棋方，五子棋0代表黑，1代表白。象棋0代表红，1代表黑
        self.winner: int = 2  # 0,1代表获胜玩家，-1代表平局，2代表未决胜负
        self.terminated: bool = False
        self.truncated: bool = False

    @classmethod
    @abstractmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        """无UI的情况下，通过控制台交互，将用户输入转换为动作输出"""
        ...

    @classmethod
    @abstractmethod
    def describe_move(cls, state: NDArray, action: int) -> None:
        """用于无UI界面，在控制台描述所走棋步"""
        ...

    @classmethod
    @abstractmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        """将逻辑state转换为适合神经网络的one-hot state"""
        ...

    @classmethod
    @abstractmethod
    def step_fn(cls, state: NDArray, action: int, player_to_move: int) -> tuple[
        NDArray[np.float32], int, bool, bool, dict]:
        """脱离环境的step方法，可用于MCTS"""
        ...

    def run(self, players: list['Player'], silent: bool = False) -> int:
        """模拟玩家比赛，玩家1胜返回0，玩家2胜返回1，平局返回-1"""
        self.reset()
        for player in players:
            player.silent = silent
            player.reset()

        index = 0
        while True:
            if not silent:
                print(f'-----player{index + 1}-----')

            action = players[index].get_action(self.state, self.last_action, self.player_to_move)
            _, reward, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                outcome = self.winner
                break
            index = 1 - index
        return outcome

    def random_order_play(self, players: list['Player'], silent: bool = False) -> int:
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

    def evaluate(self, players: list['Player'], n_rounds: int = 100) -> list[int]:
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

    @property
    def valid_actions(self) -> NDArray[np.int_]:
        return self.get_valid_actions(self.state, self.player_to_move)

    @classmethod
    @abstractmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """获取合法动作的类方法"""
        ...

    def set_winner(self, winner: int) -> None:
        """胜负已分，设置winner，终止游戏terminated = True"""
        self.terminated = True
        self.winner = winner

    @classmethod
    @abstractmethod
    def check_winner(cls, state: NDArray, perspective_player: int, action_just_executed: int) -> int:
        """检查胜负情况，相对于perspective_player来说
               :param action_just_executed: 刚刚做过的动作
               :param state: 棋盘表示
               :param perspective_player:相对于这个玩家来说的结果， 0或1
               :return: 1胜，0平，-1负, 2未分胜负"""
        ...

    @classmethod
    @abstractmethod
    def virtual_step(cls, state: NDArray[np.float32], action: int) -> NDArray[np.float32]:
        """只改变state，不计算输赢和奖励"""
        ...

    @classmethod
    @abstractmethod
    def action2move(cls, action: int) -> tuple[int, ...]:
        ...

    @classmethod
    @abstractmethod
    def move2action(cls, move: tuple[int, ...]) -> int:
        ...

    def reset_status(self) -> None:
        self.player_to_move = 0
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.winner = 2


GameStatus: TypeAlias = Literal['new', 'playing', 'finished']


class GameUI(ABC):
    def __init__(self, env: BaseEnv, players: list["Player"], img_path: str) -> None:

        self.env = env
        self.players = players
        self.status: GameStatus = 'new'
        self.piece_sound = pygame.mixer.Sound('sound/place_stone.mp3')
        self.win_sound = pygame.mixer.Sound('sound/win.mp3')
        pygame.mixer.music.load('sound/bgm.mp3')
        pygame.mixer.music.set_volume(0.5)
        self.start_btn = Button("Start", self.start, (200, 680), color='green')
        self.reverse_player_btn = Button(f'First:{players[0].description}', self.reverse_player, pos=(200, 740),
                                         color='grey')
        self.timers = {0: Timer(limit=60000, func=self.time_up), 1: Timer(limit=60000, func=self.time_up)}
        self.history = []
        self.screen = pygame.display.get_surface()
        self.image = pygame.image.load(img_path)
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.cursor_grid: tuple[int, int] | None = None
        self.settings = {}

    def handle_input(self, event: pygame.event.Event) -> None:
        if self.status == 'playing':
            player = self.players[self.env.player_to_move]
            from player import Human  # 放在这里是延迟导入，避免循环依赖
            if isinstance(player, Human):  # 处理人类玩家交互
                if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                    self.cursor_grid = self._pos2grid(event.pos)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    player.selected_grid = self.cursor_grid
                    self.handle_human_input()
        else:
            self.start_btn.handle_input(event)
            self.reverse_player_btn.handle_input(event)

    def update(self) -> None:
        if self.status == 'playing':
            player = self.players[self.env.player_to_move]
            action = player.pending_action
            if action != -1:
                self.history.append((action, self.env.player_to_move))
                self.env.step(action)
                self.play_place_sound(action)
                player.pending_action = -1
                if self.env.terminated or self.env.truncated:
                    self.set_win_status()
                else:
                    self.switch_side()
            else:
                player.update(self.env)

    @abstractmethod
    def handle_human_input(self) -> None:
        ...

    @abstractmethod
    def play_place_sound(self, action: int) -> None:
        """执行action时播放的音效"""

    def switch_side(self):
        current_player = self.env.player_to_move
        prev_player = 1 - current_player
        # 重设timer
        self.timers[prev_player].reset()
        self.timers[current_player].activate()

    def time_up(self):
        for side, timer in self.timers.items():
            if timer.remain == 0:
                self.env.set_winner(1 - side)
                self.set_win_status()

    def set_win_status(self):
        self.status = 'finished'
        self.start_btn.text = "Restart"
        pygame.mixer.music.stop()
        self.win_sound.play()

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
        self.timers[self.env.player_to_move].activate()
        pygame.mixer.music.play()

    def reverse_player(self):
        self.players.reverse()
        self.reverse_player_btn.text = f'First: {self.players[0].description}'

    def draw_victory(self, path):
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (200, 200))
        x = self.rect.centerx
        y = 60
        rect = image.get_rect(center=(x, y))
        self.screen.blit(image, rect.topleft)
        font = pygame.font.Font(None, 36)
        text = font.render(f'winner:{self.players[1 - self.env.player_to_move].description}', True, (255, 255, 30))
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def draw_new_game_title(self):
        font = pygame.font.Font(None, 70)
        text = font.render('New Game', True, 'orange')
        x = self.rect.centerx
        y = 60
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def draw_player(self):
        self.timers[self.env.player_to_move].update()
        white_sec = self.timers[1].remain // 1000
        black_sec = self.timers[0].remain // 1000
        font = pygame.font.Font(None, 60)
        white = font.render(f'{self.players[1].description}  White:{white_sec:02}', True, 'orange')
        black = font.render(f'{self.players[0].description} Black:{black_sec:02}', True, 'orange')
        white_rect = white.get_rect(midleft=(60, 60))
        black_rect = black.get_rect(midleft=(60, 740))
        self.screen.blit(white, white_rect.topleft)
        self.screen.blit(black, black_rect.topleft)

    def _pos2grid(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """根据屏幕坐标，返回棋盘位置，超出棋盘返回None
        :param pos:屏幕坐标(x,y)
        :return 棋盘坐标(row,col)
        """
        center_x, center_y = self.screen.get_rect().center
        rows, columns, _ = self.env.shape
        grid_size = self.settings['grid_size']

        col: int = round((pos[0] - center_x) / grid_size + (columns - 1) / 2)
        row: int = round((pos[1] - center_y) / grid_size + (rows - 1) / 2)

        return (row, col) if 0 <= col < columns and 0 <= row < rows else None

    def _grid2pos(self, grid: tuple[int, int]) -> tuple[int, int]:
        """
        根据棋盘交叉点位置返回该位置top left坐标。即以交叉点为中心，grid_size为边长的矩形左上角端点坐标。
        :param grid:棋盘坐标(row,col)
        :return 交叉点top left屏幕坐标(x,y)
        """
        row, col = grid
        center_x, center_y = self.screen.get_rect().center
        rows, columns, _ = self.env.shape
        grid_size = self.settings['grid_size']

        x = (col - (columns - 1) / 2) * grid_size + center_x - grid_size // 2
        y = (row - (rows - 1) / 2) * grid_size + center_y - grid_size // 2
        return int(x), int(y)
