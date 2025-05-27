import threading
import random

import pygame
import torch

from config import MODEL_PATH, DEVICE
from constant import BOARD_GRID_SIZE
from deepMcts import DeepMCTS
from inference import InferenceEngine
from mcts import MCTS
from network import Net


class Player:
    def __init__(self):
        self.is_active = False
        self.pending_action = None
        self._thinking = False

    def handle_input(self, event, env, last_move):
        pass

    def update(self, env, last_move):
        pass

    def draw(self):
        pass

    def get_action(self, env, last_action):
        pass

    def reset(self):
        self.is_active = False
        self.pending_action = None
        self._thinking = False


class Human(Player):
    def __init__(self, shape=(15, 15)):
        super().__init__()
        self.cursor = pygame.image.load('graphics/cursor.png')
        self.screen = pygame.display.get_surface()
        self.cursor_pos = -1, -1
        self.rows, self.columns = shape
        self.center_x, self.center_y = None, None

    def handle_input(self, event):
        if self.is_active:
            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                self.cursor_pos = self._pos2index(event.pos)

            if event.type == pygame.MOUSEBUTTONDOWN:
                self._thinking = True

    def update(self, env, last_move):
        if self.is_active:
            if self._thinking:
                if env.is_placeable(*self.cursor_pos):
                    self.pending_move = self.cursor_pos
                    self.cursor_pos = -1, -1
                self._thinking = False

    def get_action(self, env, last_action):
        env.render()
        while True:
            while True:
                txt = input('输入落子位置坐标，示例"1,2"代表第1行第2列:')
                txt.replace('，', ',')
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

    def draw(self):
        if self._onboard():
            self.screen.blit(self.cursor, self._index2pos(*self.cursor_pos))

    def _pos2index(self, pos: tuple[int, int]) -> tuple[int, int]:
        if self.center_x is None:
            self.center_x, self.center_y = self.screen.get_rect().center
        col = round((pos[0] - self.center_x) / BOARD_GRID_SIZE) + self.columns // 2
        row = round((pos[1] - self.center_y) / BOARD_GRID_SIZE) + self.rows // 2
        return row, col

    def _index2pos(self, row, col) -> tuple[int, int]:
        if self.center_x is None:
            self.center_x, self.center_y = self.screen.get_rect().center
        y = (row - self.rows // 2) * BOARD_GRID_SIZE + self.center_y - 18
        x = (col - self.columns // 2) * BOARD_GRID_SIZE + self.center_x - 18
        return x, y

    def _onboard(self) -> bool:
        return (0 <= self.cursor_pos[0] < self.rows) and (0 <= self.cursor_pos[1] < self.columns)

    def reset(self):
        super().reset()
        self.cursor_pos = -1, -1


class AI(Player):
    def __init__(self, shape=(9, 9), model_id=-1, iteration=1000, silent=False):
        super().__init__()
        model = Net(256, shape[0] * shape[1]).to(DEVICE)
        path = MODEL_PATH if model_id == -1 else f'./data/model_{model_id}.pt'
        model.load_state_dict(torch.load(path))
        self.infer = InferenceEngine(model)
        self._iteration = iteration
        self.mcts = None
        self._thread = None
        self.silent = silent

    def handle_input(self, event, board, last_move):
        """
        适配人类玩家，检查AI是否落子,落子后发布事件通知

        """
        pass

    def get_action(self, env, last_action):
        if not self.silent:
            print('思考中...')
        self._run_mcts(env.state, last_action)
        if not self.silent:
            h, w = env.action2coordinate(int(self.pending_action))
            print(f'选择落子：({h + 1},{w + 1})')
        return self.pending_action

    def _run_mcts(self, state, last_action):
        if self.mcts is None:
            self.mcts = DeepMCTS(state, self.infer)
        else:
            self.mcts.apply_action(last_action)
        # start = time.time()
        self.mcts.run(self._iteration)
        # print(f'{self._iteration} iteration took {time.time() - start:.2f} seconds')
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def draw(self):
        pass

    def reset(self):
        super().reset()
        self.mcts = None
        self._thread = None


class MCTSPlayer(Player):
    def __init__(self, iteration=1000):
        super().__init__()
        self.mcts = None
        self._thread = None
        self._iteration = iteration

    def handle_input(self, event):
        pass

    def get_action(self, env, last_action):
        print('思考中...')
        self._run_mcts(env.state, last_action)
        h, w = env.action2coordinate(int(self.pending_action))
        print(f'选择落子：({h + 1},{w + 1})')
        return self.pending_action

    def update(self, env, last_action):
        if not self._thinking:
            # 新线程运行MCTS
            self._thinking = True
            self._thread = threading.Thread(target=self._run_mcts, args=(env.state.copy(), last_action))
            self._thread.start()

    def _run_mcts(self, state, last_action):
        if self.mcts is None:
            self.mcts = MCTS(state)
        else:
            self.mcts.apply_opponent_action(state, last_action)
        # start = time.time()
        self.mcts.run(self._iteration)
        # print(f'{self._iteration} iteration took {time.time() - start:.2f} seconds')
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def draw(self):
        pass

    def reset(self):
        super().reset()
        self.mcts = None
        self._thread = None


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, env, last_action):
        valid_action = env.valid_actions()
        return random.choice(valid_action)

    def reset(self):
        pass
