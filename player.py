import asyncio
import json
import threading
import random

import aiohttp
import requests
import pygame
import torch

from config import MODEL_PATH, DEVICE, BASE_URL
from constant import BOARD_GRID_SIZE
from deepMcts import NeuronMCTS
from functions import is_onboard
from inference import InferenceEngine
from mcts import MCTS
from network import Net


class Player:
    def __init__(self):
        self.is_active = False
        self.pending_action = None
        self._thinking = False

    def handle_input(self, event):
        pass

    def update(self, env):
        pass

    def draw(self):
        pass

    def get_action(self, env):
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
                if is_onboard(*self.cursor_pos, self.rows, self.columns):
                    self._thinking = True
                else:
                    self._thinking = False

    def update(self, env):
        if self.is_active and self._thinking:
            action = env.coordinate2action(*self.cursor_pos)
            if action in env.valid_actions():
                self.pending_action = action
                self.cursor_pos = -1, -1
            self._thinking = False

    def get_action(self, env):
        env.render()
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

    def draw(self):
        if is_onboard(*self.cursor_pos, self.rows, self.columns):
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

    def reset(self):
        super().reset()
        self.cursor_pos = -1, -1


class AIClient(Player):
    def __init__(self, player_idx, model_idx):
        super().__init__()
        self.player_idx = player_idx
        self.model_idx = model_idx
        self.request_setup()

    def update(self, env):
        if not self._thinking:
            # 新线程运行MCTS
            self._thinking = True
            threading.Thread(target=self.request_move, args=(env.state, env.last_action)).start()

    def request_move(self, state, last_action):
        url = BASE_URL + 'make_move'
        payload = {
            'state': state.tolist(),
            'last_action': last_action,
            'player_idx': self.player_idx
        }
        response = self.post_request(url, payload)

        self.pending_action = response.get('action')
        self._thinking = False

    def request_reset(self):
        url = BASE_URL + 'reset'
        payload = {'player_idx': self.player_idx}
        threading.Thread(target=self.post_request, args=(url, payload)).start()

    def request_setup(self):
        url = BASE_URL + 'setup'
        payload = {'player_idx': self.player_idx, 'model_idx': self.model_idx}
        threading.Thread(target=self.post_request, args=(url, payload)).start()

    def post_request(self, url, payload):
        try:
            headers = {'content-type': 'application/json'}
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'发生错误：{e}')
        return None

    def reset(self):
        self.request_reset()
        super().reset()


class AIServer(Player):
    def __init__(self, infer_engine, n_simulation=500, silent=False):
        super().__init__()
        self.infer = infer_engine
        self._n_simulation = n_simulation
        self.mcts = None
        self.silent = silent

    def get_action(self, env):
        if not self.silent:
            print('思考中...')
        self.run_mcts(env.state, env.last_action)
        if not self.silent:
            h, w = env.action2coordinate(int(self.pending_action))
            print(f'选择落子：({h + 1},{w + 1})')
        return self.pending_action

    def run_mcts(self, state, last_action):
        if self.mcts is None:
            self.mcts = NeuronMCTS(state, self.infer)
        else:
            self.mcts.apply_action(state, last_action)
        self.mcts.run(self._n_simulation)
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def reset(self):
        super().reset()
        self.mcts = None


class MCTSPlayer(Player):
    def __init__(self, n_simulation=1000):
        super().__init__()
        self.mcts = None
        self._thread = None
        self._n_simulation = n_simulation

    def get_action(self, env):
        print('思考中...')
        self._run_mcts(env.state, env.last_action)
        h, w = env.action2coordinate(int(self.pending_action))
        print(f'选择落子：({h + 1},{w + 1})')
        return self.pending_action

    def update(self, env):
        if not self._thinking:
            # 新线程运行MCTS
            self._thinking = True
            self._thread = threading.Thread(target=self._run_mcts, args=(env.state.copy(), env.last_action))
            self._thread.start()

    def _run_mcts(self, state, last_action):
        if self.mcts is None:
            self.mcts = MCTS(state)
        else:
            self.mcts.apply_opponent_action(state, last_action)
        # start = time.time()
        self.mcts.run(self._n_simulation)
        # print(f'{self._iteration} iteration took {time.time() - start:.2f} seconds')
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def reset(self):
        super().reset()
        self.mcts = None
        self._thread = None


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, env):
        valid_action = env.valid_actions()
        return random.choice(valid_action)
