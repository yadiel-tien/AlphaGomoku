import json
import threading
import random

import pygame
import requests
from config import CONFIG, SETTINGS
from deepMcts import NeuronMCTS
from mcts import MCTS


class Player:
    def __init__(self):
        self.is_active = False
        self.pending_action = None
        self._thinking = False
        self.description = 'Player'

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
        self.cursor = pygame.image.load('graphics/gomoku/cursor.png')
        self.screen = pygame.display.get_surface()
        self.cursor_pos = -1, -1
        self.rows, self.columns = shape
        self.center_x, self.center_y = None, None
        self.description = 'Human'

    def _is_cursor_valid(self) -> bool:
        """判断当前光标位置是否在棋盘上"""
        x, y = self.cursor_pos
        return 0 <= x < self.rows and 0 <= y < self.columns

    def handle_input(self, event):
        if self.is_active:
            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                self.cursor_pos = self._pos2index(event.pos)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self._is_cursor_valid():
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
        return env.handle_human_input()

    def draw(self):
        if self._is_cursor_valid():
            self.screen.blit(self.cursor, self._index2pos(*self.cursor_pos))

    def _pos2index(self, pos: tuple[int, int]) -> tuple[int, int]:
        if self.center_x is None:
            self.center_x, self.center_y = self.screen.get_rect().center
        col = round((pos[0] - self.center_x) / SETTINGS['grid_size']) + self.columns // 2
        row = round((pos[1] - self.center_y) / SETTINGS['grid_size']) + self.rows // 2
        return row, col

    def _index2pos(self, row, col) -> tuple[int, int]:
        if self.center_x is None:
            self.center_x, self.center_y = self.screen.get_rect().center
        y = (row - self.rows // 2) * SETTINGS['grid_size'] + self.center_y - 18
        x = (col - self.columns // 2) * SETTINGS['grid_size'] + self.center_x - 18
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
        self.description = f'AI({model_idx})'

    def update(self, env):
        if not self._thinking:
            # 新线程运行MCTS
            self._thinking = True
            threading.Thread(target=self.request_move, args=(env.state, env.last_action)).start()

    def request_move(self, state, last_action):
        url = CONFIG['base_url'] + 'make_move'
        payload = {
            'state': state.tolist(),
            'last_action': last_action,
            'player_idx': self.player_idx
        }
        response = self.post_request(url, payload)

        self.pending_action = response.get('action')
        self._thinking = False

    def request_reset(self):
        url = CONFIG['base_url'] + 'reset'
        payload = {'player_idx': self.player_idx}
        t = (threading.Thread(target=self.post_request, args=(url, payload), name='request reset'))
        t.start()
        t.join()

    def request_setup(self):
        url = CONFIG['base_url'] + 'setup'
        payload = {'player_idx': self.player_idx, 'model_idx': self.model_idx}
        t = threading.Thread(target=self.post_request, args=(url, payload), name='request setup')
        t.start()
        t.join()

    def post_request(self, url, payload):
        try:
            headers = {'content-type': 'application/json'}
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=10  # 添加超时设置
            )
            response.raise_for_status()  # 自动处理4xx/5xx错误

            return response.json()

        except requests.exceptions.HTTPError as http_err:
            error_msg = f'HTTP错误 ({response.status_code}): '
            try:
                error_data = response.json()
                error_msg += str(error_data.get('error', error_data))
            except ValueError:
                error_msg += response.text or str(http_err)
            print(error_msg)
            raise  # 重新抛出异常

        except json.JSONDecodeError as json_err:
            error_msg = f'响应不是有效的JSON: {str(json_err)}'
            print(error_msg)
            raise requests.exceptions.RequestException(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f'请求失败: {str(e)}'
            print(error_msg)
            raise  # 重新抛出异常

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
        self.description = f'Server {self.infer.model_index}'

    def get_action(self, env):
        if not self.silent:
            print('思考中...')
        self.run_mcts(env)
        if not self.silent:
            env.describe_move(int(self.pending_action))
        return self.pending_action

    def run_mcts(self, env):
        if self.mcts is None:
            self.infer.start()
            self.mcts = NeuronMCTS(env, self.infer)
        else:
            self.mcts.apply_action(env)
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
        self.description = 'MCTS'

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
            self._thread = threading.Thread(target=self._run_mcts, args=(env.state.copy(), env.last_action),
                                            name='MCTS run')
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
