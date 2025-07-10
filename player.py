from __future__ import annotations
import json
import threading
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from numpy.typing import NDArray

from config import CONFIG
from deepMcts import NeuronMCTS
from functions import get_class
from inference import InferenceEngine

if TYPE_CHECKING:
    from env import BaseEnv
from mcts import MCTS


class Player:
    def __init__(self, env_name: str):
        self.pending_action: int = -1
        self.is_thinking: bool = False
        self.description: str = 'Player'
        self.env_class = get_class(env_name)

    def update(self, env: BaseEnv) -> None:
        pass

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        pass

    def reset(self) -> None:
        self.pending_action = -1
        self.is_thinking = False


class Human(Player):
    def __init__(self, env_name: str):
        super().__init__(env_name)
        self.description = 'Human'
        self.selected_grid: tuple[int, int] | None = None

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        return self.env_class.handle_human_input(state, last_action, player_to_move)

    def reset(self) -> None:
        super().reset()
        self.selected_grid = None


class AIClient(Player):
    def __init__(self, player_idx: int, model_idx: int, env_name: str) -> None:
        super().__init__(env_name)
        self.player_idx = player_idx
        self.model_idx = model_idx
        self.request_setup()
        self.description = f'AI({model_idx})'

    def update(self, env: BaseEnv) -> None:
        """负责启动推理线程"""
        if not self.is_thinking:
            # 新线程运行MCTS
            self.is_thinking = True
            threading.Thread(target=self.request_move, args=(env.state, env.last_action, env.player_to_move)).start()

    def request_move(self, state: np.ndarray, last_action: int, player_to_move: int) -> None:
        """给server发请求，获取action"""
        url = CONFIG['base_url'] + 'make_move'
        payload = {
            'state': state.tolist(),
            'last_action': last_action,
            'player_idx': self.player_idx,
            'player_to_move': player_to_move
        }
        response = self.post_request(url, payload)

        self.pending_action = response.get('last_action')
        self.is_thinking = False

    def request_reset(self) -> None:
        """告知server重置"""
        url = CONFIG['base_url'] + 'reset'
        payload = {'player_idx': self.player_idx}
        t = (threading.Thread(target=self.post_request, args=(url, payload), name='request reset'))
        t.start()
        t.join()

    def request_setup(self) -> None:
        """告知server创建推理引擎"""
        url = CONFIG['base_url'] + 'setup'
        payload = {'player_idx': self.player_idx, 'model_idx': self.model_idx, 'env_class': self.env_class.__name__}
        t = threading.Thread(target=self.post_request, args=(url, payload), name='request setup')
        t.start()
        t.join()

    def post_request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求的基础方法，获取反馈，处理错误"""
        response = None
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
            error_msg = f'HTTP错误 ({response.status_code if response else "Unknown"}): '
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

    def reset(self) -> None:
        self.request_reset()
        super().reset()


class AIServer(Player):
    def __init__(self, infer_engine: InferenceEngine, env_name: str, n_simulation=500, silent=False) -> None:
        super().__init__(env_name)
        self.infer = infer_engine
        self._n_simulation = n_simulation
        self.mcts: NeuronMCTS | None = None
        self.silent = silent  # silent=True减少日志信息
        self.description = f'Server {self.infer.model_index}'

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        """获取动作"""
        if not self.silent:
            print('思考中...')
        self.run_mcts(state, last_action, player_to_move)
        if not self.silent:
            self.env_class.describe_move(state, int(self.pending_action))
        return self.pending_action

    def run_mcts(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """运行mcts，根据模拟结果选择最优动作"""
        if self.mcts is None:
            self.infer.start()
            self.mcts = NeuronMCTS(
                state=state,
                env_class=self.env_class,
                infer=self.infer,
                last_action=last_action,
                player_to_move=player_to_move,
            )
        else:
            self.mcts.apply_action(last_action)
        self.mcts.run(self._n_simulation)
        self.pending_action = self.mcts.choose_action()
        self.is_thinking = False

    def reset(self) -> None:
        super().reset()
        self.mcts = None


class MCTSPlayer(Player):
    def __init__(self, env_name: str, n_simulation=1000) -> None:
        super().__init__(env_name)
        self.mcts: MCTS | None = None
        self._thread: threading.Thread | None = None
        self._n_simulation = n_simulation
        self.description = 'MCTS'

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        print('思考中...')
        self._run_mcts(state, last_action)
        self.env_class.describe_move(state, self.pending_action)
        return self.pending_action

    def update(self, env: BaseEnv) -> None:
        if not self._thinking:
            # 新线程运行MCTS
            self.is_thinking = True
            self._thread = threading.Thread(target=self._run_mcts, args=(env.state.copy(), env.last_action),
                                            name='MCTS run')
            self._thread.start()

    def _run_mcts(self, state: np.ndarray, last_action: int) -> None:
        if self.mcts is None:
            self.mcts = MCTS(state)
        else:
            self.mcts.apply_opponent_action(state, last_action)
        # start = time.time()
        self.mcts.run(self._n_simulation)
        # print(f'{self._iteration} iteration took {time.time() - start:.2f} seconds')
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def reset(self) -> None:
        super().reset()
        self.mcts = None
        self._thread = None


class RandomPlayer(Player):
    def __init__(self, env_name: str) -> None:
        super().__init__(env_name)

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        return random.choice(self.env_class.get_valid_actions(state, player_to_move))
