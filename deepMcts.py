import collections
from typing import Self

import numpy as np
from numpy.typing import NDArray

from env import BaseEnv
from inference import InferenceEngine


class DummyNode(object):
    """作为根节点的父节点使用，简化逻辑"""

    def __init__(self):
        self.parent = None
        self.child_n = collections.defaultdict(float)
        self.child_w = collections.defaultdict(float)


class NeuronNode:
    def __init__(self, state: NDArray, action_just_executed: int, player_to_move: int,
                 env_class: type[BaseEnv],
                 parent: Self = None,
                 c=3.5):
        self.state = state
        self.parent = parent if parent else DummyNode()
        self.c = c
        self.env = env_class
        self.n_actions = env_class.n_actions
        self.player_to_move = player_to_move
        self.last_action = action_just_executed

        self.valid_actions = self.env.get_valid_actions(state, self.player_to_move)
        self.children = {}
        self.child_n = np.zeros(self.n_actions, dtype=np.float32)  # 访问次数
        self.child_w = np.zeros(self.n_actions, dtype=np.float32)  # 累计价值
        self.child_p = np.zeros(self.n_actions, dtype=np.float32)  # 先验概率
        self.is_expanded = False

    def __repr__(self) -> str:
        return f'NeuronNode(last_action={self.last_action},N={self.n},W={self.w:.2f})'

    @property
    def n(self) -> float:
        return self.parent.child_n[self.last_action]

    @n.setter
    def n(self, value: float) -> None:
        self.parent.child_n[self.last_action] = value

    @property
    def w(self) -> float:
        return self.parent.child_w[self.last_action]

    @w.setter
    def w(self, value: float) -> None:
        self.parent.child_w[self.last_action] = value

    @property
    def child_q(self) -> NDArray[np.float32]:
        return self.child_w / (1 + self.child_n)  # 避免除0

    @property
    def child_u(self) -> NDArray[np.float32]:
        return self.c * self.child_p * np.sqrt(self.n) / (1 + self.child_n)

    @property
    def child_scores(self) -> NDArray[np.float32]:
        return self.child_q + self.child_u

    def get_child(self, action: int) -> Self:
        """在当前动作下执行action后产生一个子Node返回"""
        if action in self.valid_actions:
            if action not in self.children:  # 扩展真正节点
                new_state = self.env.virtual_step(self.state, action)
                self.children[action] = NeuronNode(
                    state=new_state,
                    action_just_executed=action,
                    player_to_move=1 - self.player_to_move,
                    env_class=self.env,
                    parent=self
                )
            return self.children[action]
        raise ValueError('Invalid last_action')

    def select(self) -> Self:
        """计算P UCT得分最高的孩子节点，选取该孩子返回，没有创建时先创建"""
        index = np.argmax(self.child_scores[self.valid_actions])
        action = self.valid_actions[index]
        return self.get_child(int(action))

    def evaluate(self, infer: InferenceEngine, is_self_play=False) -> float:
        # 对手赢奖励1，因为是从根节点的孩子节点看过来的，从根节点的话应该反过来
        winner = self.env.check_winner(self.state, self.player_to_move, self.last_action)
        if winner == 1 - self.player_to_move:
            return 1.0
        elif winner == self.player_to_move:
            return -1.0
        elif winner == -1:
            return 0.0
        state = self.env.convert_to_network(self.state, self.player_to_move)
        policy, value = infer.request(state, is_self_play=is_self_play)
        self.child_p = policy
        # 只在根节点添加噪声
        if is_self_play and isinstance(self.parent, DummyNode):
            self.inject_noise()
        self.is_expanded = True
        return value

    def back_propagate(self, result: float) -> None:
        """反向传播评估结果"""
        node = self
        while not isinstance(node, DummyNode):
            node.n += 1
            node.w += result
            result = -result
            node = node.parent

    def inject_noise(self, alpha=0.3, noise_weight=0.25) -> None:
        """根节点添加狄利克雷噪声，增加随机性"""
        legal_len = len(self.valid_actions)
        if legal_len == 0:
            return

        noise = np.random.dirichlet([alpha] * legal_len)
        self.child_p[self.valid_actions] = noise * noise_weight + (1 - noise_weight) * self.child_p[
            self.valid_actions]


class NeuronMCTS:
    def __init__(self, state: NDArray, env_class: type[BaseEnv], infer: InferenceEngine, last_action=-1,
                 player_to_move: int = 0,
                 is_self_play=False):
        self.root = NeuronNode(
            state=state,
            action_just_executed=last_action,
            player_to_move=player_to_move,
            env_class=env_class,
            parent=None,
        )
        self.infer = infer
        self.is_self_play = is_self_play

    def choose_action(self) -> int:
        """选择访问量最大的孩子作为根节点，并裁剪树"""
        action = int(np.argmax(self.root.child_n))
        self.root = self.root.get_child(action)
        self.root.parent = DummyNode()
        return action

    def apply_action(self, last_action: int) -> None:
        """根据对手动作，向下推进树，裁剪掉多余的"""
        node = self.root.get_child(last_action)

        self.root = node
        self.root.parent = DummyNode()

    def run(self, n_simulation=1000) -> None:
        """进行MCTS模拟，模拟完成后可根据孩子节点状态选择优势动作"""
        for _ in range(n_simulation):
            node = self.root
            # selection & Expansion
            while node.is_expanded:
                node = node.select()

            # Evaluation
            value = node.evaluate(self.infer, is_self_play=self.is_self_play)

            # Back Propagation
            node.back_propagate(value)

    def get_pi(self, temperature=1.0) -> NDArray[np.float32]:
        """将不同孩子节点的访问次数转换为概率分布
        :param temperature: 温度控制概率分布的集中程度，越小越集中，当为0集中到一点：时最大访问次数孩子的概率为1，其他都为0；为1时等同访问次数分布"""
        child_n = self.root.child_n[self.root.valid_actions]
        # 计算已扩展子节点的概率
        if temperature == 0:
            # 完全贪婪,访问次数最多的概率为1，其余为0
            pi = np.zeros_like(child_n)
            pi[np.argmax(child_n)] = 1
        else:
            child_n = child_n ** (1 / temperature)
            pi = child_n / np.sum(child_n)

        # 返回所有动作的概率，包括不合法的(概率0）
        pi_full = np.zeros_like(self.root.child_n, dtype=np.float32)
        pi_full[self.root.valid_actions] = pi
        return pi_full
