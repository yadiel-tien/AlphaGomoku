import collections
import copy
from typing import Self

import numpy as np

from inference import request_mp_infer


class DummyNode(object):
    """作为根节点的父节点使用，简化逻辑"""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class NeuronNode:
    def __init__(self, env, parent: Self = None, c_puct=3.5):
        self.env = copy.deepcopy(env)
        self.parent = parent if parent else DummyNode()
        self.children = {}
        self.child_N = np.zeros(env.action_space.n, dtype=np.float32)  # 访问次数
        self.child_W = np.zeros(env.action_space.n, dtype=np.float32)  # 累计价值
        self.child_P = np.zeros(env.action_space.n, dtype=np.float32)  # 先验概率
        self.is_expanded = False
        self.c_puct = c_puct
        self.valid_actions = env.valid_actions()

    def __repr__(self):
        return f'NeuronNode(action={self.env.last_action},N={self.N},W={self.W:.2f})'

    @property
    def N(self):
        return self.parent.child_N[self.env.last_action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.env.last_action] = value

    @property
    def W(self):
        return self.parent.child_W[self.env.last_action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.env.last_action] = value

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)  # 避免除0

    @property
    def child_U(self):
        return self.c_puct * self.child_P * np.sqrt(self.N) / (1 + self.child_N)

    @property
    def child_scores(self):
        return self.child_Q + self.child_U

    def get_child(self, action):
        if action in self.valid_actions:
            if action not in self.children:  # 扩展真正节点
                new_env = copy.deepcopy(self.env)
                new_env.step(action)
                self.children[action] = NeuronNode(
                    env=new_env,
                    parent=self
                )
            return self.children[action]
        raise ValueError('Invalid action')

    def select(self):
        index = np.argmax(self.child_scores[self.valid_actions])
        action = self.valid_actions[index]
        return self.get_child(action)

    def evaluate(self, req_q, is_self_play=False):
        if self.env.winner == 1 - self.env.current_player:
            return 1.0
        elif self.env.winner == self.env.current_player:
            return -1.0
        elif self.env.winner == -1:
            return 0.0

        policy, value = request_mp_infer(self.env.state, req_q)
        self.child_P = policy
        # 只在根节点添加噪声
        if is_self_play and isinstance(self.parent, DummyNode):
            self.inject_noise()
        self.is_expanded = True
        return value

    def back_propagate(self, result):
        node = self
        while not isinstance(node, DummyNode):
            node.N += 1
            node.W += result
            result = -result
            node = node.parent

    def inject_noise(self, alpha=0.3, noise_weight=0.25):
        legal_len = len(self.valid_actions)
        if legal_len == 0:
            return

        noise = np.random.dirichlet([alpha] * legal_len)
        self.child_P[self.valid_actions] = noise * noise_weight + (1 - noise_weight) * self.child_P[
            self.valid_actions]


class NeuronMCTS:
    def __init__(self, env, req_q, is_self_play=False):
        self.root = NeuronNode(env)
        self.req_q = req_q
        self.is_self_play = is_self_play

    def choose_action(self):
        action = np.argmax(self.root.child_N)
        self.root = self.root.get_child(action)
        self.root.parent = DummyNode()
        return action

    def apply_action(self, env):
        node = self.root.get_child(env.last_action)
        if node is None or not np.array_equal(node.env.state, env.state):
            self.root = NeuronNode(env)
        else:
            self.root = node
            self.root.parent = DummyNode()

    def run(self, n_simulation=1000):
        for _ in range(n_simulation):
            node = self.root
            # selection & Expansion
            while node.is_expanded:
                node = node.select()

            # Evaluation
            value = node.evaluate(self.req_q, is_self_play=self.is_self_play)

            # Back Propagation
            node.back_propagate(value)

    def get_pi(self, temperature=1.0):
        child_N = self.root.child_N[self.root.valid_actions]
        # 计算已扩展子节点的概率
        if temperature == 0:
            # 完全贪婪,访问次数最多的概率为1，其余为0
            pi = np.zeros_like(child_N)
            pi[np.argmax(child_N)] = 1
        else:
            child_N = child_N ** (1 / temperature)
            pi = child_N / np.sum(child_N)

        # 返回概率和对应动作
        pi_full = np.zeros_like(self.root.child_N, dtype=np.float32)
        pi_full[self.root.valid_actions] = pi
        return pi_full
