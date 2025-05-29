import collections

import numpy as np
from functions import apply_action, is_win


class DummyNode(object):
    """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class NeuronNode:
    def __init__(self, state, last_action=None, parent=None, c_puct=3):
        self.state = state
        self.last_action = last_action
        self.parent = parent if parent else DummyNode()
        self.children = {}
        self.child_N = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 访问次数
        self.child_W = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 累计价值
        self.child_P = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 先验概率
        self.is_expanded = False
        self.c_puct = c_puct
        self.valid_actions = np.flatnonzero((state[:, :, 0] + state[:, :, 1]) == 0)

    def __repr__(self):
        return f'NeuronNode(action={self.last_action},N={self.N},W={self.W:.2f})'

    @property
    def N(self):
        return self.parent.child_N[self.last_action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.last_action] = value

    @property
    def W(self):
        return self.parent.child_W[self.last_action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.last_action] = value

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
                new_state = apply_action(self.state, action)
                self.children[action] = NeuronNode(
                    new_state,
                    last_action=action,
                    parent=self
                )
            return self.children[action]

        return None

    def select(self):
        index = np.argmax(self.child_scores[self.valid_actions])
        action = self.valid_actions[index]
        return self.get_child(action)

    def evaluate(self, inference_engine, add_noise):
        if is_win(self.state, self.last_action):
            return 1.0
        elif len(self.valid_actions) == 0:
            return 0.0

        policy, value = inference_engine.request(self.state)
        self.child_P = policy
        # 只在根节点添加噪声
        if add_noise and isinstance(self.parent, DummyNode):
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
    def __init__(self, root_state, inference_engine, last_action=None, is_self_play=False):
        self.root = NeuronNode(root_state, last_action)
        self.infer = inference_engine
        self.is_self_play = is_self_play

    def choose_action(self):
        action = np.argmax(self.root.child_N)
        self.root = self.root.get_child(action)
        self.root.parent = DummyNode()
        return action

    def apply_action(self, state, last_action):
        node = self.root.get_child(last_action)
        if node is None or not np.array_equal(node.state, state):
            print('new node')
            self.root = NeuronNode(state, last_action)
        else:
            print('found node')
            self.root = node
            self.root.parent = DummyNode()

    def run(self, iteration=1000):
        for _ in range(iteration):
            node = self.root
            # selection & Expansion
            while node.is_expanded:
                node = node.select()

            # Evaluation
            value = node.evaluate(self.infer, add_noise=self.is_self_play)

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
