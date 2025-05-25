import collections
import time
import uuid

import numpy as np
from functions import apply_move, is_win


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
    def __init__(self, state, last_move, parent, c_puct=1.25, inference_queue=None, result_dict=None):
        self.state = state
        self.action = last_move[0] * state.shape[1] + last_move[1] if last_move else 0
        self.mark = 1 - last_move[2] if last_move else 0
        self.parent = parent if parent else DummyNode()
        self.children = {}
        self.child_N = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 访问次数
        self.child_W = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 累计价值
        self.child_P = np.zeros(state.shape[0] * state.shape[1], dtype=np.float32)  # 先验概率
        self.is_expanded = False
        self.is_evaluated = False
        self.c_puct = c_puct
        self.valid_indices = np.flatnonzero((state[:, :, 0] + state[:, :, 1]) == 0)
        self.is_leaf = is_win(state, last_move) or len(self.valid_indices) == 0

    @property
    def N(self):
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        return self.W / (1 + self.N)  # 避免除0

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)  # 避免除0

    @property
    def U(self):
        return self.parent.child_U[self.action]

    @property
    def child_U(self):
        return self.c_puct * self.child_P * np.sqrt(self.N) / (1 + self.child_N)

    def get_child(self, action):
        if action in self.valid_indices:
            if action not in self.children:  # 扩展真正节点
                move = divmod(action, self.state.shape[1]) + (self.mark,)
                state = apply_move(self.state, move)
                self.children[action] = NeuronNode(state, move, self)
            return self.children[action]

        raise IndexError('Action not available')

    def select(self):
        index = np.argmax(self.child_Q[self.valid_indices] + self.child_U[self.valid_indices])
        action = self.valid_indices[index]
        return self.get_child(action)

    def expand(self):
        # 采样合法动作，归一化
        probs = self.child_P[self.valid_indices]
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        # 选择最优孩子
        index = np.random.choice(len(self.valid_indices), p=probs)
        action = self.valid_indices[index]

        # 标记已扩展
        self.is_expanded = True
        return self.get_child(action)

    def evaluate_with_network(self, inference_engine):
        # 将state转为tensor，h,w,c->1,c,h,w
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # state_tensor = torch.tensor(np.transpose(self.state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
        # with (torch.no_grad()):
        #     # policy_logits [B,255],value[B],B=1
        #     policy_logits, value = network(state_tensor)
        #     policy_probs = F.softmax(policy_logits, dim=0)
        #     self.child_P = policy_probs.cpu().numpy()
        #     self.is_evaluated = True
        #     return value.item()
        policy, value = inference_engine.request(self.state)
        if policy.shape != (81,):
            print('state.shape:', self.state.shape)
            print('policy.shape:', policy.shape)

        self.child_P = policy
        self.is_evaluated = True
        return value

    def back_propagate(self, result):
        node = self
        while type(node) is not DummyNode:
            node.N += 1
            node.W += result
            result = -result
            node = node.parent


class DeepMCTS:
    def __init__(self, root_state: np.ndarray, inference_engine=None):
        self.root = NeuronNode(root_state, None, None)
        self.infer = inference_engine

    def choose_move(self):
        action = np.argmax(self.root.child_N)
        self.root = self.root.get_child(action)
        self.root.parent = DummyNode()
        move = divmod(action, self.root.state.shape[1]) + (self.root.mark,)
        return move

    def apply_move(self, move):
        action = move[0] * self.root.state.shape[1] + move[1]
        self.root = self.root.get_child(action)
        self.root.parent = DummyNode()

    def run(self, iteration=1000):
        for _ in range(iteration):
            node = self.root
            # selection
            while node.is_expanded:
                node = node.select()

            # Expansion
            if node.is_evaluated and not node.is_leaf:
                node = node.expand()

            # Evaluation
            value = node.evaluate_with_network(self.infer)

            # Back Propagation
            node.back_propagate(value)

    def get_pi(self, temperature=1):
        child_N = self.root.child_N[self.root.valid_indices]
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
        pi_full[self.root.valid_indices] = pi
        return pi_full
