import random
import math

from typing import List, Optional

import numpy as np

from functions import is_win, apply_action


class Node:
    def __init__(self, state: np.ndarray, action: Optional[int], parent: Optional['Node'] = None):
        self.state = state
        self.last_action = action
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.score = 0
        self.valid_actions = np.flatnonzero((state[:, :, 0] + state[:, :, 1]) == 0)
        self._uct = None
        self.uct_dirty = True
        self.is_leaf = self._check_leaf()

    def __repr__(self):
        return f'Node(action={self.last_action}, visits={self.visits})'

    def _check_leaf(self):
        """已经胜利、棋盘下满则为叶子节点"""
        if self.last_action is None:
            return False
        return is_win(self.state, self.last_action) or len(self.valid_actions) == 0

    @property
    def is_fully_expanded(self):
        """判断子节点是否全部展开"""
        return len(self.valid_actions) == len(self.children)

    def _update_uct(self, c=1.414) -> None:
        if self.visits == 0:
            self._uct = float('Inf')
        elif not self.parent:
            # 没有父节点则不再计算探索项
            self._uct = self.score / self.visits
        else:
            exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
            self._uct = (self.score / self.visits) + exploration
        self.uct_dirty = False

    @property
    def uct_score(self):
        """使用uct_dirty标记是否需要更新，不需要则直接返回"""
        if self.uct_dirty:
            self._update_uct()
        return self._uct

    def rollout(self) -> int:
        # 对于任意node，对手已胜利就返回1
        if is_win(self.state, self.last_action):
            return 1
        state = self.state.copy()
        current_player = 0
        while True:
            valid_actions = np.flatnonzero((state[:, :, 0] + state[:, :, 1]) == 0)
            if len(valid_actions) == 0:
                return 0  # draw
            action = random.choice(valid_actions)
            state = apply_action(state, action)
            current_player = 1 - current_player

            # current_player==0代表self，1代表对手
            if is_win(state, action):
                return 1 if current_player == 0 else -1

    def select(self):
        return max(self.children, key=lambda c: c.uct_score)

    def expand(self):
        tried_moves = [child.last_action for child in self.children]
        untried_moves = [a for a in self.valid_actions if a not in tried_moves]
        # 从未扩展的动作里随机选择一个
        action = random.choice(untried_moves)
        new_state = apply_action(self.state, action)

        child = Node(new_state, action, self)
        self.children.append(child)
        return child

    def back_propagate(self, result):
        node = self
        while node:
            node.visits += 1
            node.score += result
            node.uct_dirty = True
            for child in node.children:
                child.uct_dirty = True
            result = -result
            node = node.parent


class MCTS:
    def __init__(self, root_state: np.ndarray):
        self.root = Node(root_state, None)

    def choose_action(self):
        self.root = max(self.root.children, key=lambda child: child.visits)
        return self.root.last_action

    def apply_opponent_action(self, state, action):
        """将MCTS树推进到对手落子后的节点，若未找到则创建新节点。"""
        for child in self.root.children:
            if child.last_action == action:
                child.parent = None
                self.root = child
                self.root.uct_dirty = True
                return
        print('mcts miss')
        self.root = Node(state, action)

    def run(self, iteration=1000):
        for _ in range(iteration):
            node = self.root
            # selection
            while node.is_fully_expanded and node.children:
                node = node.select()

            # Expansion
            if not node.is_leaf:
                node = node.expand()

            # Simulation
            result = node.rollout()

            # Back Propagation
            node.back_propagate(result)
