import copy
import random
import math

from typing import List, Optional

import numpy as np


class Node:
    def __init__(self, env, parent: Optional['Node'] = None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.score = 0
        self.valid_actions = env.valid_actions
        self._uct = None
        self.uct_dirty = True
        self.is_leaf = self.env.terminated

    def __repr__(self):
        return f'Node(action={self.env.last_action}, visits={self.visits})'

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
        if self.env.winner == 1 - self.env.player_to_move:
            return 1
        elif self.env.winner == self.env.player_to_move:
            return -1
        elif self.env.winner == -1:
            return 0

        current_player = self.env.player_to_move
        env = self.env.copy()
        while True:
            action = random.choice(env.valid_actions)
            env.step(action)

            if env.winner == current_player:
                return 1
            elif env.winner == 1 - current_player:
                return -1
            elif env.winner == -1:
                return 0

    def select(self):
        return max(self.children, key=lambda c: c.uct_score)

    def expand(self):
        tried_moves = [child.env.last_action for child in self.children]
        untried_moves = [a for a in self.valid_actions if a not in tried_moves]
        # 从未扩展的动作里随机选择一个
        action = random.choice(untried_moves)
        new_env = copy.deepcopy(self.env)
        new_env.step(action)
        child = Node(new_env, self)
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
    def __init__(self, env):
        self.root = Node(env, None)

    def choose_action(self):
        self.root = max(self.root.children, key=lambda child: child.visits)
        return self.root.env.last_action

    def apply_opponent_action(self, state, action):
        """将MCTS树推进到对手落子后的节点，若未找到则创建新节点。"""
        for child in self.root.children:
            if child.env.last_action == action:
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
