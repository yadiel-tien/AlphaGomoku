import random
import math

from typing import List, Optional, Tuple

import numpy as np

from functions import available_moves, is_win, apply_move


class Node:
    def __init__(self, state: np.ndarray, last_move: Optional[Tuple[int, int, int]], parent: Optional['Node'] = None):
        self.state = state
        self.last_move = last_move
        self.mark = 1 - last_move[2] if last_move else 0
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.score = 0
        self.valid_moves = available_moves(state)
        self._uct = None
        self.uct_dirty = True
        self.is_leaf = self._check_leaf()

    def _check_leaf(self):
        """已经胜利、棋盘下满则为叶子节点"""
        if self.last_move is None:
            return False
        return is_win(self.state, self.last_move) or not self.valid_moves

    @property
    def is_fully_expanded(self):
        """判断子节点是否全部展开"""
        return len(self.valid_moves) == len(self.children)

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
        if is_win(self.state, self.last_move):
            return 1
        state = self.state.copy()
        mark = self.mark
        steps_limit = 20
        steps = 0
        while True:
            valid_moves = available_moves(state)
            if not valid_moves or steps > steps_limit:
                return 0  # draw
            w, h = random.choice(valid_moves)
            move = w, h, mark
            state = apply_move(state, move)
            steps += 1
            if is_win(state, move):
                return -1 if mark == self.mark else 1
            mark = 1 - mark

    def select(self):
        return max(self.children, key=lambda c: c.uct_score)

    def expand(self):
        tried_moves = [child.last_move[:2] for child in self.children]
        untried_moves = [m for m in self.valid_moves if m not in tried_moves]
        # 从未扩展的动作里随机选择一个
        w, h = random.choice(untried_moves)
        move = w, h, self.mark
        new_state = apply_move(self.state, move)

        child = Node(new_state, move, self)
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

    def choose_move(self):
        self.root = max(self.root.children, key=lambda child: child.N)
        return self.root.last_move

    def apply_opponent_move(self, state, move):
        """将MCTS树推进到对手落子后的节点，若未找到则创建新节点。"""
        for child in self.root.children:
            if child.last_move == move:
                child.parent = None
                self.root = child
                self.root.uct_dirty = True
                return
        print('mcts miss')
        self.root = type(self.root)(state, move)

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
