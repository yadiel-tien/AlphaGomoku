import os.path
import pickle
from collections import deque
import random

import numpy as np
from numpy.ma.core import indices

from config import SETTINGS
from functions import apply_symmetry


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, mcts_prob, winner):
        """添加数据并进行数据增强"""
        if state.shape == (10, 9, 20):  # 象棋不扩充
            self.buffer.append((state, mcts_prob, winner))
        else:
            augmented_data = self.augment_data(state, mcts_prob, winner)
            self.buffer.extend(augmented_data)

    def augment_data(self, state, mcts_prob, winner):
        """通过旋转和翻转棋盘进行数据增强"""
        augmented_samples = []
        shape = state.shape[:2]
        for i in range(8):
            transformed_state = apply_symmetry(state, i, shape)
            transformed_prob = apply_symmetry(mcts_prob, i, shape)
            augmented_samples.append((transformed_state, transformed_prob, winner))

        return augmented_samples

    def get_batch(self, min_win_ratio=0):
        """随机采样一个 batch 进行训练"""
        batch = random.sample(self.buffer, self.batch_size)
        states, probs, winners = zip(*batch)  # [B,H,W,C],[B,H*W],[B],第一个维度是tuple，需转为ndarray
        states = np.transpose(np.array(states), (0, 3, 1, 2))  # [B,H,W,C]->[B,C,H,W]
        return states, np.array(probs), np.array(winners, dtype=np.float32)

    def save(self):
        path = SETTINGS['buffer_path']
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self):
        path = SETTINGS['buffer_path']
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.buffer = pickle.load(f)

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        string = ''
        for state, _, _ in self.buffer:
            string += "&" * 50 + '\n'
            for row in state:
                string += '  '.join(
                    ['X' if cell[0] else 'O' if cell[1] else '.' for cell in row]) + '\n'
        return string
