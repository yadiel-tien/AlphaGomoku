import os.path
import pickle
from collections import deque
import random

import numpy as np

from config import BUFFER_PATH


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, mcts_prob, winner):
        """添加数据并进行数据增强"""
        augmented_data = self.augment_data(state, mcts_prob, winner)
        self.buffer.extend(augmented_data)

    def augment_data(self, state, mcts_prob, winner):
        """通过旋转和翻转棋盘进行数据增强"""
        augmented_samples = []
        for i in range(4):  # 旋转 0°, 90°, 180°, 270°
            rotated_state = np.rot90(state, k=i, axes=(0, 1))
            rotated_prob = np.rot90(mcts_prob.reshape(state.shape[:2]), k=i, axes=(0, 1)).flatten()
            augmented_samples.append((rotated_state, rotated_prob, winner))

            for j in range(2):
                flipped_state = np.flip(rotated_state, axis=j)  # 水平翻转
                flipped_prob = np.flip(rotated_prob.reshape(state.shape[:2]), axis=j).flatten()
                augmented_samples.append((flipped_state, flipped_prob, winner))

        return augmented_samples

    def get_batch(self):
        """随机采样一个 batch 进行训练"""
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        states, probs, winners = zip(*batch)  # [B,H,W,2],[B,H*W],[B],第一个维度是tuple，需转为ndarray
        states = np.transpose(np.array(states), (0, 3, 1, 2))  # [B,H,W,2]->[B,2,H,W]
        return states, np.array(probs), np.array(winners)

    def save(self, path=BUFFER_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path=BUFFER_PATH):
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
