import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch import optim
from board import GomokuEnv
from config import MODEL_PATH
from deepMcts import DeepMCTS
from inference import InferenceEngine
from network import Net
from replay import ReplayBuffer
import torch.nn.functional as F


class Trainer:
    def __init__(self, rows, columns, n_workers):
        self.rows, self.columns = rows, columns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(256, rows * columns).to(self.device)
        self.buffer = ReplayBuffer(30000, 128)
        self.infer = InferenceEngine(self.model, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.pool = ThreadPoolExecutor(n_workers)

    def load(self):
        self.buffer.load()
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH))

    def save(self, index=0):
        self.buffer.save()
        torch.save(self.model.state_dict(), f'./data/model_{index}.pt')

    def make_env(self):
        return GomokuEnv(self.rows, self.columns)

    def self_play(self, n_games=1, n_simulation=100, temperature=1):
        """自我对弈，收集每步的state，pi，z"""
        dataset = []
        env = self.make_env()
        for i in range(n_games):
            states = []
            pis = []
            players = []
            current_player = 0
            state, _ = env.reset()
            mcts = DeepMCTS(state, self.infer)
            done = False
            reward = 0

            while not done:
                mcts.run(n_simulation)  # 模拟
                pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi
                # 采集数据
                states.append(np.copy(state))
                pis.append(np.copy(pi))
                players.append(current_player)
                # 根据pi来选择动作
                action = np.random.choice(len(pi), p=pi)
                state, reward, done, _, _ = env.step(action)  # 执行落子
                mcts.apply_move(env.action2index(action))  # mcts也要根据action进行对应裁剪
                current_player = 1 - current_player  # 切换玩家，0或1
            # 一局结束，获取winner ID（0，1）,平局-1
            winner = env.get_winner(reward, 1 - current_player)

            # 根据winner和player的到针对player的比赛结果z，收集数据
            for state, pi, player in zip(states, pis, players):
                if winner == player:
                    z = 1
                elif winner == -1:
                    z = 0
                else:
                    z = -1
                dataset.append((state, pi, z))
        return dataset

    def fit(self, epochs=100):
        """从buffer中获取数据，训练神经网络"""
        for epoch in range(epochs):
            # 批量数据获取，转tensor
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).to(self.device)  # 【B，2,H,W]
            pis = torch.from_numpy(pis).to(self.device)  # [B,H*W]
            zs = torch.from_numpy(zs).to(self.device)  # [B]

            # 模型前向推理
            policy_logits, values = self.model(states)

            # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
            policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
            # 均方差损失，使value的结果趋近与mcts模拟出来的z，[B]。(values-z)**2最坏的情况为4，前面乘0.25是为了归一化，
            value_loss = 0.25 * F.mse_loss(values, zs)

            # 用总的损失进行反向梯度更新
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()  # 清空旧梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
                )

    def train(self, epochs=30, total_games=100, n_workers=32):
        # 根据保存的路径获取开始轮次
        iteration_start = int(MODEL_PATH.split('_')[-1].split('.')[0]) if os.path.exists(MODEL_PATH) else 0

        for i in range(epochs):
            print(f'Current Iteration:{i + 1},Total Iteration:{iteration_start + i + 1}')
            experiences = []

            # 多线程并行采集对战数据
            start = time.time()
            futures = [self.pool.submit(self.self_play, n_simulation=200) for _ in range(total_games)]
            for f in as_completed(futures):
                experiences.extend(f.result())

            duration = time.time() - start
            print(
                f'采集到{len(experiences)}数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / len(experiences) :.6f}。')

            for experience in experiences:
                self.buffer.add(*experience)

            # 进行训练，使神经网络学习mcts经验,weight_decay为l2正则
            start = time.time()
            self.fit(epochs=100)
            duration = time.time() - start
            print(f"训练完成，用时{duration:.2f}秒。")

            # 保存数据，便于以后使用
            self.save(iteration_start + i + 1)

    def shutdown(self):
        self.pool.shutdown()
        self.infer.shutdown()


if __name__ == '__main__':
    # 总的轮次
    trainer = Trainer(9, 9, n_workers=12)
    trainer.train(total_games=24)
    trainer.shutdown()
