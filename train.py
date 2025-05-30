import copy
import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch import optim
from board import GomokuEnv
from config import MODEL_PATH, DEVICE
from deepMcts import NeuronMCTS
from inference import InferenceEngine
from network import Net
from player import AIServer
from replay import ReplayBuffer
import torch.nn.functional as F


class Trainer:
    def __init__(self, rows, columns, n_workers, best_model_index):
        self.rows, self.columns = rows, columns
        self.model = Net(256, rows * columns).to(DEVICE)
        self.buffer = ReplayBuffer(100_000, 128)
        self.infer = InferenceEngine(self.model)
        # weight_decay为l2正则
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.pool = ThreadPoolExecutor(n_workers)
        self.best_model_index = best_model_index

    def load(self):
        self.buffer.load()
        path = f'./data/model_{self.best_model_index}.pt'
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))

    def save(self, index=0):
        self.buffer.save()
        torch.save(self.model.state_dict(), f'./data/model_{index}.pt')

    def make_env(self):
        return GomokuEnv(self.rows, self.columns)

    def self_play1game(self, n_simulation, temperature):
        env = self.make_env()
        states, pis, players = [], [], []
        state, _ = env.reset()
        mcts = NeuronMCTS(state, self.infer, is_self_play=True)
        done = False
        reward = 0

        while not done:
            mcts.run(n_simulation)  # 模拟
            pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi
            # 采集数据
            states.append(np.copy(state))
            pis.append(np.copy(pi))
            players.append(env.current_player)
            # 根据pi来选择动作
            action = np.random.choice(len(pi), p=pi)
            state, reward, done, _, _ = env.step(action)  # 执行落子
            mcts.apply_action(state, action)  # mcts也要根据action进行对应裁剪

        # 一局结束，获取winner ID（0，1）,平局-1
        winner = 1 - env.current_player if reward else -1

        experiences = []
        # 根据winner和player的到针对player的比赛结果z，收集数据
        for state, pi, player in zip(states, pis, players):
            z = 1 if winner == player else 0 if winner == -1 else -1
            experiences.append((state, pi, z))

        return experiences

    def self_play(self, n_games, n_simulation=100, temperature=1):
        """自我对弈，收集每步的state，pi，z"""
        start = time.time()
        dataset = []
        futures = [self.pool.submit(self.self_play1game, n_simulation, temperature) for _ in range(n_games)]
        for f in as_completed(futures):
            dataset.extend(f.result())
        for d in dataset:
            self.buffer.add(*d)
        duration = time.time() - start
        print(
            f'采集到{len(dataset)}原始数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / len(dataset) :.4f}。')

    def fit(self, epochs=100):
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        for epoch in range(epochs):
            # 批量数据获取，转tensor
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).to(DEVICE)  # 【B，2,H,W]
            pis = torch.from_numpy(pis).to(DEVICE)  # [B,H*W]
            zs = torch.from_numpy(zs).to(DEVICE)  # [B]

            # 模型前向推理
            policy_logits, values = self.model(states)
            # print(f"value.mean={values.mean().item():.4f}, std={values.std().item():.4f}")

            # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
            policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
            # 均方差损失，使value的结果趋近与mcts模拟出来的z，[B]。(values-z)**2最坏的情况为4，前面乘0.25是为了归一化，
            value_loss = 0.25 * F.mse_loss(values, zs)

            # 用总的损失进行反向梯度更新
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()  # 清空旧梯度
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 将梯度范数裁剪到1.0
            self.optimizer.step()  # 更新参数

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
                )

        duration = time.time() - start
        print(f"{epochs}轮训练完成，共用时{duration:.2f}秒。")

    def train(self, n_games=100, epochs=100):
        # 根据保存的路径获取开始轮次
        iteration_start = self.best_model_index

        for i in range(epochs):
            iteration = iteration_start + i + 1
            print(f'Current Iteration:{i + 1}/{epochs},Total Iteration:{iteration}')

            # 多线程并行采集对战数据
            tao = 1 - i / epochs
            if tao < 0.1:
                tao = 0
            self.self_play(n_games, tao)

            # 进行训练
            self.fit(epochs=100)

            # 评估
            if i % 10 == 0 and iteration != self.best_model_index:
                self.eval(iteration)

        print(f'当前最好模型为{self.best_model_index}')

    def eval(self, index):
        start = time.time()
        print(f'目前最佳model:{self.best_model_index},待评估model：{index}')
        to_be_eval = InferenceEngine(self.model)
        futures = []
        for _ in range(20):
            env = self.make_env()
            players = [AIServer(to_be_eval), AIServer(self.infer)]
            futures.append(self.pool.submit(env.random_order_play, players, silent=True))
        result = []
        for future in as_completed(futures):
            result.append(future.result())
        win_rate = result.count((1, 0)) / len(result)
        draw_rate = result.count((0, 0)) / len(result)
        print(f"win_rate:{win_rate:.2%},draw_rate:{draw_rate:.2%}")
        duration = time.time() - start
        if win_rate > 0.55:
            self.best_model_index = index
            # 保存模型
            self.save(index)
            # 将训练好的参数同步给推理模型
            self.infer.update_model(self.model)
            print(f'最佳玩家更新为{index},评估用时:{duration:.2f}秒')
        else:
            print(f'最佳玩家未更新,仍旧为{self.best_model_index},评估用时:{duration:.2f}')

    def __del__(self):
        self.pool.shutdown()


if __name__ == '__main__':
    # 总的轮次
    trainer = Trainer(9, 9, n_workers=12, best_model_index=311)
    trainer.load()
    trainer.train(n_games=24, epochs=800)
    # trainer.eval(737)
