import copy
import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch import optim
from board import GomokuEnv
from config import MODEL_PATH, DEVICE
from deepMcts import DeepMCTS
from inference import InferenceEngine
from network import Net
from player import RandomPlayer, AI
from replay import ReplayBuffer
import torch.nn.functional as F


class Trainer:
    def __init__(self, rows, columns, n_workers, best_model_index):
        self.rows, self.columns = rows, columns
        self.model = Net(256, rows * columns).to(DEVICE)
        self.buffer = ReplayBuffer(30000, 128)
        self.infer = InferenceEngine(self.model)
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

    def self_play(self, n_games=1, n_simulation=100, temperature=1):
        """自我对弈，收集每步的state，pi，z"""
        dataset = []
        env = self.make_env()
        for i in range(n_games):
            states = []
            pis = []
            players = []
            state, _ = env.reset()
            mcts = DeepMCTS(state, self.infer, is_self_play=True)
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
                mcts.apply_action(action)  # mcts也要根据action进行对应裁剪

            # 一局结束，获取winner ID（0，1）,平局-1
            if reward:
                winner = 1 - env.current_player
            else:
                winner = -1

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
            self.optimizer.step()  # 更新参数

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
                )

    def train(self, epochs=100, total_games=100):
        # 根据保存的路径获取开始轮次
        iteration_start = self.best_model_index

        for i in range(epochs):
            print(f'Current Iteration:{i + 1},Total Iteration:{iteration_start + i + 1}')
            experiences = []

            # 多线程并行采集对战数据
            start = time.time()
            futures = [self.pool.submit(self.self_play, n_simulation=500) for _ in range(total_games)]
            for f in as_completed(futures):
                experiences.extend(f.result())

            duration = time.time() - start
            print(
                f'采集到{len(experiences)}原始数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / len(experiences) :.4f}。')

            for experience in experiences:
                self.buffer.add(*experience)

            # 进行训练，使神经网络学习mcts经验,weight_decay为l2正则
            start = time.time()
            self.fit(epochs=100)
            duration = time.time() - start
            print(f"训练完成，用时{duration:.2f}秒。")

            # 将训练好的参数同步给推理模型
            self.infer.update_model(self.model)

            # 保存数据，便于以后使用
            self.save(iteration_start + i + 1)

    def eval(self, index):
        futures = []
        for _ in range(10):
            env = self.make_env()
            players = [AI(model_id=index, silent=True), AI(model_id=self.best_model_index, silent=True)]
            futures.append(self.pool.submit(env.evaluate, players, 2))
        result = []
        for future in as_completed(futures):
            result.extend(future.result())
        win_rate = result.count((1, 0)) / len(result)
        print(f"win_rate:{win_rate:.2f}")
        if win_rate > 0.55:
            self.best_model_index = index
            print(f'最佳玩家更新为{index}')
        else:
            print(f'最佳玩家为更新,仍旧为{self.best_model_index}')

    def shutdown(self):
        self.pool.shutdown()
        self.infer.shutdown()


if __name__ == '__main__':
    # 总的轮次
    trainer = Trainer(9, 9, n_workers=12, best_model_index=0)
    # for epoch in range(20):
    #     trainer.load()
    #     trainer.train(total_games=24)
    #     all_file = os.listdir('./data/')
    #     index = max(int(f.split('_')[-1].split('.')[0]) for f in all_file)
    #     trainer.eval(index)
    # trainer.shutdown()
    trainer.load()
    trainer.train(total_games=24)
    trainer.shutdown()
