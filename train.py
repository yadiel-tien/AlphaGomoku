import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch import optim
from board import GomokuEnv
from config import MODEL_PATH, DEVICE
from deepMcts import NeuronMCTS
from inference import InferenceEngine, make_engine
from network import Net
from player import AIServer
from replay import ReplayBuffer
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)


class Trainer:
    def __init__(self, rows, columns, n_workers, best_model_index):
        self.rows, self.columns = rows, columns
        self.model = Net(256, rows * columns).to(DEVICE)
        self.buffer = ReplayBuffer(200_000, 128)
        self.latest_infer = InferenceEngine(self.model)
        self.best_infer = make_engine(best_model_index)
        # weight_decay为l2正则
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.pool = ThreadPoolExecutor(n_workers, thread_name_prefix='trainer-')
        self.best_model_index = best_model_index
        self.logger = logging.getLogger('trainer')
        self.eval_logger = logging.getLogger('eval')
        self.play_logger = logging.getLogger('selfplay')
        self.fit_logger = logging.getLogger('fit')

    def load(self):
        self.buffer.load()
        path = f'./data/model_{self.best_model_index}.pt'
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))

    def save(self, index=0):
        self.buffer.save()
        torch.save(self.model.state_dict(), f'./data/model_{index}.pt')

    def self_play1game(self, n_simulation):
        env = GomokuEnv(self.rows, self.columns)
        states, pis, players = [], [], []
        state, _ = env.reset()
        mcts = NeuronMCTS(state, self.best_infer, is_self_play=True)
        done = False
        reward = 0
        step = 0

        while not done:
            temperature = 0 if step >= 5 else 1  # 前几步鼓励探索
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
            step += 1
        # 一局结束，获取winner ID（0，1）,平局-1
        winner = 1 - env.current_player if reward else -1

        experiences = []
        # 根据winner和player的到针对player的比赛结果z，收集数据
        for state, pi, player in zip(states, pis, players):
            z = 1 if winner == player else -0.1 if winner == -1 else -1
            experiences.append((state, pi, z))

        return experiences

    def self_play(self, n_games, n_simulation):
        """自我对弈，收集每步的state，pi，z"""
        start = time.time()
        dataset = []
        futures = [self.pool.submit(self.self_play1game, n_simulation) for _ in range(n_games)]
        z_sample = []
        for f in as_completed(futures):
            z_sample.append(f.result()[0][2])
            dataset.extend(f.result())
        win_count = z_sample.count(1) + z_sample.count(-1)
        draw_count = z_sample.count(-0.1)
        self.play_logger.info(
            f'self playing {n_games} games,win:{win_count / n_games:.2%},draw:{draw_count / n_games:.2%}')
        for d in dataset:
            self.buffer.add(*d)
        duration = time.time() - start
        self.play_logger.info(
            f'采集到{len(dataset)}原始数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / len(dataset) :.4f}。'
        )

    def fit(self, epochs=100):
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        for epoch in range(epochs):
            # 批量数据获取，转tensor
            states, pis, zs = self.buffer.get_batch(min_win_ratio=0.5)
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
                self.fit_logger.info(
                    f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
                )

        duration = time.time() - start
        self.fit_logger.info(f"{epochs}轮训练完成，共用时{duration:.2f}秒。")

    def train(self, n_games=100, n_simulation=360, epochs=100):
        # 根据保存的路径获取开始轮次
        iteration_start = self.best_model_index
        for i in range(epochs):
            iteration = iteration_start + i + 1
            self.logger.info(
                f'Current Iteration:{i + 1}/{epochs},Total Iteration:{iteration}，Best Model Index:{self.best_model_index}。')

            # 多线程并行采集对战数据
            self.self_play(n_games, n_simulation)

            # 进行训练
            self.fit(epochs=100)

            # 评估
            if (i + 1) % 5 == 0 and iteration != self.best_model_index:
                self.eval(iteration)
                # 保存模型
                self.save(iteration)
        self.logger.info(f'当前最好模型为{self.best_model_index}')

    def eval(self, index):
        start = time.time()
        self.eval_logger.info(f'目前最佳model:{self.best_model_index},待评估model：{index}')
        if self.latest_infer is None:
            self.latest_infer = InferenceEngine(self.model)
        else:
            self.latest_infer.update_model(self.model)
            self.latest_infer.start()

        futures = []
        for _ in range(20):
            env = GomokuEnv(self.rows, self.columns)
            players = [AIServer(self.latest_infer), AIServer(self.best_infer)]
            futures.append(self.pool.submit(env.random_order_play, players, silent=True))
        result = []
        for future in as_completed(futures):
            result.append(future.result())
        win_rate = result.count((1, 0)) / len(result)
        draw_rate = result.count((0, 0)) / len(result)
        self.eval_logger.info(f"win_rate:{win_rate:.2%},draw_rate:{draw_rate:.2%}")
        duration = time.time() - start
        if win_rate + draw_rate / 2 > 0.55:
            self.best_model_index = index
            self.best_infer.update_model(self.model)
            self.eval_logger.info(f'最佳玩家更新为{index},评估用时:{duration:.2f}秒')
        elif win_rate + draw_rate / 2 < 0.4:
            self.model.load_state_dict(torch.load(f'data/model_{self.best_model_index}.pt'))
            self.eval_logger.info(f'当前model回退至{self.best_model_index},评估用时:{duration:.2f}秒')
        else:
            self.eval_logger.info(f'最佳玩家未更新,仍旧为{self.best_model_index},评估用时:{duration:.2f}秒')
        self.latest_infer.shutdown()

    def shutdown(self):
        self.pool.shutdown()
        self.best_infer.shutdown()
        self.latest_infer.shutdown()


if __name__ == '__main__':
    # 总的轮次
    trainer = Trainer(9, 9, n_workers=12, best_model_index=979)
    trainer.load()
    trainer.train(n_games=24, epochs=2000)
    # trainer.eval(313)
    trainer.shutdown()
