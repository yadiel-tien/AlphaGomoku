import glob
import os.path
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from board import GomokuEnv
from config import CONFIG
from deepMcts import NeuronMCTS
from inference import InferenceEngine, make_engine
from network import Net
from player import AIServer
from replay import ReplayBuffer
import torch.nn.functional as F
import logging
import multiprocessing as mp


def get_logger(name, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s')

        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


def self_play1game(infer, n_simulation):
    env = GomokuEnv(*CONFIG['board_shape'])
    states, pis, players = [], [], []
    state, _ = env.reset()
    mcts = NeuronMCTS(state, infer, is_self_play=True)
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
        z = 1 if winner == player else 0 if winner == -1 else -1
        experiences.append((state, pi, z))

    return experiences


def self_play(best_model_index, n_games, n_simulation):
    """自我对弈，收集每步的state，pi，z"""

    play_logger = get_logger('selfplay')
    process_name = mp.current_process().name
    play_logger.info(f'{process_name} self play begin')
    infer = make_engine(best_model_index)
    buffer = ReplayBuffer(200_000, 128)
    index = int(process_name.split('-')[-1].split('.')[0])
    file_name = f'Process-{index % 2 + 1}'
    buffer.load(name=file_name)

    start = time.time()
    with  ThreadPoolExecutor(8, thread_name_prefix='self_play-') as pool:
        futures = [pool.submit(self_play1game, infer, n_simulation) for _ in range(n_games)]
        z_sample = []
        game_count, data_count = 0, 0
        for f in as_completed(futures):
            game_count += 1
            experiences = f.result()
            z_sample.append(experiences[0][2])
            data_count += len(experiences)
            for experience in experiences:
                buffer.add(*experience)
            if game_count % 10 == 0:
                draw_count = z_sample.count(0)
                win_count = len(z_sample) - draw_count
                play_logger.info(f'{process_name}:self playing 10 games,win/loss:{win_count} ,draw:{draw_count}')
                duration = time.time() - start
                play_logger.info(
                    f'{process_name}:采集到{data_count}条原始数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / data_count :.4f}。'
                )
                data_count = 0
                z_sample = []
                start = time.time()

    buffer.save(name=file_name)
    infer.shutdown()
    play_logger.info(f'{process_name} self play end')


def read_best_index():
    if os.path.exists(CONFIG['best_index_path']):
        with open(CONFIG['best_index_path'], "rb") as f:
            return pickle.load(f)
    return None


def read_latest_index():
    model_files = glob.glob("./data/model_*.pt")
    if model_files:
        return max(
            int(f.split("_")[1].split(".")[0])
            for f in model_files
        )
    else:
        return None


def write_best_index(best_index):
    with open(CONFIG['best_index_path'], "wb") as f:
        pickle.dump(best_index, f)


class Trainer:
    def __init__(self):
        self.logger = get_logger('main')
        self.fit_logger = get_logger('fit')
        self.eval_logger = get_logger('eval')
        self.best_model_index = read_best_index()
        self.best_infer = make_engine(self.best_model_index)
        self.latest_model_index = read_latest_index()
        h, w = CONFIG['board_shape']
        self.model = Net(256, h * w).to(CONFIG['device'])
        if self.latest_model_index is not None:
            self.model.load_state_dict(
                torch.load(f'data/model_{self.latest_model_index}.pt', map_location=CONFIG['device'])
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.latest_infer = InferenceEngine(self.model)
        self.buffer = ReplayBuffer(400_000, 128)

    def merge_buffer(self):
        buffer1 = ReplayBuffer(200_000, 128)
        buffer2 = ReplayBuffer(200_000, 128)
        buffer1.load('Process-1')
        buffer2.load('Process-2')
        self.buffer.buffer.extend(buffer1.buffer)
        self.buffer.buffer.extend(buffer2.buffer)
        self.buffer.win_buffer.extend(buffer1.win_buffer)
        self.buffer.win_buffer.extend(buffer2.win_buffer)

    def run(self, start, iteration):
        mp.set_start_method('spawn')
        for i in range(start, start + iteration):
            self.logger.info(f'iteration {i} start,best_model_index: {self.best_model_index}')
            # self_play
            self.self_play(n_simulation=500)
            # 将两个进程的创建的buffer合并
            self.merge_buffer()
            # 训练网络，保存网络
            self.latest_model_index = i
            self.fit(epochs=500)
            torch.save(self.model.state_dict(), f'./data/model_{i}.pt')
            # 评价
            if i % 5 == 0:
                self.eval()

    def self_play(self, n_simulation):
        processes = [mp.Process(target=self_play, args=(self.best_model_index, 50, n_simulation), daemon=True) for _ in
                     range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    def fit(self, epochs=100):
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        for epoch in range(epochs):
            # 批量数据获取，转tensor
            states, pis, zs = self.buffer.get_batch(min_win_ratio=0.5)
            states = torch.as_tensor(states, device=CONFIG['device'])  # 【B，2,H,W]
            pis = torch.as_tensor(pis, device=CONFIG['device'])  # [B,H*W]
            zs = torch.as_tensor(zs, device=CONFIG['device'])  # [B]

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

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.fit_logger.info(
                    f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
                )
        self.latest_infer.update_from_model(self.model)
        duration = time.time() - start
        self.fit_logger.info(f"iteration:{self.latest_model_index}--{epochs}轮训练完成，共用时{duration:.2f}秒。")

    def eval(self):
        start = time.time()
        if self.latest_model_index is None:
            self.latest_model_index = 0
        self.eval_logger.info(f'目前最佳model:{self.best_model_index},待评估model：{self.latest_model_index}')
        with ThreadPoolExecutor(8, thread_name_prefix='eval-') as pool:
            futures = []
            for _ in range(20):
                env = GomokuEnv(*CONFIG['board_shape'])
                players = [AIServer(self.latest_infer), AIServer(self.best_infer)]
                futures.append(pool.submit(env.random_order_play, players, silent=True))
            result = []
            for future in as_completed(futures):
                result.append(future.result())
            win_rate = result.count((1, 0)) / len(result)
            draw_rate = result.count((0, 0)) / len(result)
            self.eval_logger.info(f"win_rate:{win_rate:.2%},draw_rate:{draw_rate:.2%}")
            duration = time.time() - start
            if win_rate + draw_rate / 2 >= 0.55:
                self.best_model_index = self.latest_model_index
                self.best_infer.update_from_index(self.best_model_index)
                write_best_index(self.best_model_index)
                self.eval_logger.info(f'最佳玩家更新为{self.best_model_index},评估用时:{duration:.2f}秒')
            else:
                self.eval_logger.info(f'最佳玩家未更新,仍旧为{self.best_model_index},评估用时:{duration:.2f}秒')

    def shutdown(self):
        self.best_infer.shutdown()
        self.latest_infer.shutdown()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(55, 100)
    trainer.shutdown()
