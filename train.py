import glob
import os.path
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm

from chess import ChineseChess
from gomoku import GomokuEnv
from config import SETTINGS, CONFIG
from deepMcts import NeuronMCTS
from inference import InferenceEngine, make_engine, run_mp_infer_engine
from network import Net
from player import AIServer
from replay import ReplayBuffer
import torch.nn.functional as F
import logging
import multiprocessing as mp

global_req_q = None


def init_worker(req_q):
    global global_req_q
    global_req_q = req_q


def get_logger(name, log_dir=SETTINGS['log_dir']):
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


def self_play1game(n_simulation):
    env = GomokuEnv()
    env.reset()
    mcts = NeuronMCTS(env, global_req_q, is_self_play=True)
    step = 0
    experiences = []

    while not env.terminated:
        temperature = 0.2 if step > 2 else 1  # 前几步鼓励探索
        mcts.run(n_simulation)  # 模拟
        pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi
        q = mcts.root.W / (mcts.root.N + 1)
        experiences.append((np.copy(env.state), pi, q))
        # 根据pi来选择动作
        action = np.random.choice(len(pi), p=pi)
        env.step(action)  # 执行落子
        mcts.apply_action(env)  # mcts也要根据action进行对应裁剪
        step += 1
    return experiences


def self_play_worker(best_model_index, n_games, n_simulation):
    """自我对弈，收集每步的state，pi，z"""

    play_logger = get_logger('selfplay')
    process_name = mp.current_process().name
    play_logger.info(f'{process_name} self play begin...')
    infer = make_engine(best_model_index)

    start = time.time()
    dataset = []
    with  ThreadPoolExecutor(8, thread_name_prefix='self_play-') as pool:
        futures = [pool.submit(self_play1game, infer, n_simulation) for _ in range(n_games)]
        z_sample = []
        game_count, data_count = 0, 0
        for f in as_completed(futures):
            game_count += 1
            experiences = f.result()
            z_sample.append(experiences[0][2])
            data_count += len(experiences)
            dataset.extend(experiences)
            if game_count % 10 == 0:
                draw_count = z_sample.count(0)
                win_count = len(z_sample) - draw_count
                play_logger.info(f'{process_name}:self playing 10 games,win/loss:{win_count} ,draw:{draw_count}')
                duration = time.time() - start
                play_logger.info(
                    f'{process_name}:{game_count // 10}:采集到{data_count}条原始数据，用时{duration:.2f}秒,平均每个用时数据用时{duration / data_count :.4f}秒。'
                )
                data_count = 0
                z_sample = []
                start = time.time()
    infer.shutdown()
    play_logger.info(f'{process_name} self play end')
    return dataset


def read_best_index():
    if os.path.exists(SETTINGS['best_index_path']):
        with open(SETTINGS['best_index_path'], "rb") as f:
            return pickle.load(f)
    return None


def read_latest_index():
    prefix = SETTINGS['model_path_prefix']
    patten = prefix + '*.pt'
    model_files = glob.glob(patten)
    if model_files:
        return max(
            int(f.split("_")[1].split(".")[0])
            for f in model_files
        )
    else:
        return None


def write_best_index(best_index):
    with open(SETTINGS['best_index_path'], "wb") as f:
        pickle.dump(best_index, f)


class Trainer:
    def __init__(self):
        self.logger = get_logger('main')
        self.fit_logger = get_logger('fit')
        self.eval_logger = get_logger('eval')
        self.selfplay_logger = get_logger('selfplay')
        self.best_model_index = read_best_index()
        self.best_infer = make_engine(self.best_model_index)
        self.latest_model_index = read_latest_index()
        self.model = Net(SETTINGS['n_filter'], SETTINGS['n_cells'], SETTINGS['n_res_blocks'],
                         SETTINGS['n_channels'], SETTINGS['n_actions']).to(CONFIG['device'])
        if self.latest_model_index is not None:
            if self.latest_model_index - self.best_model_index >= 30:
                self.latest_model_index = self.best_model_index
            self.logger.info(f'最新模型加载为{self.latest_model_index}.')
            self.model.load_state_dict(
                torch.load(SETTINGS['model_path_prefix'] + f'{self.latest_model_index}.pt', map_location=CONFIG['device'])
            )
        else:
            self.latest_model_index = 1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.latest_infer = InferenceEngine(self.model)
        self.buffer = ReplayBuffer(500_000, 128)
        self.buffer.load()

    def run(self, iteration, n_simulation, n_evaluation):
        mp.set_start_method('spawn')
        for i in range(iteration):
            self.logger.info(
                f'iteration {i + 1}/{iteration} start,best: {self.best_model_index},latest: {self.latest_model_index}')
            # self_play
            self.self_play(n_simulation=n_simulation)
            # 训练网络，保存网络
            self.latest_model_index += 1
            self.fit(epochs=100)
            torch.save(self.model.state_dict(), f'./data/model_{self.latest_model_index}.pt')
            # 评价
            if (i + 1) % 5 == 0:
                self.eval(n_evaluation)

    def self_play(self, n_simulation, n_games=100):
        dataset = []
        start = time.time()
        req_q = mp.Queue()
        # 启动推理模型
        infer_proc = mp.Process(target=run_mp_infer_engine, args=(self.best_model_index, req_q))
        infer_proc.start()
        with mp.Pool(processes=30, initializer=init_worker, initargs=(req_q,)) as pool:
            results = [pool.apply_async(self_play1game, args=(n_simulation,)) for _ in
                       range(n_games)]
            for res in tqdm(results):
                dataset.extend(res.get())

        req_q.put(None)  # 关闭推理信号
        infer_proc.join()

        for data in dataset:
            self.buffer.add(*data)
        self.buffer.save()
        self.selfplay_logger.info(
            f'selfplay {n_games}局游戏，收集到原始数据{len(dataset)}条,对战model:{self.best_model_index},耗时{time.time() - start:.2f}秒')

    def fit(self, epochs=100):
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        for epoch in range(epochs):
            # 批量数据获取，转tensor
            states, pis, zs = self.buffer.get_batch()
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
        self.fit_logger.info(f"iteration{self.latest_model_index}:{epochs}轮训练完成，共用时{duration:.2f}秒。")

    def eval(self, n_evaluation):
        start = time.time()
        if self.latest_model_index is None:
            self.latest_model_index = 0
        self.eval_logger.info(f'目前最佳model:{self.best_model_index},待评估model：{self.latest_model_index}')
        with ThreadPoolExecutor(8, thread_name_prefix='eval-') as pool:
            futures = []
            for _ in range(n_evaluation):
                env = GomokuEnv()
                players = [AIServer(self.latest_infer), AIServer(self.best_infer)]
                futures.append(pool.submit(env.random_order_play, players, silent=True))
            result = []
            for future in as_completed(futures):
                result.append(future.result())
            win_rate = result.count(0) / len(result)
            draw_rate = result.count(-1) / len(result)
            self.eval_logger.info(f"win_rate:{win_rate:.2%},draw_rate:{draw_rate:.2%}")
            duration = time.time() - start
            if win_rate + draw_rate / 2 >= 0.55:
                self.best_model_index = self.latest_model_index
                self.best_infer.update_from_index(self.best_model_index)
                write_best_index(self.best_model_index)
                self.eval_logger.info(f'最佳玩家更新为{self.best_model_index},评估用时:{duration:.2f}秒')
            elif self.latest_model_index - self.best_model_index >= 30 or win_rate < 0.4:
                self.model.load_state_dict(torch.load(f'./data/model_{self.best_model_index}.pt'))
                self.latest_infer.update_from_model(self.model)
                self.latest_model_index = self.best_model_index
                self.eval_logger.info(
                    f'因最新玩家表现不佳，退回到{self.best_model_index}重新训练,评估用时:{duration:.2f}秒')
            else:
                self.eval_logger.info(f'最佳玩家未更新,仍旧为{self.best_model_index},评估用时:{duration:.2f}秒')

    def shutdown(self):
        self.best_infer.shutdown()
        self.latest_infer.shutdown()


if __name__ == '__main__':
    trainer = Trainer()
    mp.set_start_method('spawn')
    # trainer.run(iteration=200, n_simulation=200, n_evaluation=50, n_workers=2)
    trainer.self_play(100, 30)
    trainer.shutdown()
    # write_best_index(130)
