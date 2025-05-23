import os.path
import threading
import time

import torch
import numpy as np
from torch import optim
import multiprocessing as mp
from board import GomokuEnv
from config import MODEL_PATH
from deepMcts import DeepMCTS
from network import Net
from replay import ReplayBuffer
import torch.nn.functional as F
import queue


def self_play(inference_queue, result_dict, env, n_games, n_simulation, temperature=0.5):
    dataset = []

    for i in range(n_games):
        states = []
        pis = []
        players = []
        current_player = 0
        state, _ = env.reset()
        mcts = DeepMCTS(state, inference_queue, result_dict)
        done = False
        reward = 0

        while not done:
            mcts.run(n_simulation)
            pi = mcts.get_pi(temperature)
            states.append(np.copy(state))
            pis.append(np.copy(pi))
            players.append(current_player)

            action = np.random.choice(len(pi), p=pi)
            state, reward, done, _, _ = env.step(action)
            mcts.apply_move(env.action2index(action))
            current_player = 1 - current_player
        winner = env.get_winner(reward, 1 - current_player)  # winner ID（0，1）,平局-1
        for state, pi, player in zip(states, pis, players):
            if winner == player:
                z = 1
            elif winner == -1:
                z = 0
            else:
                z = -1
            dataset.append((state, pi, z))
        print(f'Game {i + 1}: winner is {winner}')
    return dataset


def _self_play_worker(arg):
    return self_play(*arg)


def parallel_self_play(inference_queue, result_dict, make_env, total_games, n_simulation, temperature=1, n_workers=16):
    games_per_worker = total_games // n_workers
    tasks = [(inference_queue, result_dict, make_env(), games_per_worker, n_simulation, temperature) for _ in
             range(n_workers)]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_self_play_worker, tasks)
    combined = []
    for result in results:
        combined.extend(result)
    return combined


def inference_loop(model, inference_queue, result_dict, batch_size, max_delay, device):
    while True:
        batch = []
        ids = []
        start_time = time.time()
        while len(batch) < batch_size and time.time() - start_time < max_delay:
            try:
                req_id, state = inference_queue.get(timeout=0.02)
                ids.append(req_id)
                batch.append(np.transpose(state, axes=[2, 0, 1]))
            except queue.Empty:
                continue
        if len(batch) == 0:
            continue
        batch = np.stack(batch, axis=0)
        state = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            policy_logits, value = model(state)
        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
        value = value.cpu().numpy()
        for idx, req_id in enumerate(ids):
            result_dict[req_id] = {
                'policy': policy_probs[idx],
                'value': value[idx]
            }


def start_inference(model, inference_queue, result_dict, batch_size=12, max_delay=0.05, device='cpu'):
    model.eval()
    model.to(device)
    thread = threading.Thread(target=inference_loop,
                              args=(model, inference_queue, result_dict, batch_size, max_delay, device),
                              daemon=True)
    thread.start()


def train(model, buffer, optimizer, epochs=1, device='cpu'):
    model.to(device)
    for epoch in range(epochs):
        # 批量数据获取，转tensor
        states, pis, zs = buffer.get_batch()
        states = torch.tensor(states, dtype=torch.float32).to(device)  # 【B，2,H,W]
        pis = torch.tensor(pis, dtype=torch.float32).to(device)  # [B,H*W]
        zs = torch.tensor(zs, dtype=torch.float32).to(device)  # [B]

        # 模型前向推理
        policy_logits, values = model(states)

        # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
        policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
        # 均方差损失，使value的结果趋近与mcts模拟出来的z，[B]。(values-z)**2最坏的情况为4，前面乘0.25是为了归一化，
        value_loss = 0.25 * F.mse_loss(values, zs)

        # 用总的损失进行反向梯度更新
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
                f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
            )


if __name__ == '__main__':
    mp.set_start_method('spawn')  # 避免多线程报错
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w = 9, 9
    net = Net(256, h * w).to(my_device)  # 神经网络
    replay_buffer = ReplayBuffer(30000, 128)  # 数据缓存

    # 加载之前的数据
    replay_buffer.load()

    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH))

    # 推理队列，启动推理线程
    manager = mp.Manager()
    request_queue = manager.Queue()
    result_map = manager.dict()
    start_inference(net, request_queue, result_map, device=my_device)

    # 优化器
    adam = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # 总的轮次
    iteration_start = int(MODEL_PATH.split('_')[-1].split('.')[0])

    # 本次训练
    for i in range(30):
        print(f'Current Iteration:{i + 1},Total Iteration:{iteration_start + i + 1}')
        start = time.time()
        # 并行采集对战数据
        experiences = parallel_self_play(
            inference_queue=request_queue,
            result_dict=result_map,
            make_env=lambda: GomokuEnv(h, w),
            total_games=24,
            n_simulation=200,
            temperature=1,
            n_workers=24
        )
        # 串行
        # experiences = self_play(net, GomokuEnv(h, w), 20, 200, temperature=1)

        print(f'采集到{len(experiences)}数据，用时{time.time() - start:.2f}秒。')

        start = time.time()
        for experience in experiences:
            replay_buffer.add(*experience)

        # 进行训练，使神经网络学习mcts经验,weight_decay为l2正则
        train(net, replay_buffer, adam, epochs=100, device=my_device)
        print(f"训练完成，用时{time.time() - start:.2f}秒。")
        # 保存数据，便于以后使用
        replay_buffer.save()
        torch.save(net.state_dict(), f'./data/model_{iteration_start + i + 1}.pt')
