import copy
import os
import queue
import threading
import multiprocessing as mp
import time
import random
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

from config import CONFIG
from functions import apply_symmetry, reverse_symmetry_prob
from network import Net

settings = CONFIG[CONFIG['game_name']]


class InferenceRequest:
    def __init__(self, state, event):
        self.state = state
        self.policy = None
        self.value = None
        self.event = event


class InferenceEngine:
    def __init__(self, model, batch_size=4, max_delay=1e-3):
        # 深拷贝与训练模型隔离，否则训练无法进行
        self.model = copy.deepcopy(model).to(CONFIG['device']).eval()
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.model_index = -1
        self.thread = None
        self._running = False
        self.start()

    def start(self) -> None:
        """启动推理线程"""
        if not self._running:
            self._running = True
        if self.thread is None:
            self.thread = threading.Thread(target=self._inference_loop, daemon=True, name='IE loop')
            self.thread.start()

    def update_from_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.model.to(CONFIG['device']).eval()
        self.model_index = -1

    def update_from_index(self, index, path_prefix=settings['model_path_prefix']):
        path = path_prefix + f'{index}.pt'
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.to(CONFIG['device']).eval()
            self.model_index = index
            print(f'Model {index} loaded.')
        else:
            print(f'Model {index} not found from "{path}".')

    def _inference_loop(self):
        while self._running:
            batch, requests = [], []
            try:
                # 获取第一个请求， 队列空一直等待put直到超时
                request = self.queue.get(timeout=0.5)
                requests.append(request)
            except queue.Empty:
                # 队列空就从头开始循环
                continue

            # 在最大时间内获取尽可能多的request组成batch
            start = time.time()
            while len(requests) < self.batch_size:
                remaining = self.max_delay - (time.time() - start)
                try:
                    request = self.queue.get(timeout=max(.0, remaining))
                    requests.append(request)
                except queue.Empty:
                    break

            # 处理batch，转为tensor
            batch = [np.transpose(request.state, axes=[2, 0, 1]) for request in requests]
            batch_tensor = torch.as_tensor(np.stack(batch), device=CONFIG['device'])

            # 交模型推理，取回结果
            with torch.no_grad():
                logits, values = self.model(batch_tensor)
                # 避免batch只有一个时，出现维度错误
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                    values = values.unsqueeze(0)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                values = values.cpu().numpy()

            # 结果交给请求方，通知request继续
            for i in range(len(requests)):
                requests[i].policy = probs[i]
                requests[i].value = values[i]
                requests[i].event.set()

    def request(self, state: NDArray, is_self_play=False) -> tuple[NDArray, float]:
        if not self._running:
            raise RuntimeError('Inference engine is not running')
        if is_self_play:  # 自我对弈时数据收集时会进行数据扩充
            symmetry_idx = 0
        elif state.shape == (10, 9, 20):  # 象棋
            symmetry_idx = 0
        else:  # 五子棋
            symmetry_idx = random.randint(0, 7)
        random_transformed_state = apply_symmetry(state, symmetry_idx,
                                                  shape=state.shape[:2]) if symmetry_idx != 0 else state
        event = threading.Event()
        request = InferenceRequest(random_transformed_state, event=event)
        self.queue.put(request)
        event.wait()  # 阻塞等待推理线程处理完并设置 event
        reversed_policy = reverse_symmetry_prob(request.policy, symmetry_idx,
                                                shape=state.shape[:2]) if symmetry_idx != 0 else request.policy
        return reversed_policy, request.value

    def shutdown(self):
        self._running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            self.thread = None

    def __del__(self):
        self.shutdown()

    @classmethod
    def make_engine(cls, model_idx: int) -> Self:
        model = Net(settings['n_filter'], settings['n_cells'], settings['n_res_blocks'], settings['n_channels'],
                    settings['n_actions'])
        infer = cls(model)
        infer.update_from_index(model_idx)
        return infer


def run_mp_infer_engine(model_index, req_q):
    model = Net(settings['n_filter'], settings['n_cells'], settings['n_res_blocks'], settings['n_channels'],
                settings['n_actions']).to(CONFIG['device']).eval()
    path = os.path.join(settings['model_path_prefix'], f'{model_index}.pt')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    batch_size = settings['batch_size']
    max_delay = settings['max_delay']
    running = True
    while running:
        batch, pipes = [], []
        try:
            request = req_q.get(timeout=0.5)
            if request is None:
                break
            batch.append(request[0])
            pipes.append(request[1])
        except queue.Empty:
            continue  # 确保获取至少一个state
        # 在最大时间内获取尽可能多的request组成batch
        start = time.time()
        while len(batch) < batch_size:
            remaining = max_delay - (time.time() - start)
            try:
                request = req_q.get(timeout=max(.0, remaining))
                if request is None:
                    running = False
                    break
                batch.append(request[0])
                pipes.append(request[1])
            except queue.Empty:
                break

        # 处理batch，转为tensor
        batch_tensor = torch.as_tensor(np.stack(batch), device=CONFIG['device'])

        # 交模型推理，取回结果
        with torch.no_grad():
            logits, values = model(batch_tensor)
            # 避免batch只有一个时，出现维度错误
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
                values = values.unsqueeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            values = values.cpu().numpy()

        # 结果交给请求方，通知request继续
        for i, cnn in enumerate(pipes):
            cnn.send((probs[i], values[i]))
            cnn.close()


def request_mp_infer(state, req_q):
    parent_cnn, child_cnn = mp.Pipe()
    if state is not None:
        req_q.put((np.transpose(state, axes=[2, 0, 1]), child_cnn))
    else:
        req_q.put(None)
    policy, value = parent_cnn.recv()
    parent_cnn.close()
    return policy, value
