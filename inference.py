import copy
import queue
import threading
import time
import random

import numpy as np
import torch

from config import CONFIG
from functions import apply_symmetry, reverse_symmetry_prob
from network import Net


class InferenceRequest:
    def __init__(self, state, event):
        self.state = state
        self.policy = None
        self.value = None
        self.event = event


class InferenceEngine:
    def __init__(self, model, batch_size=8, max_delay=1e-3):
        # 深拷贝与训练模型隔离，否则训练无法进行
        self.model = copy.deepcopy(model).to(CONFIG['device']).eval()
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.thread = None
        self._running = False
        self.start()

    def start(self):
        if not self._running:
            self._running = True
        if self.thread is None:
            self.thread = threading.Thread(target=self._inference_loop, daemon=True, name='IE loop')
            self.thread.start()

    def update_from_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.model.to(CONFIG['device']).eval()

    def update_from_index(self, index):
        self.model.load_state_dict(torch.load(f'data/model_{index}.pt'))
        self.model.to(CONFIG['device']).eval()

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

    def request(self, state, is_self_play=False):
        if not self._running:
            raise RuntimeError('Inference engine is not running')
        symmetry_idx = 0 if is_self_play else random.randint(0, 7)
        random_transformed_state = apply_symmetry(state, symmetry_idx, shape=state.shape[:2])
        event = threading.Event()
        request = InferenceRequest(random_transformed_state, event=event)
        self.queue.put(request)
        event.wait()  # 阻塞等待推理线程处理完并设置 event
        reversed_policy = reverse_symmetry_prob(request.policy, symmetry_idx, shape=state.shape[:2])
        return reversed_policy, request.value

    def shutdown(self):
        self._running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            self.thread = None

    def __del__(self):
        self.shutdown()


def make_engine(model_idx):
    h, w = CONFIG['board_shape']
    model = Net(256, h * w)
    if model_idx is not None:
        model.load_state_dict(torch.load(f'data/model_{model_idx}.pt'))
    return InferenceEngine(model)
