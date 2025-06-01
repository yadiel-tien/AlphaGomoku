import copy
import queue
import threading
import time
import random

import numpy as np
import torch

from config import DEVICE
from functions import apply_symmetry, reverse_symmetry_prob
from network import Net


class InferenceRequest:
    def __init__(self, state, event):
        self.state = state
        self.policy = None
        self.value = None
        self.event = event


class InferenceEngine:
    def __init__(self, model, batch_size=12, max_delay=5e-4):
        # 深拷贝与训练模型隔离，否则训练无法进行
        self.model = copy.deepcopy(model).to(DEVICE).eval()
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.condition = threading.Condition()
        self.thread = None
        self._running = False
        self.start()

    def start(self):
        if not self._running:
            self._running = True
        if self.thread is None:
            self.thread = threading.Thread(target=self._inference_loop, daemon=True,name='IE loop')
            self.thread.start()

    def update_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.model.to(DEVICE).eval()

    def _inference_loop(self):
        while self._running:
            batch, requests = [], []
            with self.condition:
                while self.queue.empty() and self._running:
                    # 队列空一直等待通知
                    self.condition.wait(timeout=0.5)
                # self._running=False,退出
                if self.queue.empty():
                    break
                # 获取第一个请求
                request = self.queue.get_nowait()
                requests.append(request)

            start = time.time()
            while len(requests) < self.batch_size:
                remaining = self.max_delay - (time.time() - start)
                try:
                    request = self.queue.get(timeout=max(0, remaining))
                    requests.append(request)
                except queue.Empty:
                    break
            batch = [np.transpose(request.state, axes=[2, 0, 1]) for request in requests]
            batch_tensor = torch.from_numpy(np.stack(batch)).to(DEVICE)
            with torch.no_grad():
                logits, values = self.model(batch_tensor)
                # 避免batch只有一个时，出现维度错误
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                    values = values.unsqueeze(0)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                values = values.cpu().numpy()

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
        with self.condition:
            self.queue.put(request)
            self.condition.notify()
        event.wait()
        reversed_policy = reverse_symmetry_prob(request.policy, symmetry_idx, shape=state.shape[:2])
        return reversed_policy, request.value

    def shutdown(self):
        self._running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            self.thread = None

    def __del__(self):
        self.shutdown()


def make_engine(model_idx):
    model = Net(256, 9 * 9)
    model.load_state_dict(torch.load(f'data/model_{model_idx}.pt'))
    return InferenceEngine(model)
