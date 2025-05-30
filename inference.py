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


class InferenceEngine:
    def __init__(self, model, batch_size=12, max_delay=1e-3):
        # 深拷贝与训练模型隔离，否则训练无法进行
        self.model = copy.deepcopy(model).to(DEVICE).eval()
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._running = True
        self.thread.start()

    def update_model(self, model):
        self.model.load_state_dict(model.state_dict())

    def _inference_loop(self):
        while self._running:
            batch, results = [], []
            start_time = time.time()
            while len(batch) < self.batch_size and time.time() - start_time < self.max_delay:
                try:
                    state, result = self.queue.get(timeout=1e-6)
                    batch.append(np.transpose(state, axes=[2, 0, 1]))
                    results.append(result)
                except queue.Empty:
                    continue
            if not batch: continue

            batch_tensor = torch.from_numpy(np.stack(batch)).to(DEVICE)
            with torch.no_grad():
                logits, values = self.model(batch_tensor)
                # 避免batch只有一个时，出现维度错误
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                    values = values.unsqueeze(0)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                values = values.cpu().numpy()

            for i in range(len(results)):
                results[i]['policy'] = probs[i]
                results[i]['value'] = values[i]

    def request(self, state, is_self_play=False):
        result = {}
        symmetry_idx = 0 if is_self_play else random.randint(0, 7)
        random_transformed = apply_symmetry(state, symmetry_idx, shape=state.shape[:2])
        self.queue.put((random_transformed, result))
        while 'policy' not in result:
            time.sleep(1e-6)
        reversed_policy = reverse_symmetry_prob(result['policy'], symmetry_idx, shape=state.shape[:2])
        return reversed_policy, result['value']

    def __del__(self):
        self._running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def make_engine(model_idx):
    model = Net(256, 9 * 9)
    model.load_state_dict(torch.load(f'data/model_{model_idx}.pt'))
    return InferenceEngine(model)
