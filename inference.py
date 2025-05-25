import queue
import threading
import time

import numpy as np
import torch


class InferenceEngine:
    def __init__(self, model, device='cpu', batch_size=12, max_delay=1e-3):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._running = True
        self.thread.start()

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
            # print(f'batch size: {len(batch)}')
            batch_tensor = torch.from_numpy(np.stack(batch)).to(self.device)
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

    def request(self, state):
        result = {}
        self.queue.put((state, result))
        while 'policy' not in result:
            time.sleep(1e-6)
        return result['policy'], result['value']

    def shutdown(self):
        self._running = False
        self.thread.join()
