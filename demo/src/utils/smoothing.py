from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Deque

import numpy as np


class TemporalSmoother:
    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {window}")
        self.window = int(window)
        self.queues: DefaultDict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, face_id: int, prob_spoof: float) -> float:
        queue = self.queues[int(face_id)]
        queue.append(float(prob_spoof))
        return float(np.mean(queue))

    def reset(self) -> None:
        self.queues.clear()
