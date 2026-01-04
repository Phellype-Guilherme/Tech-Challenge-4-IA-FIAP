from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

class MotionAnomalyDetector:
    def __init__(self, window: int = 50, z_th: float = 4.5):
        self.window = window
        self.z_th = z_th
        self.values: List[float] = []

    def update(self, motion_value: float) -> Optional[float]:
        self.values.append(float(motion_value))
        if len(self.values) < self.window:
            return None
        w = np.array(self.values[-self.window:], dtype=np.float32)
        mu = float(w.mean())
        sd = float(w.std() + 1e-6)
        return float((motion_value - mu) / sd)

    def is_anomaly(self, z: Optional[float]) -> bool:
        return z is not None and abs(z) >= self.z_th
