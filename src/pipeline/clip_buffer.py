from __future__ import annotations
from collections import deque
from typing import Deque, List, Tuple
import numpy as np
import cv2

class ClipBuffer:
    def __init__(self, maxlen: int = 16, size: int = 112):
        self.maxlen = maxlen
        self.size = size
        self.frames: Deque[np.ndarray] = deque(maxlen=maxlen)

    def add(self, frame_bgr: np.ndarray, box_xyxy: Tuple[int,int,int,int]):
        x1,y1,x2,y2 = box_xyxy
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self.frames.append(rgb)

    def ready(self) -> bool:
        return len(self.frames) >= self.maxlen

    def get(self) -> List[np.ndarray]:
        return list(self.frames)
