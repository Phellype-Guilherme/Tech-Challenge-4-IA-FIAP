from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class LabelScore:
    label: str
    score: float

class TemporalSmoother:
    def __init__(self, window: int = 12):
        self.window = window
        self.hist: Dict[int, List[LabelScore]] = {}

    def push(self, tid: int, label: str, score: float) -> LabelScore:
        arr = self.hist.setdefault(tid, [])
        arr.append(LabelScore(label, float(score)))
        if len(arr) > self.window:
            del arr[0:len(arr)-self.window]
        agg: Dict[str, float] = {}
        for ls in arr:
            agg[ls.label] = agg.get(ls.label, 0.0) + max(0.0, ls.score)
        best_label = max(agg.items(), key=lambda kv: kv[1])[0]
        best_score = float(agg[best_label] / max(1, len(arr)))
        return LabelScore(best_label, best_score)

def expand_box(x1: int, y1: int, x2: int, y2: int, shape,
               pad_x_ratio: float = 0.25, pad_top_ratio: float = 0.10, pad_bottom_ratio: float = 1.10):
    h, w = shape[0], shape[1]
    bw = max(1, x2-x1)
    bh = max(1, y2-y1)
    pad_x = int(bw * pad_x_ratio)
    pad_top = int(bh * pad_top_ratio)
    pad_bottom = int(bh * pad_bottom_ratio)
    nx1 = max(0, x1-pad_x)
    nx2 = min(w-1, x2+pad_x)
    ny1 = max(0, y1-pad_top)
    ny2 = min(h-1, y2+pad_bottom)
    return nx1, ny1, nx2, ny2
