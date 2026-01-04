from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class ActionResult:
    label_raw: str
    label_human: str
    score: float

HUMAN_MAP = {
    "dancing": "dançando",
    "walking": "andando",
    "running": "correndo",
    "sitting": "sentado",
    "standing": "em pé",
    "clapping": "aplaudindo",
    "talking": "conversando",
    "drinking": "bebendo",
    "eating": "comendo",
    "typing": "digitando",
    "reading": "lendo",
    "writing": "escrevendo",
    "using computer": "trabalhando no computador",
    "answering questions": "reunião / conversa",
    "laughing": "rindo",
    "crying": "chorando",
}

def to_human(label: str) -> str:
    l = label.lower()
    for k, v in HUMAN_MAP.items():
        if k in l:
            return v
    return label

class ActionRecognizer:
    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.video import r3d_18, R3D_18_Weights
        self.torch = torch
        self.device = torch.device(device)
        weights = R3D_18_Weights.KINETICS400_V1
        self.categories = weights.meta["categories"]
        self.model = r3d_18(weights=weights).to(self.device)
        self.model.eval()
        self.banned = {"massaging person's head", "applying cream", "waxing eyebrows", "dying hair"}

    def predict(self, clip_rgb: List[np.ndarray]) -> Optional[ActionResult]:
        if not clip_rgb or len(clip_rgb) < 8:
            return None
        import torch
        arr = np.stack(clip_rgb, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
        x = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (C,T,H,W)
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        x = (x - mean) / std
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        topk = torch.topk(probs, k=5)
        chosen = None
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            label = self.categories[int(idx)]
            if label in self.banned and score < 0.80:
                continue
            chosen = (label, score)
            break
        if not chosen:
            return None
        label, score = chosen
        return ActionResult(label_raw=label, label_human=to_human(label), score=float(score))
