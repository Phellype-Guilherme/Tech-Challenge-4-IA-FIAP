from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

@dataclass
class EmotionResult:
    label: str
    score: float
    raw: Dict[str, Any]

class DeepFaceEmotion:
    def __init__(self):
        from deepface import DeepFace  # type: ignore
        self.DeepFace = DeepFace

    def analyze(self, bgr_frame: np.ndarray, face_box_xyxy: tuple[int,int,int,int]) -> Optional[EmotionResult]:
        x1,y1,x2,y2 = face_box_xyxy
        h,w,_ = bgr_frame.shape
        bw = max(1, x2-x1)
        bh = max(1, y2-y1)
        pad_x = int(bw * 0.35)
        pad_y = int(bh * 0.45)
        nx1 = max(0, x1-pad_x); nx2 = min(w-1, x2+pad_x)
        ny1 = max(0, y1-pad_y); ny2 = min(h-1, y2+pad_y)
        crop = bgr_frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            res = self.DeepFace.analyze(
                img_path=rgb,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )
            if isinstance(res, list):
                res = res[0]
            emo = res.get("dominant_emotion")
            scores = res.get("emotion", {}) or {}
            score = float(scores.get(emo, 0.0)) / 100.0 if isinstance(scores.get(emo, 0.0), (int,float)) else 0.0
            return EmotionResult(label=str(emo), score=score, raw={"emotion": scores})
        except Exception:
            return None
