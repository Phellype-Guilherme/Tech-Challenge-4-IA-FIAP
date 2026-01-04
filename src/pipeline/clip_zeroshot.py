from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class ClipResult:
    label_en: str
    label_pt: str
    score: float
    scores: Dict[str, float]

DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    ("a person dancing", "dançando"),
    ("people working on computers", "trabalhando no computador"),
    ("a business meeting", "reunião de trabalho"),
    ("a dentist treating a patient", "atendimento odontológico"),
    ("a person typing on a laptop", "digitando no notebook"),
    ("a person giving a presentation", "apresentação / palestra"),
    ("a person talking to camera", "falando para câmera"),
    ("a person using a phone", "mexendo no celular"),
    ("a person drinking", "bebendo"),
    ("a person eating", "comendo"),
    ("a person laughing", "rindo"),
    ("a person crying", "chorando"),
    ("a person reading", "lendo"),
    ("a person writing", "escrevendo"),
]

class ClipZeroShot:
    def __init__(self, device: str = "cpu", prompts: Optional[List[Tuple[str,str]]] = None):
        import torch, open_clip
        self.torch = torch
        self.device = torch.device(device)
        self.prompts = prompts or DEFAULT_PROMPTS

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        texts = [p[0] for p in self.prompts]
        with torch.no_grad():
            tokens = tokenizer(texts).to(self.device)
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, rgb_image: np.ndarray) -> ClipResult:
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(rgb_image)
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ self.text_features.T) * 100.0
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        scores = {self.prompts[i][0]: float(probs[i]) for i in range(len(self.prompts))}
        best_i = int(np.argmax(probs))
        en, pt = self.prompts[best_i]
        return ClipResult(label_en=en, label_pt=pt, score=float(probs[best_i]), scores=scores)
