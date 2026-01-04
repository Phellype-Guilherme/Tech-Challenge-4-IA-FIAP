from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter, defaultdict

def summarize(events: List[Dict[str, Any]], total_frames_analyzed: int, anomaly_count: int) -> str:
    act_counter = Counter()
    emo_by_person = defaultdict(Counter)
    act_by_person = defaultdict(Counter)
    anomalies = []

    for e in events:
        if e.get("anomaly"):
            anomalies.append(e)
        for p in e.get("people", []):
            tid = p.get("track_id")
            if tid is None:
                continue
            act = (p.get("activity") or {}).get("label_pt") or "atividade indefinida"
            emo = (p.get("emotion") or {}).get("label") or None
            act_counter[act] += 1
            act_by_person[tid][act] += 1
            if emo:
                emo_by_person[tid][emo] += 1

    lines = []
    lines.append("RESUMO AUTOMÁTICO — Tech Challenge (Vídeo)")
    lines.append("="*60)
    lines.append(f"Total de frames analisados: {total_frames_analyzed}")
    lines.append(f"Número de anomalias detectadas: {anomaly_count}")
    lines.append("")
    lines.append("Atividades (frequência aproximada):")
    total_acts = sum(act_counter.values()) or 1
    for act, cnt in act_counter.most_common(12):
        pct = 100.0 * cnt / total_acts
        lines.append(f" - {act}: {cnt} ({pct:.1f}%)")

    lines.append("")
    lines.append("Emoções por pessoa (IDs do tracking):")
    if not emo_by_person:
        lines.append(" - (nenhum rosto/emoção detectado)")
    else:
        for tid in sorted(emo_by_person.keys()):
            c = emo_by_person[tid]
            total = sum(c.values()) or 1
            parts = [f"{k}:{v} ({int(round(100*v/total))}%)" for k,v in c.most_common(4)]
            lines.append(f" - Pessoa {tid}: " + ", ".join(parts))

    lines.append("")
    lines.append("Atividades por pessoa (IDs do tracking):")
    for tid in sorted(act_by_person.keys()):
        c = act_by_person[tid]
        total = sum(c.values()) or 1
        parts = [f"{k}:{v} ({int(round(100*v/total))}%)" for k,v in c.most_common(4)]
        lines.append(f" - Pessoa {tid}: " + ", ".join(parts))

    lines.append("")
    lines.append("Principais anomalias (amostras):")
    for a in anomalies[:8]:
        lines.append(f" - Frame {a['frame_index']}: movimento atípico (z={a['anomaly']['z']:.2f})")
    return "\n".join(lines)
