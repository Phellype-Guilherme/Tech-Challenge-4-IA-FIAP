from __future__ import annotations
import argparse, os, json
from typing import Dict, Any, List
import cv2
import numpy as np
from tqdm import tqdm

from src.pipeline.clip_buffer import ClipBuffer
from src.pipeline.action_recog import ActionRecognizer
from src.pipeline.clip_zeroshot import ClipZeroShot
from src.pipeline.emotion import DeepFaceEmotion
from src.pipeline.anomaly import MotionAnomalyDetector
from src.pipeline.summarizer import summarize
from src.utils.utils import TemporalSmoother, expand_box

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--action-every", type=int, default=16)
    p.add_argument("--emotion-every", type=int, default=2)
    p.add_argument("--clip-every", type=int, default=8)
    p.add_argument("--clip-th", type=float, default=0.24)
    p.add_argument("--action-th", type=float, default=0.30)
    p.add_argument("--smooth-window", type=int, default=12)
    p.add_argument("--yolo-model", default="yolov8n.pt")
    return p.parse_args()

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    args = parse_args()
    safe_makedirs(args.out)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir vídeo: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(args.out, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps / max(1,args.frame_skip), (W, H))

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    dev = "cpu" if args.device=="cpu" else "cuda"
    action = ActionRecognizer(device=dev)
    clipzs = ClipZeroShot(device=dev)
    emo = DeepFaceEmotion()

    smoother = TemporalSmoother(window=args.smooth_window)
    clip_buffers: Dict[int, ClipBuffer] = {}
    last_activity: Dict[int, Dict[str, Any]] = {}
    last_emotion: Dict[int, Dict[str, Any]] = {}

    anomaly = MotionAnomalyDetector(window=50, z_th=4.5)

    events: List[Dict[str, Any]] = []
    total_frames_analyzed = 0
    anomaly_count = 0

    frame_index = -1
    analyzed_index = -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total_frames, desc="Processando vídeo", unit="frame")

    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        pbar.update(1)

        if frame_index % args.frame_skip != 0:
            continue

        analyzed_index += 1
        total_frames_analyzed += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
        diff = cv2.absdiff(gray, prev_gray)
        prev_gray = gray
        motion_val = float(np.mean(diff))
        z = anomaly.update(motion_val)
        is_anom = anomaly.is_anomaly(z)
        if is_anom:
            anomaly_count += 1

        results = yolo.track(frame, persist=True, verbose=False, classes=[0])
        r0 = results[0]
        boxes = r0.boxes

        people_payload = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else np.arange(len(xyxy), dtype=int)

            for (x1,y1,x2,y2), tid in zip(xyxy, ids):
                tid = int(tid)
                clip_buffers.setdefault(tid, ClipBuffer(maxlen=args.clip_len, size=112))

                ex1,ey1,ex2,ey2 = expand_box(x1,y1,x2,y2, frame.shape)
                clip_buffers[tid].add(frame, (ex1,ey1,ex2,ey2))

                act = None
                # CLIP
                if analyzed_index % args.clip_every == 0:
                    crop = frame[ey1:ey2, ex1:ex2]
                    if crop.size != 0:
                        rgb = cv2.cvtColor(cv2.resize(crop, (224,224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                        cr = clipzs.predict(rgb)
                        sm = smoother.push(tid, cr.label_pt, cr.score)
                        if cr.score >= args.clip_th:
                            act = {"label_pt": sm.label, "score": float(cr.score), "source": "clip"}

                # Action fallback
                if act is None and analyzed_index % args.action_every == 0 and clip_buffers[tid].ready():
                    ar = action.predict(clip_buffers[tid].get())
                    if ar and ar.score >= args.action_th:
                        act = {"label_pt": ar.label_human, "score": float(ar.score), "source": "action", "raw": ar.label_raw}

                if act is None:
                    act = last_activity.get(tid, {"label_pt":"atividade indefinida","score":0.0,"source":"none"})
                last_activity[tid] = act

                # Emotion (proxy box)
                if analyzed_index % args.emotion_every == 0:
                    er = emo.analyze(frame, (x1,y1,x2,y2))
                    if er:
                        last_emotion[tid] = {"label": er.label, "score": float(er.score), "raw": er.raw}
                emo_payload = last_emotion.get(tid)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"Pessoa {tid} | {act['label_pt']} ({act['source']})", (x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                if emo_payload:
                    cv2.putText(frame, f"{emo_payload['label']} ({emo_payload['score']:.2f})", (x1, min(H-10, y2+18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

                people_payload.append({
                    "track_id": tid,
                    "box": [int(x1),int(y1),int(x2),int(y2)],
                    "activity": act,
                    "emotion": emo_payload,
                })

        if is_anom and z is not None:
            cv2.putText(frame, f"ANOMALIA z={z:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        writer.write(frame)

        events.append({
            "frame_index": frame_index,
            "analyzed_index": analyzed_index,
            "timestamp_sec": float(frame_index / fps),
            "motion": {"mean_absdiff": motion_val, "z": float(z) if z is not None else None},
            "anomaly": {"z": float(z)} if is_anom and z is not None else None,
            "people": people_payload
        })

    pbar.close()
    cap.release()
    writer.release()

    report = summarize(events, total_frames_analyzed, anomaly_count)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.out, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total_frames_analyzed": total_frames_analyzed, "anomalies": anomaly_count}, f, ensure_ascii=False, indent=2)

    print(report)
    print(f"\n✅ Vídeo anotado: {out_video_path}")

if __name__ == "__main__":
    main()
