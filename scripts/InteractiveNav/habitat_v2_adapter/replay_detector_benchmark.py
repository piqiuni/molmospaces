#!/usr/bin/env python3
"""Replay saved Habitat RGB frames through several Ultralytics detectors.

This is deliberately offline: it never changes the Habitat policy or feeds
detector output back into navigation.  Without a --gt-json file the reported
target_hit_rate is only a replay proxy, not detection accuracy.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, statistics, time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

ALIASES = {"tv", "television", "tv_monitor", "monitor", "tv monitor"}

def files(root, limit=0):
    xs = sorted(Path(root).glob("*.png"))
    return xs[:limit] if limit else xs

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1]); bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / max(1e-9, aa+bb-inter)

def load_gt(path):
    if not path: return {}
    obj = json.loads(Path(path).read_text())
    return obj.get("frames", obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gt-json", default="")
    ap.add_argument("--visual-reference", default="")
    ap.add_argument("--visual-bbox", default="", help="reference x1,y1,x2,y2 in pixels")
    ap.add_argument("--prompt-list", default="", help="comma-separated text classes for text/world modes")
    ap.add_argument("--model", action="append", default=[], help="name=path; repeatable")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = files(args.frames_dir, args.max_frames)
    if not paths: raise SystemExit("no PNG frames")
    gt = load_gt(args.gt_json)
    prompt_list = [x.strip() for x in args.prompt_list.split(",") if x.strip()] or ["television"]
    specs = []
    for item in args.model:
        name, path = item.split("=", 1); specs.append((name, Path(path)))
    if not specs: raise SystemExit("pass at least one --model name=path")
    all_summary = []
    for name, weight in specs:
        t0 = time.perf_counter();
        if name.startswith("yoloe_text"):
            model = YOLOE(str(weight)); model.set_classes(prompt_list); mode = "text"
        elif name.startswith("yoloe_visual"):
            if not args.visual_reference or not args.visual_bbox:
                raise SystemExit("yoloe_visual requires --visual-reference and --visual-bbox")
            model = YOLOE(str(weight)); mode = "visual"
            prompt = {"bboxes": np.asarray([[float(x) for x in args.visual_bbox.split(",")]], dtype=np.float32),
                      "cls": np.asarray([0], dtype=np.int32)}
            # Build the visual embedding once from a reference outside the scored sequence.
            model.predict(str(paths[0]), refer_image=args.visual_reference, visual_prompts=prompt,
                          predictor=YOLOEVPSegPredictor, device=args.device, imgsz=args.imgsz,
                          conf=args.conf, verbose=False)
        elif name.startswith("yolo_world"):
            model = YOLOWorld(str(weight)); model.set_classes(prompt_list); mode = "text"
        elif name.startswith("yoloe_pf"):
            model = YOLOE(str(weight)); mode = "prompt_free"
        else:
            model = YOLO(str(weight)); mode = "closed_set"
        load_s = time.perf_counter()-t0
        rows=[]; lat=[]; target_hits=0; first_hit=None; gap=0; max_gap=0
        for p in paths[:args.warmup]:
            model.predict(str(p), device=args.device, imgsz=args.imgsz, conf=args.conf, verbose=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        for idx,p in enumerate(paths):
            im=cv2.imread(str(p)); h,w=im.shape[:2]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            st=time.perf_counter(); rr=model.predict(im, device=args.device, imgsz=args.imgsz, conf=args.conf, verbose=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ms=(time.perf_counter()-st)*1000; lat.append(ms)
            det=[]
            r=rr[0]; names=r.names if hasattr(r,'names') else {}
            if r.boxes is not None:
                for box,score,cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    label=str(names.get(int(cls), int(cls))).lower()
                    if mode == "visual" and label == "object0": label = "television"
                    b=[float(x) for x in box]; target=label in ALIASES
                    det.append({"label":label,"confidence":float(score),"bbox_xyxy":b,"target":target})
            targets=[d for d in det if d["target"]]
            hit=bool(targets); target_hits += int(hit)
            if hit and first_hit is None: first_hit=idx+1
            gap = 0 if hit else gap+1; max_gap=max(max_gap,gap)
            row={"frame":p.name,"index":idx+1,"latency_ms":ms,"detections":det}
            g=gt.get(p.name)
            if g:
                gb=g.get("bbox"); visible=bool(g.get("visible", gb is not None)); matches=max((iou(d["bbox_xyxy"],gb) for d in targets), default=0.0) if gb else 0.0
                row["gt"]={"visible":visible,"bbox":gb,"best_iou":matches}
            rows.append(row)
        (out/(name+".jsonl")).write_text("\n".join(json.dumps(x) for x in rows)+"\n")
        peak_alloc=peak_res=0
        if torch.cuda.is_available():
            peak_alloc=torch.cuda.max_memory_allocated()/2**20; peak_res=torch.cuda.max_memory_reserved()/2**20
            torch.cuda.reset_peak_memory_stats()
        label_counts={}
        for row in rows:
            for d in row["detections"]: label_counts[d["label"]]=label_counts.get(d["label"],0)+1
        s={"model":name,"weight":str(weight),"weight_sha256":hashlib.sha256(weight.read_bytes()).hexdigest() if weight.exists() else None,"mode":mode,"prompt_list":prompt_list if mode=="text" else None,"frames":len(paths),"target_hits":target_hits,"target_hit_rate":target_hits/len(paths),"first_target_hit_frame":first_hit,"longest_no_hit_gap":max_gap,"total_detection_boxes":sum(label_counts.values()),"unique_labels":sorted(label_counts),"label_box_counts":dict(sorted(label_counts.items(),key=lambda kv:(-kv[1],kv[0]))),"latency_p50_ms":statistics.median(lat),"latency_p95_ms":float(np.percentile(lat,95)),"latency_mean_ms":statistics.mean(lat),"load_s":load_s,"peak_vram_allocated_mib":peak_alloc,"peak_vram_reserved_mib":peak_res,"accuracy_status":"gt_evaluated" if gt else "no_ground_truth_target_hit_proxy"}
        if gt:
            vis=tp=fp=fn=0
            for row in rows:
                g=row.get("gt");
                if not g: continue
                if g["visible"]: vis+=1
                matched=max((rowd.get("target") and iou(rowd["bbox_xyxy"],g["bbox"]) for rowd in row["detections"]),default=0)>=0.5 if g.get("bbox") else False
                tp += int(matched and g["visible"]); fn += int(g["visible"] and not matched); fp += int(not matched and any(d["target"] for d in row["detections"]))
            s.update({"gt_visible_frames":vis,"tp_iou50":tp,"fp_iou50":fp,"fn_iou50":fn,"precision_iou50":tp/max(1,tp+fp),"recall_iou50":tp/max(1,tp+fn)})
        all_summary.append(s); del model; gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (out/"summary.json").write_text(json.dumps({"frames_dir":args.frames_dir,"frames":len(paths),"target":"television","models":all_summary},indent=2))
    print(json.dumps(all_summary,indent=2))

if __name__ == "__main__": main()
