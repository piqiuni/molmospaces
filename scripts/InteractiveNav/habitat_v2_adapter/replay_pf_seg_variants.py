#!/usr/bin/env python3
"""Run one PF model and render true mask overlays for several thresholds."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np

from optimize_pf_replay import optimize_detections

def draw(image, detections, masks, title):
    canvas=image.copy(); overlay=canvas.copy(); h,w=image.shape[:2]
    for d,mask in zip(detections,masks):
        selected=mask>0.5
        overlay[selected]=(0,180,70) if d['label'] in {'door','cabinet','drawer','refrigerator'} else (255,150,0)
    canvas=cv2.addWeighted(overlay,0.42,canvas,0.58,0)
    for d in detections:
        b=[int(round(v)) for v in d['bbox_xyxy']]
        color=(0,220,80) if d['label'] in {'door','cabinet','drawer','refrigerator'} else (255,180,0)
        cv2.rectangle(canvas,(b[0],b[1]),(b[2],b[3]),color,3,cv2.LINE_AA)
        shown = d['label']
        if d.get('raw_label') and d['raw_label'] != d['label']:
            shown = f"{shown} ({d['raw_label']})"
        cv2.putText(canvas,f"{shown} {d['confidence']:.2f}",(b[0]+2,max(22,b[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(w,44),(12,12,12),-1)
    cv2.putText(canvas,f"{title} | mask instances={len(detections)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.74,(245,245,245),2,cv2.LINE_AA)
    return canvas

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames-dir',type=Path,required=True); ap.add_argument('--weight',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--infer-conf',type=float,default=.25); ap.add_argument('--max-det',type=int,default=60); ap.add_argument('--profile',choices=('raw','interaction'),default='interaction'); ap.add_argument('--thresholds',default='.25,.35,.45,.55'); ap.add_argument('--name',default='YOLOE PF')
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted([p for p in a.frames_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])
    if not paths: raise SystemExit('no frames')
    import torch
    from ultralytics import YOLOE
    model=YOLOE(str(a.weight)); warm=cv2.imread(str(paths[0]));
    for _ in range(3): model.predict(warm,device=a.device,imgsz=a.imgsz,conf=a.infer_conf,max_det=a.max_det,retina_masks=True,verbose=False)
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    thresholds=[float(x) for x in a.thresholds.split(',')]
    dirs={t:a.output_dir/f'threshold_{t:.2f}'.replace('.','p') for t in thresholds}
    for d in dirs.values(): d.mkdir(parents=True,exist_ok=True)
    rows=[]; lat=[]
    threshold_stats={str(t): {'frames_with_boxes': 0, 'boxes': 0, 'label_counts': {}} for t in thresholds}
    for index,path in enumerate(paths,1):
        image=cv2.imread(str(path));
        if torch.cuda.is_available(): torch.cuda.synchronize()
        st=time.perf_counter(); result=model.predict(image,device=a.device,imgsz=a.imgsz,conf=a.infer_conf,max_det=a.max_det,retina_masks=True,verbose=False)[0]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms=(time.perf_counter()-st)*1000; lat.append(ms)
        boxes=result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0,4)); scores=result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty(0); cls=result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.empty(0,dtype=int); masks=result.masks.data.detach().cpu().numpy() if result.masks is not None else np.empty((0,*image.shape[:2])); names=result.names
        all_d=[]
        for j,(b,s,c) in enumerate(zip(boxes,scores,cls)):
            all_d.append({'label':str(names.get(int(c),c)),'confidence':float(s),'bbox_xyxy':[float(v) for v in b],'mask_index':j})
        for threshold in thresholds:
            selected=[d for d in all_d if d['confidence']>=threshold]
            if a.profile=='interaction':
                # optimize_detections performs canonicalization, filtering and class-aware NMS.
                selected=optimize_detections(selected,image.shape[1],image.shape[0],threshold,'interaction')
            stat=threshold_stats[str(threshold)]
            stat['frames_with_boxes'] += int(bool(selected))
            stat['boxes'] += len(selected)
            for item in selected:
                label=item['label']
                stat['label_counts'][label]=stat['label_counts'].get(label,0)+1
            chosen=[]
            # Match optimized records back to the highest-IoU original mask of same label.
            for d in selected:
                candidates=[x for x in all_d if x['label']==d.get('raw_label',x['label']) or x['label']==d['label']]
                source=max(candidates,key=lambda x: x['confidence'],default=None)
                if source is not None: chosen.append(masks[int(source['mask_index'])])
            cv2.imwrite(str(dirs[threshold]/f'frame_{index:04d}.jpg'),draw(image,selected,chosen,f'{a.name} {a.profile} conf>={threshold:.2f}'),[cv2.IMWRITE_JPEG_QUALITY,96])
        rows.append({'index':index,'source':str(path),'latency_ms':ms,'raw_detections':len(all_d)})
    for stat in threshold_stats.values():
        stat['label_counts']=dict(sorted(stat['label_counts'].items(), key=lambda kv:(-kv[1],kv[0])))
    summary={'model':a.name,'weight':str(a.weight),'weight_sha256':hashlib.sha256(a.weight.read_bytes()).hexdigest(),'frames':len(rows),'profile':a.profile,'thresholds':thresholds,'threshold_summary':threshold_stats,'latency_p50_ms':float(np.median(lat)),'latency_p95_ms':float(np.percentile(lat,95)),'latency_mean_ms':float(np.mean(lat)),'peak_vram_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.0,'peak_vram_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else 0.0}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)); (a.output_dir/'timing.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
