#!/usr/bin/env python3
"""Render aligned detector replay JSONL files as an H.264 comparison video."""
import argparse, json, subprocess, shutil, zlib
from pathlib import Path
import cv2, numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--frames-dir',required=True); p.add_argument('--output',required=True)
    p.add_argument('--result',action='append',required=True,help='label=jsonl'); a=p.parse_args()
    specs=[]
    for x in a.result:
        label,path=x.split('=',1); rows={r['frame']:r for r in map(json.loads,Path(path).read_text().splitlines())}; specs.append((label,rows))
    frames=sorted(Path(a.frames_dir).glob('*.png')); out=Path(a.output); tmp=out.parent/(out.stem+'_frames'); tmp.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(frames,1):
        base=cv2.imread(str(f)); panels=[]
        for label,rows in specs:
            im=base.copy(); row=rows.get(f.name,{})
            for d in row.get('detections',[]):
                x1,y1,x2,y2=map(int,d['bbox_xyxy'])
                seed=zlib.crc32(d['label'].encode()); color=(64+seed%192,64+(seed//193)%192,64+(seed//37249)%192)
                if d.get('target'): color=(0,255,0)
                cv2.rectangle(im,(x1,y1),(x2,y2),color,3 if d.get('target') else 2)
                cv2.putText(im,f"{d['label']} {d['confidence']:.2f}",(max(0,x1),max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
            hit=any(d.get('target') for d in row.get('detections',[])); ms=row.get('latency_ms',0)
            cv2.rectangle(im,(0,0),(im.shape[1],38),(20,20,20),-1)
            cv2.putText(im,f"{label} | {'TV HIT' if hit else 'TV MISS'} | boxes={len(row.get('detections',[]))} | {ms:.1f} ms",(8,25),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),2)
            panels.append(im)
        while len(panels)<6: panels.append(np.zeros_like(base))
        canvas=np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:6])])
        cv2.imwrite(str(tmp/f'{i:06d}.png'),canvas)
    ffmpeg = shutil.which('ffmpeg') or '/home/ldl/conda_envs/mlspaces/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    subprocess.run([ffmpeg,'-y','-framerate','10','-i',str(tmp/'%06d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(out)
if __name__=='__main__': main()
