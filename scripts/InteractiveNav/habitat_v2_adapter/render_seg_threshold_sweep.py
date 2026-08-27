#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import cv2, numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument('--variant-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--ffmpeg',required=True); p.add_argument('--fps',type=float,default=6); p.add_argument('--height',type=int,default=540); p.add_argument('--thresholds',default='0.25,0.35,0.45,0.55'); a=p.parse_args()
 dirs=[a.variant_dir/f'threshold_{x.replace(".","p")}' for x in a.thresholds.split(',')]; groups=[sorted(d.glob('frame_*.jpg')) for d in dirs]; n={len(x) for x in groups}
 if len(n)!=1 or not next(iter(n)): raise SystemExit([len(x) for x in groups])
 first=cv2.imread(str(groups[0][0])); w=int(round(a.height*first.shape[1]/first.shape[0]/2)*2); outw=w*len(groups); outh=a.height
 cmd=[a.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{outw}x{outh}','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]; a.output.parent.mkdir(parents=True,exist_ok=True); proc=subprocess.Popen(cmd,stdin=subprocess.PIPE); assert proc.stdin
 for files in zip(*groups):
  imgs=[cv2.resize(cv2.imread(str(f)),(w,a.height),interpolation=cv2.INTER_AREA) for f in files]; proc.stdin.write(np.concatenate(imgs,axis=1).tobytes())
 proc.stdin.close();
 if proc.wait(): raise SystemExit('ffmpeg failed')
 print(a.output)
if __name__=='__main__': main()
