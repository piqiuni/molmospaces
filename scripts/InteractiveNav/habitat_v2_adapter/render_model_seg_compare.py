#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import cv2, numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument('--left',type=Path,required=True); p.add_argument('--right',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--ffmpeg',required=True); p.add_argument('--fps',type=float,default=6); p.add_argument('--height',type=int,default=540); a=p.parse_args(); left=sorted(a.left.glob('frame_*.jpg')); right=sorted(a.right.glob('frame_*.jpg'))
 if len(left)!=len(right) or not left: raise SystemExit('frame mismatch')
 first=cv2.imread(str(left[0])); w=int(round(a.height*first.shape[1]/first.shape[0]/2)*2); cmd=[a.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{2*w}x{a.height}','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]; a.output.parent.mkdir(parents=True,exist_ok=True); proc=subprocess.Popen(cmd,stdin=subprocess.PIPE); assert proc.stdin
 for l,r in zip(left,right): proc.stdin.write(np.concatenate([cv2.resize(cv2.imread(str(l)),(w,a.height)),cv2.resize(cv2.imread(str(r)),(w,a.height))],1).tobytes())
 proc.stdin.close();
 if proc.wait(): raise SystemExit('ffmpeg failed')
 print(a.output)
if __name__=='__main__': main()
