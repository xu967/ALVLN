import json
import sys
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

home=Path(__file__).resolve().parent
model_root=home.parent/'baselines/depth-anything-v2'
sys.path.insert(0,str(model_root/'src/metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2
model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
model.load_state_dict(torch.load(model_root/'weights/depth_anything_v2_metric_hypersim_vits.pth',map_location='cpu',weights_only=True))
model=model.cuda().eval()

def estimate(bgr):
    return model.infer_image(bgr,input_size=322).astype('<f4')

def score_scale():
    reg=json.loads((home/'registration.json').read_text())
    samples=[]
    for item in reg['calibration_frames']:
        pred=estimate(cv2.imread(item['rgb']))[::4,::4]
        gt=np.load(item['depth'])[::4,::4]
        mask=np.isfinite(pred)&(pred>0)&np.isfinite(gt)&(gt>=.2)&(gt<=8)
        assert mask.sum()>100
        samples.append((item,gt[mask],pred[mask]))
    scale=float(np.median([np.median(gt/pr) for _,gt,pr in samples]))
    result={'scale':scale,'scenes':{},'n_frames':len(samples)}
    for scene in sorted({a['scene'] for a,_,_ in samples}):
        group=[(gt,pr) for a,gt,pr in samples if a['scene']==scene]
        corr=[spearmanr(gt,pr).statistic for gt,pr in group]
        y=np.concatenate([gt for gt,pr in group]);raw=np.concatenate([pr for gt,pr in group])
        near=y<.8
        result['scenes'][scene]={'median_frame_rank':float(np.nanmedian(corr)),'near_pixels':int(near.sum()),
            'metrics':{name:{'median_abs_m':float(np.median(abs(v-y))),'abs_rel':float(np.mean(abs(v-y)/y)),
                             'near_actual_predicted_over_1m':int((near & (v>1.)).sum())} for name,v in [('raw',raw),('corrected',raw*scale)]}}
    result['gate']=bool(.33<=scale<=3 and np.median([q['median_frame_rank'] for q in result['scenes'].values()])>.2)
    (home/'metric_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    return

def serve():
    scale=json.loads((home/'metric_check.json').read_text())['scale']
    src=sys.stdin.buffer
    target=sys.stdout.buffer
    target.write(b'RDY!');target.flush()
    while True:
        packet=src.read(320*240*3)
        if not packet:break
        assert len(packet)==320*240*3
        rgb=np.frombuffer(packet,np.uint8).reshape((240,320,3))
        d=estimate(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))*scale
        target.write(d.astype('<f4').tobytes());target.flush()
if sys.argv[1]=='calibrate':score_scale()
else:serve()
