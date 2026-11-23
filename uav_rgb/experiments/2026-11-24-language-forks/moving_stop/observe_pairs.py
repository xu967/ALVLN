"""两帧目标观测的冻结标量特征，训练与线上共用。"""
import json
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
DATA=OUT.parent
BASE=DATA/'exp-vln-acquired-approach-839934-20261124'
FEATURES=['two_found','mean_DA2_m','mean_box_area_fraction','mean_touched_border_fraction','mean_abs_horizontal_angle_deg','box_IoU','abs_log_area_ratio','min_DINO_conf']

def pair_features(obs):
    if any(v is None or v.get('box') is None or v.get('range_m') is None for v in obs):return [0.]*8
    boxes=np.clip([v['box'] for v in obs],0,[320,240,320,240]);areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);intersection=np.prod(np.maximum(0,np.min(boxes[:,2:],0)-np.max(boxes[:,:2],0)));touch=(np.abs(boxes-[0,0,320,240])<=1).mean();iou=intersection/(sum(areas)-intersection)
    return [1.,float(np.mean([v['range_m'] for v in obs])),float(areas.mean()/(320*240)),float(touch),float(np.mean([abs(v['angle_deg']) for v in obs])),float(iou),float(abs(np.log(areas[1]/areas[0]))),float(min(v['score'] for v in obs))]

def extract():
    from PIL import Image
    sys.path.insert(0,str(BASE));from approach import make_observer
    observer=make_observer(); samples=json.loads((OUT/'samples.json').read_text());cache={}
    for row in samples:
        obs=[]
        for path in row['rgb_pair']:
            key=(path,row['instruction']);
            if key not in cache:cache[key]=observer(np.asarray(Image.open(path).convert('RGB')),key[1]);
            obs.append(cache[key])
        row.update(views=obs,pair_features=pair_features(obs))
    result=dict(feature_names=FEATURES,samples=samples);tmp=OUT/'features.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(OUT/'features.json');(OUT/'FEATURES_DONE').write_text(json.dumps(dict(rc=0,samples=len(samples),frames=len(cache)))+'\n')

if __name__=='__main__':extract()
