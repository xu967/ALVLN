"""在已冻结的检测框追踪器上接入双帧判停。"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from observe_pairs import pair_features,FEATURES,BASE,OUT
sys.path.insert(0,str(BASE))
from approach import VisualApproach

class MovingStop(VisualApproach):
    def __init__(self):
        super().__init__()
        self.previous=None
        self.model=json.loads((OUT/'model.json').read_text())
        assert self.model['feature_names']==FEATURES

    def decide(self,rgb,text):
        _,_,last,ms=super().decide(rgb,text)
        obs=last['dino'];xx=pair_features([self.previous,obs]);self.previous=obs
        logit=self.model['intercept']+sum(w*x for w,x in zip(self.model['coef'],xx));stop=bool(xx[0] and logit>=0);angle=obs['angle_deg']
        action=0 if stop else 2 if angle is None or angle< -9 else 3 if angle>9 else 1
        return action,stop,dict(dino=obs,pair_features=xx,raw_logit=logit,action_proposed=action,action_intervention='task_stop' if stop else None,stop_source='moving_head' if stop else None,model_stop_requested=stop,near_streak=int(stop)*2),ms

def run():
    rc=1
    try:
        reg=json.loads((BASE/'registration.json').read_text())['rows'];args=json.loads((BASE.parent/'exp-vln-near-start-839934-20261124/registration.json').read_text())['arms']
        for row,old in zip(reg,args):
            name='moving_'+row['visible']+'_to_'+row['requested'];cmd=old['argv'].copy();dest=OUT/name;cmd[cmd.index('--scene')+1]=row['fixture'];cmd[cmd.index('--out')+1]=str(dest);cmd[cmd.index('--seed')+1]='421';env=os.environ.copy();env['PYTHONPATH']=str(OUT)+os.pathsep+str(OUT/'runtime')+os.pathsep+env.get('PYTHONPATH','');print(name,datetime.datetime.now().astimezone().isoformat(),flush=True);subprocess.run(cmd,cwd=OUT/'runtime',env=env,check=True);r=json.loads((dest/'vln_eval.json').read_text());assert not r['failures'],r['failures']
        rc=0
    finally:(OUT/'ONLINE_DONE').write_text(json.dumps(dict(rc=rc,end=datetime.datetime.now().astimezone().isoformat()))+'\n')

if __name__=='__main__':run()
