import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
DATA=OUT.parent
REG=json.loads((OUT/'registration.json').read_text())
sys.path.insert(0,str(OUT/'runtime'))

def preflight():
    from PIL import Image
    S=importlib.import_module('sim.3dgs.scene');C=importlib.import_module('sim.3dgs.collision');D=importlib.import_module('sim.3dgs.dynamics');R=importlib.import_module('sim.3dgs.renderer')
    records=[];renderer=None;occ=None
    for row in REG['rows'][::2]:
        s=S.load_scene(row['fixture']);r=row['start'];s['spawn']=r
        if renderer is None:renderer=R.make_renderer(s);occ=C.from_scene(s)
        d=D.CfSitlDynamics(s,seed=row['seed']);st=d.reset(r['x'],r['y'],r['yaw_deg'],episode=0);frame,_,_=renderer.render_view(d.camera_viewmat(s['_nav_to_world']));fp=OUT/(row['visible']+'_start.png');Image.fromarray(frame).save(fp);delta=np.asarray(st['pose'])-np.asarray([row['teacher_pose'][k] for k in 'xyz']); records.append(dict(name=row['visible'],image=str(fp),teacher_rgb=row['teacher_rgb'],xyz=st['pose'],quat=st['quat'],reset_minus_teacher_m=delta.tolist(),yaw_teacher_deg=r['yaw_deg'],yaw_actual_deg=float(np.degrees(d.yaw)),clearance_m=occ.clearance(*st['pose'])));d.close()
    (OUT/'preflight.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__' and sys.argv[1]=='preflight':preflight()

def make_observer():
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    root=DATA/'baselines'; depth_root=root/'depth-anything-v2'; sys.path.insert(0,str(depth_root/'src/metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    depth=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    depth.load_state_dict(torch.load(depth_root/'weights/depth_anything_v2_metric_hypersim_vits.pth',map_location='cpu',weights_only=True));depth=depth.eval().cuda()
    folder=next((root/'grounding-dino-tiny').glob('*'));proc=AutoProcessor.from_pretrained(folder,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(folder,local_files_only=True).eval().cuda()
    parser=importlib.import_module('sim.3dgs.grounder')
    def observe(rgb,text):
        phrase=parser._english_destination(text.lower().strip(' .')).strip(' .');expected=parser._tokens(phrase);blank=dict(phrase=phrase,box=None,score=None,angle_deg=None,range_m=None)
        if not phrase:return blank
        args=proc(images=Image.fromarray(rgb),text=phrase+'.',return_tensors='pt').to('cuda')
        with torch.inference_mode():
            pred=model(**args);result=proc.post_process_grounded_object_detection(pred,args.input_ids,threshold=.4,text_threshold=.3,target_sizes=[rgb.shape[:2]])[0]
        boxes=sorted(zip(result['scores'].tolist(),result['boxes'].tolist(),result['text_labels']),reverse=True)
        for score,box,label in boxes:
            tokens=parser._tokens(label)
            if not tokens or tokens!=expected[-len(tokens):]:continue
            x,y,X,Y=np.clip(box,0.,[320,240,320,240]);cx=(x+X)/2;cy=(y+Y)/2
            if X<=x+2 or Y<=y+2:continue
            with torch.inference_mode():metric=depth.infer_image(np.ascontiguousarray(rgb[:,:,::-1]),input_size=322)*.8651450872421265
            dx=(X-x)/4;dy=(Y-y)/4;pix=metric[max(0,int(cy-dy)):min(240,int(cy+dy)+1),max(0,int(cx-dx)):min(320,int(cx+dx)+1)];v=pix[np.isfinite(pix)&(pix>0.)];rng=float(np.median(v)) if len(v) else None
            return dict(phrase=phrase,box=box,label=label,score=score,angle_deg=float(np.rad2deg(np.arctan((cx-160)/277.1281292110204))),range_m=rng)
        return blank
    return observe

def calibrate():
    from PIL import Image
    observe=make_observer();source=DATA/'exp-vln-instructed-stop-data-20261124/samples.jsonl';samples=[json.loads(s) for s in source.read_text().splitlines()];rows=[]
    for s in samples:
        if s['split']!='train':continue
        views=[observe(np.asarray(Image.open(p).convert('RGB')),s['instruction']) for p in s['rgb_pair']];admit=all(v['range_m'] is not None and abs(v['angle_deg'])<=9 for v in views);score=max(v['range_m'] for v in views) if admit else None;rows.append(dict(group=s['group'],condition=s['condition'],label=s['label'],views=views,required_range_m=score))
    def ba(t):
        correct=lambda r:(r['required_range_m'] is not None and r['required_range_m']<=t)==bool(r['label'])
        return np.mean([np.mean([correct(r) for r in rows if r['label']==k]) for k in (0,1)])
    grid=[0.]+sorted({r['required_range_m'] for r in rows if r['required_range_m'] is not None});threshold=max(grid,key=lambda t:(ba(t),-t));record=dict(threshold_m=threshold,balanced_accuracy=ba(threshold),rows=rows,threshold_search=grid);(OUT/'calibration.json').write_text(json.dumps(record,indent=2)+'\n');print('source threshold',threshold,ba(threshold),len(rows),flush=True)

def service():
    import struct
    observe=make_observer();out=sys.stdout.buffer;out.write(b'RDY!');out.flush();stream=sys.stdin.buffer
    while True:
        head=stream.read(4)
        if not head:return
        n=struct.unpack('<I',head)[0];text=stream.read(n).decode();buf=stream.read(240*320*3)
        if len(buf)!=240*320*3:raise EOFError('truncated image')
        rec=observe(np.frombuffer(buf,np.uint8).reshape(240,320,3),text);out.write(json.dumps(rec).encode()+b'\n');out.flush()

if __name__=='__main__' and sys.argv[1] in ('calibrate','serve'):{'calibrate':calibrate,'serve':service}[sys.argv[1]]()

class VisualApproach:
    def __init__(self):
        import atexit,subprocess
        self.threshold=json.loads((OUT/'calibration.json').read_text())['threshold_m'];self.streak=0
        self.p=subprocess.Popen([str(DATA/'envs/vlm-venv/bin/python'),'-u',__file__,'serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE);atexit.register(self.p.terminate)
        if self.p.stdout.read(4)!=b'RDY!':raise RuntimeError('DINO/DA2 worker startup failed')
    def decide(self,rgb,text):
        import struct,time
        tick=time.perf_counter();bs=text.encode();self.p.stdin.write(struct.pack('<I',len(bs))+bs+np.ascontiguousarray(rgb).tobytes()); self.p.stdin.flush();obs=json.loads(self.p.stdout.readline());angle=obs['angle_deg'];dist=obs['range_m'];aligned=angle is not None and abs(angle)<=9;self.streak=self.streak+1 if aligned and dist is not None and dist<=self.threshold else 0
        stop=self.streak>=2;act=2 if angle is None or angle< -9 else 3 if angle>9 else 0 if stop else 1
        rec=dict(dino=obs,near_streak=self.streak,action_proposed=act,action_intervention='task_stop' if stop else None,stop_source='rgb_box_depth' if stop else None,model_stop_requested=stop)
        return act,stop,rec,(time.perf_counter()-tick)*1000

def evaluate():
    import subprocess,datetime
    former=json.loads((DATA/'exp-vln-near-start-839934-20261124/registration.json').read_text()); rc=1
    try:
        for policy in ('baseline','dino'):
            for row,old in zip(REG['rows'],former['arms']):
                argv=old['argv'].copy();name=policy+'_'+row['visible']+'_to_'+row['requested'];dest=OUT/name;argv[argv.index('--out')+1]=str(dest);argv[argv.index('--scene')+1]=row['fixture'];argv[argv.index('--seed')+1]='421'
                print(name,datetime.datetime.now().astimezone().isoformat(),flush=True); env=os.environ.copy()
                if policy=='dino':env['PYTHONPATH']=str(OUT)+os.pathsep+str(OUT/'runtime')+os.pathsep+env.get('PYTHONPATH','')
                runtime=OUT/'runtime' if policy=='dino' else Path(former['baseline_runtime']);subprocess.run(argv,cwd=runtime,env=env,check=True);report=json.loads((dest/'vln_eval.json').read_text());assert not report['failures'],report['failures'];assert len((dest/'episodes.jsonl').read_text().splitlines())==1
        rc=0
    finally:(OUT/'DONE').write_text('rc='+str(rc)+' '+datetime.datetime.now().astimezone().isoformat()+'\n')

if __name__=='__main__' and sys.argv[1]=='evaluate':evaluate()
