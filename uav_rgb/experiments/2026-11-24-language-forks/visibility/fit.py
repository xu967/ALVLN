import collections, hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from scipy.optimize import minimize
from transformers import CLIPProcessor,CLIPModel
out=Path(__file__).resolve().parent;base=out.parent
def rd(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
cfg=rd(out/"registration.json");review=rd(out/"qa_review.json")
def canon(x):return {"body pillow":"pillow","multi person sofa":"sofa","teatable":"coffee table","display cabinet":"cabinet"}.get(x,x)
pos=rows(out/"positive_boxes.jsonl");outside={r["sha256"]:r for r in rows(out/"outside_classes.jsonl")}
neg={};categories={s:set() for s in cfg["train"]+cfg["development"]}
for p in pos:categories[p["scene"]].add(p["category"])
for s in categories:
 labels=rd(next((base/"pubdata/sage3d/interiorgs_labels").glob("*_"+s+"/labels.json")))
 categories[s]={r["label"].lower().replace("_"," ") for r in labels if r.get("label","").lower().replace("_"," ") in cfg["vocabulary"] and r.get("bounding_box")}
for k,v in outside.items():
 c=categories[v["scene"]];absent=set(v["absent_catalog_classes"])
 neg[k]={canon(w) for w in c if all(q in absent for q in c if canon(q)==canon(w))}
seen={c for s in cfg["train"] for c in review[s].values() if c}
anchors=[]
for s in cfg["train"]:
 for i,c in review[s].items():
  if not c: continue
  options=[r for r in pos if r["scene"]==s and r["ins_id"]==i and seen&neg[r["sha256"]]-{c}]
  groups=[];used=collections.defaultdict(list)
  for k in range(3):
   if not options:break
   r=min(options,key=lambda r:(len(used[r["route"]]),hashlib.sha256((r["sha256"]+i).encode()).hexdigest()));groups.append(r);used[r["route"]].append(r["step"])
   options=[x for x in options if x['route']!=r['route'] or abs(x['step']-r['step'])>=4]
  anchors+=[dict(**r,query=c) for r in groups]
global_used=set();pairs=[]
for c in sorted(seen):
 for d in sorted(seen):
  if c>=d: continue
  left=[r for r in anchors if r["query"]==c and d in neg[r["sha256"]]]
  right=[r for r in anchors if r["query"]==d and c in neg[r["sha256"]]]
  for x in left:
   if x["sha256"] in global_used:continue
   y=next((r for r in right if r['sha256'] not in global_used and r['sha256']!=x['sha256']),None)
   if y:
    pairs.append((x,y));global_used.update([x["sha256"],y["sha256"]])
import sys
strain=sys.argv[1:]==["--all-three"]
if strain:
 qa={r["scene"]+"/"+r["ins_id"]:r for r in rd(out/"qa_rows.json") if r["scene"] in cfg["train"]}
 pairs=[]
 for k,l in rd(out/"manual_train_pairs.json"):
  a,b=qa[k],qa[l]
  pairs.append(tuple(dict(**r,query=review[r["scene"]][r["ins_id"]]) for r in (a,b)))
 assert len({r["scene"] for pair in pairs for r in pair})==3
training=[]
for i,(x,y) in enumerate(pairs):
 for a,b in ((x,y),(y,x)):
  for query,label in [(a["query"],1),(b["query"],0)]:
   training.append(dict(set="train",pair=i,path=a["rgb"],query=query,label=label,scene=a["scene"],route=a["route"],ins_id=a["ins_id"],sha256=a["sha256"]))
assert training
devreg=rd(base/"exp-vln-region-probe-20261123/nine_regions/registration.json")
names={"door":"door","handle":"door handle","large_canvas":"decorative painting","toy":"toy bear","sofa":"sofa","far_door":"door","laptop":"laptop","yellow_pillow":"pillow","mannequin":"mannequin","silver_cup":"trophy cup"}
dev=[]
for f in devreg["frames"]:
 absent=next(a for a in f["annotations"] if a["key"]=="absent");obj=absent["text"].lower().split('the ')[-1].strip('.')
 for a in f["annotations"]:
  if not a.get("box_px") or a["key"] not in names:continue
  group=f["id"]+"/"+a["key"]
  for query,label in ((names[a['key']],1),(obj,0)):
   dev.append(dict(set="historical",pair=group,path=f["path"],query=query,label=label,frame=f["id"],key=a["key"],sha256=f["sha256"],scene=Path(f["path"]).parent.parent.name))
mir=rd(base/"exp-vln-new-scenes-20261124/acquisition_audit/mirror_registration.json")["cases"]
mrows=[dict(set="mirrors",pair=r["route"],frame=r["route"],key=r["target"],path=r["frame"],mirror=r["mirror"],query={"wall_lamp_1":"wall lamp","cabinet_1":"cabinet"}[r["target"]],label=1,sha256=r["image_sha256"]) for r in mir]
# New reciprocal categories evaluated on the already registered, manually annotated historical images. No learned scores used for absence review.
balanced=rd(out/"dev_balanced.json");brows=[]
byframe={f["id"]:f for f in devreg['frames']}
for j,(fa,fb,ca,cb) in enumerate(balanced["pairs"]):
 for fi,pc,nc in [(fa,ca,cb),(fb,cb,ca)]:
  f=byframe[fi]
  for q,v in [(pc,1),(nc,0)]:brows.append(dict(set="balanced_dev",pair=j,path=f["path"],sha256=f["sha256"],frame=fi,key=q,query=q,label=v))
allrows=training+dev+brows+mrows
(out/("all_three_manifest.json" if strain else "fit_manifest.json")).write_text(json.dumps(allrows,indent=2)+"\n")
print("fit sizes",len(training),len(pairs),len({r["query"] for r in training}),len({r["path"] for r in training}),len(dev),len(brows),flush=True)
assert not ({r['sha256'] for r in training}&{r['sha256'] for r in dev+brows+mrows})
assert os.environ['CUDA_VISIBLE_DEVICES']==''
model_cfg=rd(base/"exp-vln-stop-clip-20261124/registration.json");ckpt=model_cfg["model_path"];torch.set_num_threads(6)
model=CLIPModel.from_pretrained(ckpt,local_files_only=True).eval();proc=CLIPProcessor.from_pretrained(ckpt,local_files_only=True)
tokens=sorted({r["query"] for r in allrows});txt=proc.tokenizer(["a photo of a "+s+"." for s in tokens],padding=True,return_tensors="pt")
with torch.inference_mode():
 text=model.text_projection(model.text_model(**txt).pooler_output);text=torch.nn.functional.normalize(text).numpy()
cache={}
for r in allrows:
 key=(r["path"],r.get("mirror",False))
 if key not in cache:
  im=Image.open(key[0]).convert("RGB")
  if key[1]:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
  w,h=im.size;side=round(min(w,h)*.72);views=[im]+[im.crop((x,y,x+side,y+side)) for y in (0,h-side) for x in (0,w-side)]
  pad=tuple(round(255*v) for v in proc.image_processor.image_mean);ims=[]
  for v in views:
   sq=Image.new("RGB",(max(v.size),)*2,pad);sq.paste(v,((sq.width-v.width)//2,(sq.height-v.height)//2));ims.append(sq)
  with torch.inference_mode():
   images=model.visual_projection(model.vision_model(pixel_values=proc.image_processor(ims,return_tensors="pt").pixel_values).pooler_output)
   cache[key]=torch.nn.functional.normalize(images).numpy()@text.T
 scores=cache[key][:,tokens.index(r['query'])];r["full"]=float(scores[0]);r["tile"]=float(scores[1:].max())
def fit(fields):
 X=np.array([[r[c] for c in fields] for r in training]);Y=np.array([r["label"] for r in training]);mn=X.mean(0);sd=np.maximum(X.std(0),1e-8);Z=(X-mn)/sd
 def loss(p):
  z=Z@p[:-1]+p[-1];v=expit(z)-Y
  return np.logaddexp(0,z).sum()-(Y*z).sum()+p[:-1]@p[:-1]/2,np.r_[Z.T@v+p[:-1],v.sum()]
 res=minimize(loss,np.zeros(len(fields)+1),jac=True,method="L-BFGS-B");assert res.success
 for r in allrows:r["_".join(fields)]=float(expit(((np.array([r[c] for c in fields])-mn)/sd)@res.x[:-1]+res.x[-1]))
 return dict(mean=mn.tolist(),sd=sd.tolist(),weights=res.x.tolist())
for r in allrows:r["full_only"]=r["full"]
weights1=fit(["full_only"]);weights2=fit(["full","tile"])
def score(ss):
 pairs0=collections.defaultdict(dict)
 for r in ss:pairs0[(r["pair"],r["path"])][r["label"]]=r
 o={}
 for field in ["full_only","full_tile"]:
  counts=collections.Counter()
  for x in ss:counts[("tp" if x[field]>=.5 else "fn") if x["label"] else ("fp" if x[field]>=.5 else "tn")]+=1
  paired=[vals[1][field]>vals[0][field] for vals in pairs0.values() if 0 in vals and 1 in vals]
  o[field]=dict(counts=counts,pair_correct=sum(paired),pairs=len(paired))
 return o
result=dict(train_pairs=len(pairs),train_routes=len({(r["scene"],r["route"]) for r in training}),train_instances=len({(r["scene"],r["ins_id"]) for r in training}),train_cats=sorted({r["query"] for r in training}),train_scenes=sorted({r["scene"] for r in training}),positive_count_by_train_scene=collections.Counter(r["scene"] for r in training if r["label"]),weights=dict(full=weights1,tiles=weights2),scores={name:score(ss) for name,ss in [("training",training),("historical",dev),("historical_whole",[r for r in dev if r["key"]!="handle"]),("reciprocal",brows),("mirrors",mrows)]},rows=allrows)
(out/("all_three_results.json" if strain else "fit_results.json")).write_text(json.dumps(result,indent=2)+"\n")
print({k:v for k,v in result.items() if k!="rows"},flush=True)
