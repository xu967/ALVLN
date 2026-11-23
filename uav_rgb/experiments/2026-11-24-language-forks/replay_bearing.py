"""CPU 固定权重与真实双帧，逐窗只替换当前方位，保持已发生动作前缀。"""
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

out = Path(__file__).resolve().parent
reg = json.loads((out/"registration.json").read_text())
ckpt = Path(next(p for p in reg["files"] if p.endswith("ckpt_best.bin")))
buf = ckpt.read_bytes()
assert hashlib.sha256(buf).hexdigest() == reg["files"][str(ckpt)]
magic, ver, pc, h, w, ch, nc, _, _, nw = struct.unpack_from("<8sII4i3I", buf)
assert (magic, ver, pc, ch, nc) == (b"UAVRCK01",9,12,6,5)
offset = 44 + 4*nw + 4 + 4 + 113 + 4 + 8 + 36
weights = []
for _ in range(pc):
    nd, = struct.unpack_from("<I",buf,offset)
    shape = struct.unpack_from("<"+"q"*nd, buf, offset+4)
    offset += 4+8*nd
    arr = np.frombuffer(buf, "<f4", np.prod(shape), offset).reshape(shape).copy()
    weights.append(torch.from_numpy(arr))
    offset += arr.nbytes
assert len(buf) == offset + 8*sum(t.numel() for t in weights)
torch.set_num_threads(2)


def logs(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def feature(frame, prev):
    t = torch.from_numpy(np.concatenate([frame,prev],axis=2)).permute(2,0,1)[None].float().div_(255)
    for i in [0,2,4]:
        t = F.max_pool2d(F.relu(F.conv2d(t,weights[i],weights[i+1],padding=1)),2)
    return t


def propose(t, prev, bb, ng):
    cond = torch.zeros(1,15)
    cond[0,[prev,4+bb,13+ng]] = 1
    gamma,beta = F.linear(cond,weights[8],weights[9]).chunk(2,dim=1)
    x = t * (1 + gamma[:,:,None,None]) + beta[:,:,None,None]
    H = x.shape[2]; K=min(3,H)
    flat = torch.stack([x[:,:,i*H//K:(i+1)*H//K].mean(2) for i in range(K)],dim=2).flatten(1)
    return int(F.linear(flat,weights[6],weights[7])[0,:4].argmax())


def run():
    counts = Counter(); rows=[]
    for arm in ["default_correct","default_swap","default_zero"]:
        for ep in logs(out/arm/"episodes.jsonl"):
            d=Path(ep["dir"]); action=logs(d/"actions.jsonl")
            gg=d/"grounding_trace.jsonl"
            fresh={g["step"]:g for g in logs(gg)} if gg.exists() else {}
            prev_img=None; prev_act=0; prev_bearing=8
            for i,a in enumerate(action):
                rgb=np.array(Image.open(d/a["frame"]).convert("RGB").resize((w,h),Image.Resampling.BILINEAR))
                feat=feature(rgb,rgb if prev_img is None else prev_img)
                baseline=propose(feat,prev_act,a["bearing_bin"],a["near_goal"])
                assert baseline==a["action_proposed"],(arm,ep["route_name"],i,baseline,a["action_proposed"])
                cause="fresh" if i in fresh else ("empty" if not fresh else "held")
                if i in fresh:prev_bearing=fresh[i]["raw_bearing_bin"]
                if i not in fresh and fresh and a["bearing_bin"]==0 and (prev_bearing,prev_act) in ((1,2),(7,3)):
                    cause="turn_inferred_center"
                held=propose(feat,prev_act,prev_bearing,a["near_goal"])
                search=propose(feat,prev_act,8,a["near_goal"])
                rows.append(dict(arm=arm,route=ep["route_name"],step=i,cause=cause,last_fresh=prev_bearing,
                            bearing=a["bearing_bin"],mode=a.get("grounding_mode"),prev_action=prev_act,
                            base=baseline,executed=a["action_executed"],held_bearing_proposed=held,
                            search_proposed=search,collision=a["collision"],rgb_sha256=hashlib.sha256((d/a["frame"]).read_bytes()).hexdigest()))
                counts[(cause,baseline,held,search)]+=1
                prev_img,prev_act=rgb,a["action_executed"]
    result={"sha256": {"checkpoint":hashlib.sha256(buf).hexdigest(),"cpu_code":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "rows":rows,"counts":[[list(k),v] for k,v in counts.items()],
            "scope":"Single-step model interventions on observed images and previous executed actions. CPU proposes before avoidance. Physical trajectory after any changed action is unobserved; no causal collision result. Empty instruction search and time series have distinct action/state distributions."}
    (out/"replay_bearing.json").write_text(json.dumps(result,indent=2)+"\n")
    print(counts, len(rows))


if __name__ == "__main__":
    run()
