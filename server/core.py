"""Trusted job registry, persistent evidence and subprocess lifecycle.

No simulator / trainer is bundled as an executable. Only administrators can edit
config/registry.json; the UI can only select registered IDs, never shell commands.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
RUNS = ART / 'runs'
SIMS = ART / 'simulations'
REGISTRY = ROOT / 'config' / 'registry.json'
ID_RE = re.compile(r'[A-Za-z0-9_-]{1,70}\Z', re.ASCII)
_lock = threading.RLock()
_processes: dict[str, subprocess.Popen] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')


def identifier(s: Any) -> str:
    if not isinstance(s,str) or not ID_RE.fullmatch(s):
        raise ValueError('INVALID_ID: IDs must be ASCII [A-Za-z0-9_-], max 70')
    return s


def jsonbytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def save(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_bytes(jsonbytes(obj))
    temp.replace(path)


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text('utf-8'))
    except (OSError, json.JSONDecodeError, ValueError, RecursionError):
        return default


def registry() -> dict:
    cfg = read(REGISTRY,{})
    if not isinstance(cfg,dict) or cfg.get('schema')!='navlab.registry.v1':
        raise RuntimeError('REGISTRY_INVALID: schema must be navlab.registry.v1')
    return cfg


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def entry_path(p: str) -> Path:
    x=Path(p)
    return x if x.is_absolute() else ROOT / x


def listed(cfg:dict,key:str) -> list:
    v=cfg.get(key,[])
    return v if isinstance(v,list) else []


def find_item(cfg:dict,key:str,id_value:str):
    for x in listed(cfg,key):
        if isinstance(x,dict) and x.get('id')==id_value:
            return x
    return None


def profile_check(profile:dict, *, sim=False) -> list[str]:
    reasons=[]
    if not profile.get('ready',False): reasons.append('PROFILE_NOT_VERIFIED')
    path=entry_path(str(profile.get('executable','')))
    if not path.is_file() or not os.access(path,os.X_OK):
        reasons.append('EXECUTABLE_NOT_READY')
    for p in profile.get('requirements',[]):
        if not entry_path(str(p)).exists(): reasons.append('MISSING_RESOURCE:'+str(p))
    if sim:
        # Crucial: a generic Gazebo process is NOT a closed-loop evaluator.
        if not profile.get('closed_loop_verified',False): reasons.append('CLOSED_LOOP_NOT_VERIFIED')
        if not profile.get('scene_supervisor_verified',False): reasons.append('SCENE_SUPERVISOR_NOT_VERIFIED')
    return reasons


def events_file(folder:Path):
    return folder/'ui_service'/'events.jsonl'


def add_event(folder:Path, kind:str, message:str, **extra):
    p=events_file(folder);p.parent.mkdir(parents=True,exist_ok=True)
    with _lock:
        seq=0
        if p.is_file():
            for line in p.read_text('utf-8', errors='replace').splitlines():
                try: seq=max(seq,int(json.loads(line).get('seq',0)))
                except (ValueError,TypeError,AttributeError,RecursionError): pass
        data={'schema_version':'navlab_event_v1','seq':seq+1,'kind':kind,'type':kind,
              'time':now(),'timestamp':now(),'level': 'ERROR' if kind=='error' else 'INFO',
              'source':'job_supervisor','message':message,**extra}
        with p.open('ab') as f:
            line=(json.dumps(data,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
            f.write(line);f.flush()
    return data


def events(folder:Path, after=0):
    p=events_file(folder)
    if not p.is_file(): return []
    result=[]
    with p.open('rb') as f:
        for raw in f:
            if len(raw)>262144:continue
            try:
                row=json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite')))
                if isinstance(row,dict) and int(row.get('seq',0))>after:result.append(row)
            except (ValueError,TypeError,RecursionError,OverflowError):continue
    return result[-3000:]


def record_dir(category:str, ident:str) -> Path:
    identifier(ident)
    if category=='train': return RUNS/ident
    if category=='sim': return SIMS/ident
    raise ValueError('INVALID_CATEGORY')


def run_info(folder:Path):
    x=read(folder/'ui_service'/'lifecycle.json',{})
    return x if isinstance(x,dict) else {}


def all_runs():
    if not RUNS.is_dir(): return []
    xs=[]
    for p in RUNS.iterdir():
        if not p.is_dir() or not ID_RE.fullmatch(p.name) or p.is_symlink():continue
        x=run_info(p)
        xs.append({'run_id':p.name, 'status':x.get('status','untracked'),
                   'architecture':x.get('architecture','—'), 'task':x.get('task','—'),
                   'epochs':x.get('epochs',0),'epoch':x.get('epoch',0),
                   'created_at':x.get('created_at','')})
    return sorted(xs,key=lambda x:x['created_at'],reverse=True)


def metrics(folder:Path):
    # Accept actual legacy results.csv, never derive navigation SR from accuracy.
    p=folder/'results.csv'
    if not p.is_file():return {'rows':[],'source':None}
    rows=[]
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        for rec in reader:
            def number(*keys):
                for key in keys:
                    if key in rec and rec[key] not in ('',None):
                        try:
                            v=float(rec[key]);return v if v==v and abs(v)!=float('inf') else None
                        except (TypeError,ValueError):pass
                return None
            ep=number('epoch')
            if ep is None:continue
            rows.append({'epoch':int(ep),'train_ce':number('train_ce','train/loss'),
                         'dev_ce':number('dev_ce','val/loss'),
                         'dev_accuracy':number('dev_accuracy','val/accuracy'),
                         'dev_stop_recall':number('dev_stop_recall','val/stop_recall')})
            if len(rows)>=100000:break
    return {'rows':rows,'source':'results.csv','navigation_sr':None,
            'note':'Accuracy is teacher-state classification, NOT closed-loop success.'}


def _resume_group(proc:subprocess.Popen):
    try:os.killpg(proc.pid,signal.SIGCONT)
    except ProcessLookupError:pass


def _pause_group(proc:subprocess.Popen):
    os.killpg(proc.pid,signal.SIGSTOP)
    try:
        deadline=time.monotonic()+5
        while proc.poll() is None:
            try:stat=Path(f'/proc/{proc.pid}/stat').read_text('utf-8')
            except FileNotFoundError:
                if proc.poll() is not None:return None
                raise
            if stat.rsplit(')',1)[1].split()[0] in ('T','t'):return proc
            if time.monotonic()>=deadline:raise RuntimeError('SNAPSHOT_PAUSE_TIMEOUT')
            time.sleep(.01)
        return None
    except BaseException:
        _resume_group(proc)
        raise


def audit_zip(folder:Path) -> bytes:
    if not folder.is_dir():raise FileNotFoundError('RUN_NOT_FOUND')
    snapshots=[]
    with _lock:
        paused=None
        try:
            proc=_processes.get(folder.name)
            if proc and proc.poll() is None:
                try:paused=_pause_group(proc)
                except ProcessLookupError:pass
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or p.is_symlink() or p.stat().st_size>20*1024*1024:continue
                rel=p.relative_to(folder).as_posix()
                if rel.endswith(('.pt','.pth','.ckpt','.cvimodel','.onnx')):continue
                data=p.read_bytes()
                if len(data)<=20*1024*1024:snapshots.append((rel,data))
        finally:
            if paused:_resume_group(paused)
    f=io.BytesIO(); meta=[]
    # Compression happens after writers resume; every file comes from the quiesced snapshot.
    with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        for rel,data in snapshots:
            z.writestr(rel,data)
            meta.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        z.writestr('AI_REVIEW_START_HERE.md', '# AI Evidence Bundle\n\nThis bundle only proves which artifacts were produced. No simulator, training or device performance should be inferred from missing files.\n')
        z.writestr('audit_manifest.json',jsonbytes({'created_at':now(),'status':'evidence_only','files':meta}))
    return f.getvalue()


def _worker(proc:subprocess.Popen,folder:Path):
    """Propagate actual CSV epochs to browser events, then persist exit status.

    No synthetic metric values are generated. A real trainer must write results.csv.
    """
    last_epoch=0
    def sync_csv():
        nonlocal last_epoch
        try:
            rows=metrics(folder)['rows']
            for row in rows:
                ep=row['epoch']
                if ep<=last_epoch:continue
                add_event(folder,'metric','epoch metrics written by trainer',epoch=ep,metrics=row)
                last_epoch=ep
                life=run_info(folder);life['epoch']=ep
                save(folder/'ui_service'/'lifecycle.json',life)
        except (OSError,ValueError,TypeError,csv.Error):
            # Preserve raw file; malformed partial rows will be read on the next poll.
            pass
    try:
        while proc.poll() is None:
            with _lock:sync_csv()
            time.sleep(1)
        with _lock:sync_csv()
    finally:
        rc=proc.wait()
        with _lock:
            try:
                life=run_info(folder)
                stopped=life.get('stop_requested') is True or life.get('status')=='stopping'
                status='stopped' if stopped else ('finished' if rc==0 else 'failed')
                life.update({'status':status,'ended_at':now(),'exit_code':rc})
                save(folder/'ui_service'/'lifecycle.json',life)
                add_event(folder,'status',f'process {status}' if stopped else f'process exited {rc}',
                          status=status,exit_code=rc)
            finally:
                _processes.pop(folder.name,None)


def _validate_payload(category:str,payload:dict):
    if category not in ('train','sim') or not isinstance(payload,dict):
        raise ValueError('JOB_PAYLOAD_INVALID')
    if category=='train':
        for field in ('task','architecture','name','train_cache_id','validation_cache_id'):
            if not isinstance(payload.get(field),str) or not payload[field]:
                raise ValueError(field.upper()+'_INVALID')
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}',payload['name'],re.ASCII):
            raise ValueError('NAME_INVALID')
        for field in ('task','architecture','train_cache_id','validation_cache_id'):
            identifier(payload[field])
        for field,minimum,maximum in (('epochs',1,5000),('sequence_chunk_size',1,None),('seed',0,None)):
            value=payload.get(field)
            if type(value) is not int or value<minimum or (maximum is not None and value>maximum):
                raise ValueError(field.upper()+'_INVALID')
        lr=payload.get('lr')
        try:
            valid_lr=not isinstance(lr,bool) and isinstance(lr,(int,float)) and math.isfinite(lr) and lr>0
        except OverflowError:
            valid_lr=False
        if not valid_lr:raise ValueError('LR_INVALID')
    else:
        if not isinstance(payload.get('task'),str) or not payload['task']:
            raise ValueError('TASK_INVALID')
        identifier(payload['task'])
        for field in ('checkpoint_id','split_id','scene_id'):
            identifier(payload.get(field,''))
        for field in ('auto_launch','record_observations','audit_artifacts'):
            if field in payload and type(payload[field]) is not bool:
                raise ValueError(field.upper()+'_INVALID')
        if payload.get('mode') not in ('diagnostic','official_split'):
            raise ValueError('MODE_INVALID')


def launch(category:str, payload:dict):
    """Serializes task launch; single-server, single-GPU default concurrency."""
    with _lock:
        if any(p.poll() is None for p in _processes.values()):
            raise RuntimeError('GPU_OR_SIM_RESOURCE_BUSY')
        return _launch_locked(category,payload)

def _launch_locked(category:str, payload:dict):
    """Start only a verified, administrator-registered executable profile."""
    _validate_payload(category,payload)
    cfg=registry(); key='training_profiles' if category=='train' else 'simulation_profiles'
    requested=payload.get('architecture') if category=='train' else payload.get('task')
    profiles=[p for p in listed(cfg,key) if isinstance(p,dict) and
              (p.get('architecture')==requested if category=='train' else p.get('task')==requested)]
    if not profiles:raise RuntimeError('PROFILE_NOT_REGISTERED')
    profile=profiles[0]
    if profile.get('task')!=payload.get('task'): raise ValueError('TASK_PROFILE_MISMATCH')
    reasons=profile_check(profile,sim=category=='sim')
    if reasons:raise RuntimeError('; '.join(reasons))
    if category=='train':
        for field,regkey in (('train_cache_id','datasets'),('validation_cache_id','datasets')):
            value=identifier(payload.get(field,''))
            item=find_item(cfg,regkey,value)
            if not item or not item.get('verified') or item.get('task')!=payload['task']:
                raise RuntimeError('DATASET_NOT_VERIFIED:'+field)
        if payload['train_cache_id']==payload['validation_cache_id']:
            raise RuntimeError('DATASET_SPLIT_LEAK')
    else:
        for field,key in (('checkpoint_id','checkpoints'),('split_id','splits'),('scene_id','scenes')):
            item=find_item(cfg,key,identifier(payload.get(field,'')))
            if not item or not item.get('verified') or item.get('task')!=payload['task']:
                raise RuntimeError('RESOURCE_NOT_VERIFIED:'+field)
        if payload.get('auto_launch') and not profile.get('auto_launch_supported'):
            raise RuntimeError('AUTO_LAUNCH_UNAVAILABLE')
        if payload['mode']=='official_split' and not profile.get('official_protocol_verified',False):
            raise RuntimeError('OFFICIAL_PROTOCOL_NOT_VERIFIED')
    # Job ID generated server-side (no arbitrary client-side file paths).
    job_id=('run_' if category=='train' else 'sim_')+uuid.uuid4().hex[:16]
    folder=record_dir(category,job_id)
    folder.mkdir(parents=True,exist_ok=False)
    if category=='train':
        config_file=folder/'config.json'; save(config_file,payload)
        tokens={'run_dir':str(folder),'config_file':str(config_file)}
    else:
        save(folder/'simulation_request.json',payload)
        tokens={'sim_dir':str(folder)}
    command=[str(entry_path(str(profile['executable'])))]
    for part in profile.get('argv',[]):
        if part in ('{run_dir}','{config_file}','{sim_dir}'):
            if part[1:-1] not in tokens: raise RuntimeError('PROFILE_TEMPLATE_INVALID')
            command.append(tokens[part[1:-1]])
        elif isinstance(part,str) and '{' not in part and '}' not in part:
            command.append(part)
        else:raise RuntimeError('PROFILE_TEMPLATE_INVALID')
    life={'status':'preflight','created_at':now(),'task':payload['task'],
          'architecture':payload.get('architecture','—'),'epochs':payload.get('epochs',0),
          'profile_id':profile['id'],'category':category,'command':command,'evidence':'process'}
    service=folder/'ui_service';service.mkdir(parents=True,exist_ok=True)
    save(service/'lifecycle.json',life)
    add_event(folder,'status','verified profile prepared',status='preflight')
    out=(service/'stdout.log').open('ab');err=(service/'stderr.log').open('ab')
    try:
        proc=subprocess.Popen(command,cwd=str(ROOT),stdout=out,stderr=err,
                              start_new_session=True,stdin=subprocess.DEVNULL)
    except (OSError,ValueError) as exc:
        life.update(status='failed',error_code='PROCESS_START_FAILED',detail=str(exc))
        save(service/'lifecycle.json',life)
        add_event(folder,'error','PROCESS_START_FAILED',detail=str(exc))
        raise RuntimeError('PROCESS_START_FAILED') from exc
    finally:out.close();err.close()
    with _lock:
        _processes[job_id]=proc
        life.update({'status':'running','pid':proc.pid,'started_at':now()})
        save(service/'lifecycle.json',life)
        add_event(folder,'status','process running',status='running')
    threading.Thread(target=_worker,args=(proc,folder),daemon=True).start()
    return (job_id,life)


def stop_job(category,job_id):
    folder=record_dir(category,job_id)
    if not folder.is_dir():raise FileNotFoundError('JOB_NOT_FOUND')
    with _lock:
        proc=_processes.get(job_id)
        if not proc or proc.poll() is not None:raise RuntimeError('JOB_NOT_ACTIVE')
        # Signal process group owned by this server; never pkill simulator by pattern.
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError as exc:raise RuntimeError('JOB_NOT_ACTIVE') from exc
        # Worker finalization needs the same lock, so it cannot miss this persisted stop intent.
        life=run_info(folder)
        life.update({'status':'stopping','stop_requested':True,'stop_requested_at':now()})
        save(folder/'ui_service'/'lifecycle.json',life)
        add_event(folder,'status','stop requested',status='stopping')
    return {'status':'stopping'}


def recover_interrupted():
    for base in (RUNS,SIMS):
        if not base.exists():continue
        for d in base.iterdir():
            if not d.is_dir() or d.is_symlink():continue
            life=run_info(d)
            if life.get('status') in ('running','preflight','stopping'):
                life.update(status='interrupted',error_code='SERVICE_RESTART_UNCERTAIN',updated_at=now())
                save(d/'ui_service'/'lifecycle.json',life)
                add_event(d,'error','Service restarted; external process state unknown. No remote kill was attempted.',status='interrupted')
