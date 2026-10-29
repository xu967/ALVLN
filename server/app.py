"""NavLab same-origin localhost API and static UI. Python standard-library only.

Default posture: read-only plus gated jobs. Nothing trains/flies by installing this.
Only operator-registered, preflight-verified commands can be started.
"""
from __future__ import annotations
import argparse
import csv
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit,parse_qs,unquote
from . import core as C

ALLOWED_ORIGINS=('http://127.0.0.1:', 'http://localhost:')
MAX_BODY=65536

class Handler(SimpleHTTPRequestHandler):
    server_version='NavLabLocal/8.0'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(C.ROOT/'frontend'),**kwargs)

    def log_message(self, format, *args):
        # Avoid printing URL query params in shared logs.
        import sys
        sys.stderr.write('[HTTP] '+format.split('"')[0]+'\n')

    def _host_ok(self):
        host=self.headers.get('Host','')
        port=str(self.server.server_port)
        return host in (f'127.0.0.1:{port}',f'localhost:{port}','127.0.0.1','localhost')

    def _origin_ok(self):
        origin=self.headers.get('Origin')
        if not origin:return True
        return origin in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')

    def _send(self, status, obj):
        body=C.jsonbytes(obj)
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'")
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def _error(self,status,code,detail=''):
        self._send(status,{'error':{'code':code,'detail':detail},'status':'blocked' if status==409 else 'error'})

    def _zip(self,data,name):
        self.send_response(200)
        self.send_header('Content-Type','application/zip')
        self.send_header('Content-Disposition',f'attachment; filename="{name}"')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

    def _api_get(self, route, params):
        cfg=C.registry()
        if route in ('/api/v1/health','/api/health'):
            return self._send(200,{'status':'ok','service':'navlab_api','version':'8.0','readiness':'management_only','physics_verified':False,'model_training_verified':False})
        if route=='/api/v1/capabilities':
            return self._send(200,{'service':'navlab_api','version':'8.0','tasks':['lani_v1','aerialvln_s_v1','gazebo_phys_v1'],
                'experimental':True,'ready_profiles':[p.get('id') for p in C.listed(cfg,'training_profiles') if p.get('ready') and not C.profile_check(p)],
                'simulation_auto_launch':False,'warnings':['UI has demo graphs in offline mode. Connecting does not create training results.','No default training or simulator executable is installed.']})
        if route=='/api/v1/datasets':return self._send(200,{'items':[x for x in C.listed(cfg,'datasets') if x.get('verified')]})
        if route=='/api/v1/checkpoints':
            task=params.get('task',[None])[0]
            return self._send(200,{'items':[x for x in C.listed(cfg,'checkpoints') if x.get('verified') and (task is None or x.get('task')==task)]})
        if route=='/api/v1/runs':return self._send(200,{'items':C.all_runs()})
        if route=='/api/v1/simulation/capabilities':
            task=params.get('task',[''])[0]
            profiles=[p for p in C.listed(cfg,'simulation_profiles') if p.get('task')==task]
            candidate=profiles[0] if profiles else None
            reasons=C.profile_check(candidate,sim=True) if candidate else ['SIM_PROFILE_NOT_REGISTERED']
            ready=not reasons
            return self._send(200,{'task':task,'ready':ready,'preflight':{'scene':True if ready else None,
                'model':True if ready else None,'data':True if ready else None,'rpc':True if ready else None},
                'auto_launch_supported':ready and bool(candidate.get('auto_launch_supported')) if candidate else False,
                'checkpoints':[x for x in C.listed(cfg,'checkpoints') if x.get('verified') and x.get('task')==task],
                'splits':[x for x in C.listed(cfg,'splits') if x.get('verified') and x.get('task')==task],
                'scenes':[x for x in C.listed(cfg,'scenes') if x.get('verified') and x.get('task')==task],
                'missing_reasons':reasons})
        bits=route.strip('/').split('/')
        if len(bits)>=4 and bits[:3]==['api','v1','runs']:
            jobid=C.identifier(bits[3]);folder=C.record_dir('train',jobid)
            if not folder.is_dir():return self._error(404,'RUN_NOT_FOUND')
            life=C.run_info(folder)
            if len(bits)==4:return self._send(200,{'run_id':jobid,**life})
            if len(bits)==5 and bits[4]=='metrics':return self._send(200,C.metrics(folder))
            if len(bits)==5 and bits[4]=='events':return self._sse(folder)
            if len(bits)==5 and bits[4]=='artifacts':return self._send(200,self._artifacts(folder))
            if len(bits)==5 and bits[4]=='audit-bundle':return self._zip(C.audit_zip(folder),jobid+'_ai_audit.zip')
        if len(bits)>=4 and bits[:3]==['api','v1','simulation-jobs']:
            jobid=C.identifier(bits[3]);folder=C.record_dir('sim',jobid)
            if not folder.is_dir():return self._error(404,'SIM_JOB_NOT_FOUND')
            life=C.run_info(folder)
            if len(bits)==4:return self._send(200,{'job_id':jobid,**life,'events':C.events(folder)[-100:],
                    'episode_index':None,'step':None,'action':None,'termination_reason':None,'observation_stream_status':'NOT_AVAILABLE'})
            if len(bits)==5 and bits[4]=='events':return self._sse(folder)
            if len(bits)==5 and bits[4]=='artifacts':return self._send(200,self._artifacts(folder))
            if len(bits)==5 and bits[4]=='audit-bundle':return self._zip(C.audit_zip(folder),jobid+'_sim_audit.zip')
        return self._error(404,'ENDPOINT_NOT_IMPLEMENTED','This is not a simulated result.')

    def _artifacts(self,folder):
        files=[]
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.stat().st_size<50*1024*1024:
                files.append({'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':C.sha(p)})
        return {'items':files,'generated_from':'real_filesystem'}

    def _sse(self,folder):
        try:after=int(self.headers.get('Last-Event-ID','0'))
        except ValueError:after=0
        rows=C.events(folder,after)
        payload=''.join('id: '+str(r['seq'])+'\nevent: '+('metric' if r.get('kind')=='metric' else 'status')+'\ndata: '+json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n\n' for r in rows)
        payload+=': reconnect for updates; no process state was changed\n\n'
        b=payload.encode('utf-8')
        self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
        self.send_header('Cache-Control','no-cache');self.send_header('Content-Length',str(len(b)))
        self.end_headers();self.wfile.write(b)

    def do_GET(self):
        if not self._host_ok():return self._error(421,'UNTRUSTED_HOST')
        path=urlsplit(self.path)
        if path.path.startswith('/api/'):
            try:return self._api_get(path.path,parse_qs(path.query))
            except ValueError as e:return self._error(400,'INVALID_REQUEST',str(e))
            except RuntimeError as e:return self._error(503,'REGISTRY_UNAVAILABLE',str(e))
            except (OSError,KeyError,TypeError,csv.Error) as e:return self._error(500,'READ_FAILED',type(e).__name__)
        route='/index.html' if path.path=='/' else path.path
        if path.path=='/':self.path='/index.html'
        # Static UI only; never expose config/, artifacts/, or server source over HTTP.
        if route not in ('/index.html','/app.js','/styles.css') and not re.fullmatch(r'/preview_[A-Za-z0-9_-]+\.png',route):
            return self._error(404,'FILE_NOT_PUBLIC')
        super().do_GET()

    def do_POST(self):
        if not self._host_ok():return self._error(421,'UNTRUSTED_HOST')
        if not self._origin_ok():return self._error(403,'CROSS_ORIGIN_DENIED')
        if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
            return self._error(415,'JSON_REQUIRED')
        try:
            n=int(self.headers.get('Content-Length','-1'))
            if n<0 or n>MAX_BODY:return self._error(413,'BODY_TOO_LARGE')
            data=json.loads(self.rfile.read(n),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
            if not isinstance(data,dict):return self._error(400,'OBJECT_REQUIRED')
        except (ValueError,UnicodeDecodeError,RecursionError):return self._error(400,'MALFORMED_JSON')
        route=urlsplit(self.path).path
        try:
            if route=='/api/v1/runs':
                job,life=C.launch('train',data)
                return self._send(202,{'run_id':job,'status':life['status']})
            if route=='/api/v1/simulation-jobs':
                job,life=C.launch('sim',data)
                return self._send(202,{'job_id':job,'status':life['status']})
            parts=route.strip('/').split('/')
            if len(parts)==5 and parts[:3] in (['api','v1','runs'],['api','v1','simulation-jobs']) and parts[4]=='stop':
                result=C.stop_job('train' if parts[2]=='runs' else 'sim',C.identifier(parts[3]))
                return self._send(200,result)
        except ValueError as e:return self._error(400,'INVALID_REQUEST',str(e))
        except RuntimeError as e:return self._error(409,'PRECONDITION_FAILED',str(e))
        except FileNotFoundError:return self._error(404,'JOB_NOT_FOUND')
        except (OSError,KeyError,TypeError) as e:return self._error(500,'JOB_START_FAILED',type(e).__name__)
        return self._error(501,'FEATURE_NOT_IMPLEMENTED','No arbitrary inference, evaluation or device operation is permitted.')

    def do_PUT(self):return self._error(405,'METHOD_NOT_ALLOWED')
    def do_DELETE(self):return self._error(405,'METHOD_NOT_ALLOWED')
    def do_OPTIONS(self):return self._error(405,'METHOD_NOT_ALLOWED')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if args.host not in ('127.0.0.1','localhost'):
        parser.error('SECURITY: UI service only listens on loopback. Use SSH -L or an authenticated reverse proxy.')
    C.registry();C.recover_interrupted()
    httpd=ThreadingHTTPServer((args.host,args.port),Handler)
    httpd.daemon_threads=True
    print(f'NavLab Studio API/UI at http://{args.host}:{args.port} [MANAGEMENT_ONLY; NO TRAINING/SIM CLAIMS]',flush=True)
    try:httpd.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:httpd.server_close()

if __name__=='__main__':main()
