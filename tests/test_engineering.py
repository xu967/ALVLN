import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch
from server import core as C
from server.app import Handler,ThreadingHTTPServer
from tools.scene_generator import create,generate,route

class TestScene(unittest.TestCase):
    def test_deterministic_sha_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a=generate(42,'S1',Path(d));p=Path(d)/a['sdf']
            self.assertEqual(a['sdf_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            b=generate(42,'S1',Path(d));self.assertEqual(a['sdf_sha256'],b['sdf_sha256'])
    def test_blocked_shortcut(self):
        s=create(42,'S1'); self.assertTrue(s['oracle_path'])
        self.assertTrue(any(abs(x)<1 for x,_,_ in s['obstacles']))
        self.assertTrue(any('pillar' in lm['id'] for lm in s['landmarks']))
    def test_difficulty_is_different(self):
        self.assertNotEqual(create(42,'S0')['obstacles'],create(42,'S1')['obstacles'])
        self.assertNotEqual(create(42,'S1')['obstacles'],create(42,'S2')['obstacles'])
    def test_impossible_path(self):
        self.assertEqual([],route((0,0),(3,0),[(0,0,2)]))
    def test_semantic_landmarks_not_on_oracle_route(self):
        import math
        for difficulty in ('S0','S1','S2'):
            for seed in range(12):
                s=create(seed,difficulty)
                for lm in s['landmarks']:
                    clearance=min(math.hypot(lm['x']-x,lm['y']-y) for x,y,_ in s['oracle_path'])
                    self.assertGreater(clearance,.44, (difficulty,seed,lm['id']))
    def test_no_physics_claim(self):
        with tempfile.TemporaryDirectory() as d:
            a=generate(8,'S2',Path(d));self.assertFalse(a['physics_quadrotor']);self.assertFalse(a['closed_loop_episode'])

class TestCore(unittest.TestCase):
    def test_id_prevents_traversal(self):
        for x in ('../etc','跑次','a.b','a/b',' ', 'a'*71):
            with self.assertRaises(ValueError): C.identifier(x)
        self.assertEqual('run_123',C.identifier('run_123'))
    def test_registry_is_not_ready(self):
        c=C.registry();self.assertEqual('navlab.registry.v1',c['schema'])
        self.assertFalse(c['training_profiles']);self.assertFalse(c['simulation_profiles'])
    def test_no_fake_results_csv(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(C.metrics(Path(d))['rows'],[])
            self.assertIsNone(C.metrics(Path(d))['source'])
    def test_logs_ignore_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=C.events_file(p);f.parent.mkdir()
            f.write_text('{"seq":1,"kind":"metric"}\n{"seq":2,"extra":{"x":NaN}}\n'+
                         '{"seq":3,"kind":"status"}\n')
            self.assertEqual([1,3],[r['seq'] for r in C.events(p)])
    def test_worker_csv_error_still_finalizes_process(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'ui_service').mkdir()
            C.save(folder/'ui_service'/'lifecycle.json',{'status':'running'})
            (folder/'results.csv').write_text('epoch,train_ce\n1,'+'x'*(csv.field_size_limit()+1)+'\n')
            proc=Mock();proc.poll.return_value=0;proc.wait.return_value=0
            with patch.dict(C._processes,{folder.name:proc},clear=True):
                C._worker(proc,folder)
                self.assertNotIn(folder.name,C._processes)
            self.assertEqual('finished',C.run_info(folder).get('status'))
            proc.wait.assert_called_once_with()

    def test_worker_unexpected_monitor_error_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'ui_service').mkdir()
            C.save(folder/'ui_service'/'lifecycle.json',{'status':'running'})
            proc=Mock();proc.poll.return_value=0;proc.wait.return_value=3
            with patch.object(C,'metrics',side_effect=RuntimeError('monitor boom')),patch.dict(C._processes,{folder.name:proc},clear=True):
                with self.assertRaisesRegex(RuntimeError,'monitor boom'):
                    C._worker(proc,folder)
                self.assertNotIn(folder.name,C._processes)
            life=C.run_info(folder)
            self.assertEqual('failed',life.get('status'))
            self.assertEqual(3,life.get('exit_code'))

    def test_audit_manifest_describes_exact_zip_bytes(self):
        import io,zipfile
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);evidence=folder/'results.csv';evidence.write_bytes(b'before')
            original_sha=C.sha
            def mutate_before_sha(path):
                if path==evidence:evidence.write_bytes(b'after-and-longer')
                return original_sha(path)
            with patch.object(C,'sha',side_effect=mutate_before_sha):
                bundle=C.audit_zip(folder)
            with zipfile.ZipFile(io.BytesIO(bundle)) as z:
                archived=z.read('results.csv')
                manifest=json.loads(z.read('audit_manifest.json'))
            entry=next(x for x in manifest['files'] if x['path']=='results.csv')
            self.assertEqual(len(archived),entry['bytes'])
            self.assertEqual(hashlib.sha256(archived).hexdigest(),entry['sha256'])

    def test_audit_reads_active_job_only_while_process_is_stopped(self):
        import os,signal,subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);evidence=folder/'results.csv';evidence.write_bytes(b'epoch,train_ce\n1,0.5\n')
            proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
                                  start_new_session=True,stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                original_read=Path.read_bytes;states=[]
                def checked_read(path):
                    if path==evidence:
                        stat=Path(f'/proc/{proc.pid}/stat').read_text('utf-8')
                        states.append(stat.rsplit(')',1)[1].split()[0])
                    return original_read(path)
                with patch.dict(C._processes,{folder.name:proc},clear=True),patch.object(Path,'read_bytes',checked_read):
                    C.audit_zip(folder)
                self.assertEqual(['T'],states)
                for _ in range(100):
                    stat=Path(f'/proc/{proc.pid}/stat').read_text('utf-8')
                    state=stat.rsplit(')',1)[1].split()[0]
                    if state not in ('T','t'):break
                    time.sleep(.01)
                self.assertNotIn(state,('T','t'))
                self.assertIsNone(proc.poll())
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM);os.killpg(proc.pid,signal.SIGCONT)
                proc.wait(timeout=5)

    def test_audit_resumes_process_when_snapshot_read_fails(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'results.csv').write_bytes(b'data')
            proc=Mock();proc.poll.return_value=None;proc.pid=12345
            with patch.dict(C._processes,{folder.name:proc},clear=True),patch.object(C,'_pause_group',return_value=proc),patch.object(C,'_resume_group') as resume,patch.object(Path,'read_bytes',side_effect=OSError('read failed')):
                with self.assertRaisesRegex(OSError,'read failed'):C.audit_zip(folder)
            resume.assert_called_once_with(proc)

    def test_audit_pause_failure_resumes_process_group(self):
        import signal
        proc=Mock();proc.poll.return_value=None;proc.pid=12345
        with patch.object(C.os,'killpg') as kill,patch.object(Path,'read_text',side_effect=OSError('proc unavailable')):
            with self.assertRaisesRegex(OSError,'proc unavailable'):C._pause_group(proc)
        self.assertEqual([(12345,signal.SIGSTOP),(12345,signal.SIGCONT)],
                         [call.args for call in kill.call_args_list])

    def test_worker_maps_graceful_stop_to_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'ui_service').mkdir()
            C.save(folder/'ui_service'/'lifecycle.json',{'status':'stopping','stop_requested':True})
            proc=Mock();proc.poll.return_value=0;proc.wait.return_value=0
            with patch.dict(C._processes,{folder.name:proc},clear=True):
                C._worker(proc,folder)
            life=C.run_info(folder)
            self.assertEqual('stopped',life['status'])
            self.assertEqual(0,life['exit_code'])

    def test_stop_race_with_already_exited_process_is_not_stopping(self):
        with tempfile.TemporaryDirectory() as d:
            runs=Path(d);folder=runs/'run_stop_race';folder.mkdir()
            C.save(folder/'ui_service'/'lifecycle.json',{'status':'running'})
            proc=Mock();proc.poll.return_value=None;proc.pid=12345
            with patch.object(C,'RUNS',runs),patch.dict(C._processes,{folder.name:proc},clear=True),patch.object(C.os,'killpg',side_effect=ProcessLookupError):
                with self.assertRaisesRegex(RuntimeError,'JOB_NOT_ACTIVE'):
                    C.stop_job('train',folder.name)
            life=C.run_info(folder)
            self.assertEqual('running',life['status'])
            self.assertNotIn('stop_requested',life)

    def test_training_payload_validation(self):
        valid={'task':'aerialvln_s_v1','architecture':'tinyvln_cpp_v1','name':'valid_run',
               'train_cache_id':'train_fixture','validation_cache_id':'val_fixture',
               'epochs':1,'sequence_chunk_size':16,'lr':0.0001,'seed':0}
        C._validate_payload('train',valid)
        invalid=[('name','bad name'),('epochs',0),('epochs',5001),('epochs',True),
                 ('sequence_chunk_size',0),('sequence_chunk_size',1.5),
                 ('lr',0),('lr',-1),('lr',True),('lr',float('inf')),
                 ('seed',-1),('seed',0.5),('seed',True)]
        for field,value in invalid:
            with self.subTest(field=field,value=value):
                with self.assertRaises(ValueError):C._validate_payload('train',{**valid,field:value})
        for field in valid:
            with self.subTest(missing=field):
                payload={**valid};del payload[field]
                with self.assertRaises(ValueError):C._validate_payload('train',payload)

    def test_simulation_payload_validation(self):
        valid={'task':'aerialvln_s_v1','checkpoint_id':'checkpoint','split_id':'split',
               'scene_id':'scene','auto_launch':False,'mode':'diagnostic',
               'record_observations':True,'audit_artifacts':True}
        C._validate_payload('sim',valid)
        for field in ('auto_launch','record_observations','audit_artifacts'):
            for value in ('false',0,1,None):
                with self.subTest(field=field,value=value):
                    with self.assertRaises(ValueError):C._validate_payload('sim',{**valid,field:value})
        with self.assertRaises(ValueError):C._validate_payload('sim',{**valid,'mode':'invalid'})

    def test_profile_is_blocked_until_verified(self):
        self.assertIn('PROFILE_NOT_VERIFIED',C.profile_check({'ready':False,'executable':'missing'}))
        self.assertIn('CLOSED_LOOP_NOT_VERIFIED',C.profile_check({'ready':True,'executable':'missing'},sim=True))

class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.server.daemon_threads=True
        cls.port=cls.server.server_port
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2)
    def call(self,path,method='GET',body=None,headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=10)
        h={'Host':f'127.0.0.1:{self.port}'}
        h.update(headers or {})
        c.request(method,path,body=body,headers=h)
        r=c.getresponse();data=r.read();status=r.status;c.close()
        return status,data
    def test_status_and_no_claim(self):
        status,b=self.call('/api/v1/health');self.assertEqual(200,status)
        j=json.loads(b);self.assertFalse(j['physics_verified']);self.assertFalse(j['model_training_verified'])
    def test_host_reject(self):
        status,_=self.call('/api/v1/health',headers={'Host':f'evil.example:{self.port}'})
        self.assertEqual(421,status)
    def test_json_required(self):
        status,_=self.call('/api/v1/runs','POST','{}',headers={'Content-Type':'text/plain'})
        self.assertEqual(415,status)
    def test_no_unverified_run(self):
        payload={'architecture':'tinyvln_cpp_v1','task':'aerialvln_s_v1','name':'unverified_run',
                 'train_cache_id':'train_fixture','validation_cache_id':'val_fixture',
                 'epochs':1,'sequence_chunk_size':16,'lr':0.0001,'seed':0}
        status,b=self.call('/api/v1/runs','POST',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        self.assertEqual(409,status);self.assertIn(b'PROFILE_NOT_REGISTERED',b)
    def test_invalid_job_payloads_are_rejected_before_launch(self):
        train={'architecture':'tinyvln_cpp_v1','task':'aerialvln_s_v1','name':'invalid_run',
               'train_cache_id':'train_fixture','validation_cache_id':'val_fixture',
               'epochs':-1,'sequence_chunk_size':16,'lr':0.0001,'seed':0}
        sim={'task':'aerialvln_s_v1','checkpoint_id':'checkpoint','split_id':'split','scene_id':'scene',
             'auto_launch':'false','mode':'diagnostic','record_observations':True,'audit_artifacts':True}
        for path,payload,code in [('/api/v1/runs',train,b'EPOCHS_INVALID'),
                                  ('/api/v1/simulation-jobs',sim,b'AUTO_LAUNCH_INVALID')]:
            with self.subTest(path=path):
                status,b=self.call(path,'POST',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                self.assertEqual(400,status);self.assertIn(code,b)
    def test_sim_caps_blocked(self):
        status,b=self.call('/api/v1/simulation/capabilities?task=lani_v1')
        self.assertEqual(200,status)
        self.assertFalse(json.loads(b)['ready'])
    def test_ui_is_served(self):
        status,b=self.call('/');self.assertEqual(200,status);self.assertIn(b'NavLab',b)
    def test_confidential_file_not_served(self):
        status,_=self.call('/config/registry.json');self.assertEqual(404,status)
    def test_nonfinite_body_rejected(self):
        status,_=self.call('/api/v1/runs','POST',b'{"x":NaN}',headers={'Content-Type':'application/json'})
        self.assertEqual(400,status)
    def test_cross_origin_rejected(self):
        status,_=self.call('/api/v1/runs','POST',b'{}',headers={'Content-Type':'application/json','Origin':'http://evil.example'})
        self.assertEqual(403,status)
    def test_metrics_csv_error_returns_json(self):
        with tempfile.TemporaryDirectory() as d:
            runs=Path(d);folder=runs/'run_bad_csv';folder.mkdir()
            (folder/'results.csv').write_text('epoch,train_ce\n1,'+'x'*(csv.field_size_limit()+1)+'\n')
            with patch.object(C,'RUNS',runs):
                status,b=self.call('/api/v1/runs/run_bad_csv/metrics')
            self.assertEqual(500,status)
            self.assertEqual('READ_FAILED',json.loads(b)['error']['code'])

    def test_no_fake_inference(self):
        status,b=self.call('/api/v1/inference/sessions','POST',b'{}',headers={'Content-Type':'application/json'})
        self.assertEqual(501,status)

if __name__=='__main__':unittest.main()

class TestManagedJobs(unittest.TestCase):
    """Synthetic subprocess is a SUPERVISOR TEST, never a claimed UAV trainer."""
    def test_trusted_registered_cli_and_audit(self):
        import sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'runs').mkdir();(root/'simulations').mkdir()
            script=root/'synthetic_smoke_runner.py'
            script.write_text('''import csv,sys
from pathlib import Path
args=sys.argv
out=Path(args[args.index("--output")+1]);out.mkdir(exist_ok=True,parents=True)
with (out/"results.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["epoch","train_ce","dev_ce","dev_accuracy"])
 w.writerow([1,0.7,0.8,0.25])
print("test-only subprocess completed")
''')
            config={'schema':'navlab.registry.v1',
              'training_profiles':[{'id':'test_profile','task':'aerialvln_s_v1',
                'architecture':'tinyvln_cpp_v1','ready':True,'executable':sys.executable,
                'argv':[str(script),'--output','{run_dir}']}],
              'simulation_profiles':[],
              'datasets':[{'id':'train_fixture','task':'aerialvln_s_v1','verified':True},
                          {'id':'val_fixture','task':'aerialvln_s_v1','verified':True}]}
            (root/'registry.json').write_text(json.dumps(config))
            with patch.object(C,'REGISTRY',root/'registry.json'),patch.object(C,'RUNS',root/'runs'),patch.object(C,'SIMS',root/'simulations'):
                job,life=C.launch('train',{'architecture':'tinyvln_cpp_v1','task':'aerialvln_s_v1','name':'synthetic_smoke',
                         'train_cache_id':'train_fixture','validation_cache_id':'val_fixture','epochs':1,
                         'sequence_chunk_size':16,'lr':0.0001,'seed':0})
                f=root/'runs'/job
                for _ in range(100):
                    if C.run_info(f).get('status') in ('finished','failed'):break
                    time.sleep(.05)
                self.assertEqual('finished',C.run_info(f).get('status'))
                self.assertEqual(1,C.metrics(f)['rows'][0]['epoch'])
                for _ in range(50):
                    if any(e['kind']=='metric' for e in C.events(f)):break
                    time.sleep(.02)
                self.assertTrue(any(e['kind']=='metric' for e in C.events(f)))
                import zipfile,io
                z=zipfile.ZipFile(io.BytesIO(C.audit_zip(f)))
                self.assertIn('results.csv',z.namelist())
                self.assertIn('audit_manifest.json',z.namelist())

    def test_user_stop_records_stopped_lifecycle(self):
        import os,signal,subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            runs=Path(d)/'runs';folder=runs/'run_stop_test';folder.mkdir(parents=True)
            C.save(folder/'ui_service'/'lifecycle.json',{'status':'running'})
            proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
                                  start_new_session=True,stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                with patch.object(C,'RUNS',runs),patch.dict(C._processes,{folder.name:proc},clear=True):
                    self.assertEqual({'status':'stopping'},C.stop_job('train',folder.name))
                    life=C.run_info(folder)
                    self.assertEqual('stopping',life['status'])
                    self.assertTrue(life['stop_requested'])
                    proc.wait(timeout=5)
                    C._worker(proc,folder)
                    life=C.run_info(folder)
                    self.assertEqual('stopped',life['status'])
                    self.assertEqual(-signal.SIGTERM,life['exit_code'])
                    self.assertNotIn(folder.name,C._processes)
                    self.assertTrue(any(e.get('status')=='stopped' for e in C.events(folder)))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)

    def test_simulation_cannot_run_just_because_executable_exists(self):
        import sys
        reason=C.profile_check({'ready':True,'executable':sys.executable,'requirements':[]},sim=True)
        self.assertIn('SCENE_SUPERVISOR_NOT_VERIFIED',reason)
        self.assertIn('CLOSED_LOOP_NOT_VERIFIED',reason)
