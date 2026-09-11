"""The only formal Phase 3 entry point. Run with python -m experiments.phase3.coordinator."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
import re
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time

from .common import Phase3Error, atomic, digest, identity, inventory, lock, publish, read, safe_path, sealed, under, verify_inventory
from .preflight import load_specs, resolve
from .transport import Remote, make_transport
from .worker import machine_identity

TOOL=Path(__file__).parent.resolve()
BACKTEST_ARGS=['--start_time','2023-01-01','--end_time','2025-12-31','--topk','30','--drop','5',
               '--open_cost','0.0005','--close_cost','0.0015','--min_cost','0']


def metric_csv(path,column='values'):
    with open(path) as f:
        rows=list(csv.reader(f))
    i=rows[0].index(column)
    values={r[0]:float(r[i]) for r in rows[1:] if len(r)>i and r[i]}
    if not values or not all(math.isfinite(v) for v in values.values()):
        raise Phase3Error(f'invalid metric values: {path}')
    return values


def aggregate_rows(rows):
    keys=set(rows[0]['metrics'])
    if any(set(r['metrics'])!=keys for r in rows):raise Phase3Error('inconsistent metric sets')
    return {k:{'mean':statistics.mean(r['metrics'][k] for r in rows),
               'std':statistics.stdev(r['metrics'][k] for r in rows) if len(rows)>1 else None,
               'n':len(rows)} for k in sorted(keys)}


class Coordinator:
    def __init__(self,repo,batch,machines,*,notify=False,allow_shutdown=False,fixture=False,
                 transport_factory=make_transport,poll_seconds=5):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',str(batch.get('batch_id',''))):
            raise Phase3Error('invalid batch id')
        self.repo=Path(repo).resolve(); self.batch=batch; self.machines=machines
        base=self.repo/'artifacts/_phase3'/('diagnostics' if fixture else 'batches')
        self.root=safe_path(base/batch['batch_id']); self.root.mkdir(parents=True,exist_ok=True)
        self.notify_enabled=notify; self.allow_shutdown=allow_shutdown; self.fixture=fixture
        self.factory=transport_factory; self.poll_seconds=poll_seconds
        self.python=batch.get('local_python',sys.executable)
        self.refs={}; self.accepted={}; self.event_lock=threading.Lock()
        self.failures=[]

    def event(self,key,kind,body):
        with self.event_lock:
            path=self.root/'events'/f'{key}.json'
            event=read(path) if path.exists() else {'kind':kind,'body':body,'status':'pending','attempts':0}
            if event['status']=='sent':return
            if not self.notify_enabled or self.fixture:
                event['status']='suppressed'
                atomic(path,event); return
            if event.get('status')=='failed' and time.time()<event.get('retry_after',0):return
            event['attempts']+=1
            body_path=path.with_suffix('.txt'); body_path.parent.mkdir(parents=True,exist_ok=True)
            body_path.write_text(body,encoding='utf-8')
            try:
                # Phase 2 convention: SMTP credentials live in ~/.bashrc, which a
                # non-interactive non-login shell never loads. Deliver through a
                # login shell so notifications are not deterministically lost when
                # the coordinator itself runs from a plain shell.
                command=shlex.join([self.python,str(self.repo/'experiments/notify.py'),
                                    '--subject',f'[HVQ] Phase3 {kind} — {self.batch["batch_id"]}',
                                    '--body-file',str(body_path)])
                subprocess.run(['bash','-lc',command],check=True,capture_output=True,text=True,timeout=45)
                event['status']='sent'
            except Exception as e:
                # Never include environment or SMTP response/credentials in our state.
                event.update(status='failed',error=type(e).__name__,retry_after=time.time()+60)
            atomic(path,event)

    def helper(self,action,req,name):
        path=self.root/'checks'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        request=path.with_suffix('.request.json'); atomic(request,req)
        with path.with_suffix('.log').open('a') as log:
            subprocess.run([self.python,str(TOOL/'runtime.py'),action,'--request',str(request),
                            '--output',str(path)],check=True,stdout=log,stderr=subprocess.STDOUT,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        return read(path)

    def prepare(self,deep=True):
        """CPU-only local preflight. No remote connection and no worker launch."""
        self.refs=resolve(self.repo,self.batch,self.root/'reference')
        for key,r in self.refs.items():
            if deep:
                r['data_fingerprint']=self.helper('fingerprint',
                    {'paths':{s:str(Path(r['reference_data'])/p) for s,p in r['data_files'].items()},
                     'code':r['code_local']},f'data-{key}.json')
        # Control files are frozen independently of mutable Phase 2 queue status.
        manifest={'batch':self.batch,'machines':{d['machine']:self.machines[d['machine']] for d in self.batch['devices']},
                  'references':self.refs,'tool_files':{p.name:digest(p) for p in TOOL.glob('*.py')},
                  'backtest_sha256':digest(self.repo/'backtest_qlib.py'),'backtest_args':BACKTEST_ARGS}
        if deep:
            sealed(self.root/'manifest.json',manifest)
        else:
            atomic(self.root/'dry_run.json',manifest)
        return manifest

    def restore(self):
        manifest=read(self.root/'manifest.json')
        if manifest['batch']!=self.batch or manifest['machines']!={d['machine']:self.machines[d['machine']] for d in self.batch['devices']}:
            raise Phase3Error('batch/machine configuration changed; use a new batch id')
        if manifest['tool_files']!={p.name:digest(p) for p in TOOL.glob('*.py')}:
            raise Phase3Error('executor changed; resume with the original implementation')
        if manifest['backtest_sha256']!=digest(self.repo/'backtest_qlib.py'):
            raise Phase3Error('backtest implementation changed')
        self.refs=manifest['references']
        self.protect_seed0()

    def protect_seed0(self):
        for key,r in self.refs.items():
            verify_inventory(self.repo/'artifacts'/key/'run',r['seed0_tree'],allow_symlinks=True)

    def build_job(self,device,transport):
        name=device['machine']; dest=self.root/'deploy'/name
        dest.mkdir(parents=True,exist_ok=True)
        tool=dest/'tool'; tool.mkdir(exist_ok=True)
        for p in TOOL.glob('*.py'):
            q=tool/p.name
            if q.exists() and digest(q)!=digest(p):raise Phase3Error('tool conflict')
            if not q.exists():shutil.copyfile(p,q)
        tasks=[]
        for spec in device['experiments']:
            key=spec['id']; r=self.refs[key]
            code=dest/'code'/key
            publish(r['code_local'],code)
            for src,rel in [(r['stage1_local'],f'inputs/{r["stage1_sha256"]}/{Path(r["stage1_local"]).name}'),
                            (r['jkp_local'],f'inputs/{r["jkp_sha256"]}/{Path(r["jkp_local"]).name}')]:
                p=under(dest,rel); p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and digest(p)!=digest(src):raise Phase3Error('input conflict')
                if not p.exists():shutil.copyfile(src,p)
            data_key=identity({'fingerprint':r['data_fingerprint'],'jkp':r['jkp_sha256']})
            for seed in spec.get('seeds',self.batch.get('seeds',[1,2,3,4])):
                t={k:r[k] for k in ('experiment','commit','universe','stage1_sha256','data_fingerprint','data_files')}
                t.update(seed=seed,code=f'code/{key}',data=f'data/{data_key}',
                         stage1=f'inputs/{r["stage1_sha256"]}/{Path(r["stage1_local"]).name}',
                         jkp=f'inputs/{r["jkp_sha256"]}/{Path(r["jkp_local"]).name}')
                t['task_id']=identity(t)
                tasks.append(t)
        inputs=inventory(dest)
        inputs.pop('job.json',None)
        job={'fixture':False,'tasks':tasks,'inputs':inputs,'machine_identity':self.machines[name]['identity'],
             'gpu':str(self.machines[name].get('gpu','0')), 'device_lock':str(transport.root/'.device.lock')}
        sealed(dest/'job.json',job)
        return dest,job

    def accept(self,task,source):
        manifest=read(source/'result_manifest.json')
        if manifest.get('task_id')!=task['task_id'] or manifest.get('fixture')!=self.fixture:
            raise Phase3Error('result manifest identity/fixture mismatch')
        expected={**manifest['files'],'result_manifest.json':{'sha256':digest(source/'result_manifest.json'),
                                                            'size':(source/'result_manifest.json').stat().st_size}}
        verify_inventory(source,expected)
        validation=self.helper('inspect',{'task':task,'folder':str(source),'fixture':self.fixture},
                               f'accept-{task["task_id"]}.json')
        if not self.fixture:
            self.helper('compare_prediction',{'prediction':str(source/validation['prediction']),
                        'reference':self.refs[task['experiment']]['seed0_prediction']},
                        f'prediction-{task["task_id"]}.json')
        if self.fixture:
            archive=self.root/'accepted'/task['experiment']/str(task['seed'])
        else:
            archive=under(self.repo/'artifacts',f'{task["experiment"]}/phase3/{self.batch["batch_id"]}/seed{task["seed"]}')
        publish(source,archive)
        receipt={'task':task,'archive':str(archive),'files':expected,'validation':validation,'accepted':True}
        sealed(self.root/'receipts'/f'{task["task_id"]}.json',receipt)
        self.accepted[task['task_id']]=receipt
        return receipt

    def load_receipt(self,task):
        p=self.root/'receipts'/f'{task["task_id"]}.json'
        if not p.exists():return False
        receipt=read(p)
        if receipt['task']!=task or not receipt.get('accepted'):raise Phase3Error('receipt identity mismatch')
        verify_inventory(receipt['archive'],receipt['files'])
        self.accepted[task['task_id']]=receipt
        return True

    def device_loop(self,device,job,transport,remote_root,retry=False):
        name=device['machine']; machine=self.machines[name]
        if transport.identity()!=machine['identity']:raise Phase3Error('machine identity mismatch')
        if isinstance(transport,Remote) and machine['identity']['machine_id']==machine_identity()['machine_id']:
            raise Phase3Error('remote alias points to coordinator machine')
        for t in job['tasks']:self.load_receipt(t)
        # All accepted: never contact/start a worker merely for evaluation recovery.
        if not all(t['task_id'] in self.accepted for t in job['tasks']):
            verified=transport.worker(remote_root,'verify')
            if verified['job_sha256'] != digest(self.root/'deploy'/name/'job.json'):
                raise Phase3Error('remote job manifest conflict')
            probe=transport.worker(remote_root,'probe')
            if {k:probe[k] for k in ('hostname','machine_id')} != machine['identity']:
                raise Phase3Error('probe identity mismatch')
            if probe['disk_free'] < machine.get('min_free_gb',10)*1024**3:
                raise Phase3Error('insufficient device disk space')
            if not self.fixture:
                local_env=self.helper('environment',{},f'environment-{name}.json')
                expected={k:v.split('+')[0] for k,v in local_env['packages'].items()}
                actual={k:v.split('+')[0] for k,v in probe['environment']['packages'].items()}
                if expected != actual or local_env['python'].split('.')[:2] != probe['environment']['python'].split('.')[:2]:
                    raise Phase3Error('software environment mismatch')
            atomic(self.root/'probes'/f'{name}.json',probe)
            transport.worker(remote_root,'start',retry)
        while True:
            all_accepted=all(t['task_id'] in self.accepted for t in job['tasks'])
            if all_accepted:
                state={'status':'finished','active':False,'tasks':{}}
            else:
                # SSH errors bubble out as attention-required. Resume queries the
                # existing worker and its lock; no immediate duplicate launch.
                state=transport.worker(remote_root,'status')
            for task in job['tasks']:
                tid=task['task_id']
                if tid in self.accepted:continue
                ts=state.get('tasks',{}).get(tid,{})
                if ts.get('status')=='ready':
                    source=transport.result_path(remote_root,ts['result'])
                    incoming=self.root/'incoming'/tid/str(ts['attempt'])
                    transport.pull(source,incoming)
                    self.accept(task,incoming)
            for spec in device['experiments']:
                tasks=[t for t in job['tasks'] if t['experiment']==spec['id']]
                if all(t['task_id'] in self.accepted for t in tasks):
                    self.event('experiment-'+spec['id'],'experiment multi-seed completed',
                               f'Experiment: {spec["id"]}\nRequested seeds: {[t["seed"] for t in tasks]}\n'
                               f'Training complete; all artifacts accepted locally.\nEvaluation/backtest/aggregate pending.\nBatch: {self.batch["batch_id"]}\n'
                               f'Artifacts: {[self.accepted[t["task_id"]]["archive"] for t in tasks]}')
                    ack=self.root/'acks'/name/spec['id']; ack.mkdir(parents=True,exist_ok=True)
                    sealed(ack/f'{spec["id"]}.json',{'task_ids':[t['task_id'] for t in tasks]})
                    transport.stage(ack,Path(remote_root)/'acks')
            if all(t['task_id'] in self.accepted for t in job['tasks']):
                # Require real worker termination/lock release before shutdown.
                actual=transport.worker(remote_root,'status')
                if not actual.get('active') and actual.get('status')=='finished':
                    if actual.get('device_busy'):raise Phase3Error('device used by another batch; shutdown blocked')
                    break
            if not state.get('active') and state.get('status') in ('failed','finished','running','not_started') and not all(t['task_id'] in self.accepted for t in job['tasks']):
                raise Phase3Error(f'device incomplete: {state}')
            time.sleep(self.poll_seconds)
        atomic(self.root/'devices'/f'{name}.json',{'status':'accepted','tasks':[t['task_id'] for t in job['tasks']]})
        self.finish_device(name,job,transport)

    def finish_device(self,name,job,transport):
        if not isinstance(transport,Remote):return
        if self.machines[name].get('shutdown','disabled')=='disabled' and not self.fixture:return
        for t in job['tasks']:
            if not self.load_receipt(t):raise Phase3Error('shutdown blocked: unaccepted task')
        path=self.root/'power'/f'{name}.json'
        if path.exists():
            previous=read(path)
            if previous['status'] in ('requested','requested_unconfirmed','confirmed'):return
        dry=self.fixture or not self.allow_shutdown
        atomic(path,{'status':'dry_run_pending' if dry else 'requested','identity':self.machines[name]['identity']})
        result=transport.shutdown(self.machines[name]['identity'],dry_run=dry)
        atomic(path,{'status':result,'identity':self.machines[name]['identity']})
        if result!='dry_run':
            self.event('shutdown-'+name,'remote shutdown',f'{name}: {result}. Batch {self.batch["batch_id"]}.')
        if result=='requested_unconfirmed':
            self.event('power-attention-'+name,'attention required',f'{name}: shutdown requested; power-off NOT confirmed. Do not retry blindly.')

    def evaluate(self):
        summaries={}
        for device in self.batch['devices']:
            for spec in device['experiments']:
                key=spec['id']; rows=[]; r=self.refs[key]
                seeds=spec.get('seeds',self.batch.get('seeds',[1,2,3,4]))
                available={v['task']['seed'] for v in self.accepted.values() if v['task']['experiment']==key}
                if not set(seeds).issubset(available):
                    summaries[key]={'status':'incomplete','requested_seeds':seeds,'accepted_seeds':sorted(available)}
                    continue
                for seed in [0,*seeds]:
                    if seed==0:
                        pred_metric=r['seed0_metric']; portfolio=r['seed0_backtest']
                    else:
                        receipts=[v for v in self.accepted.values() if v['task']['experiment']==key and v['task']['seed']==seed]
                        if len(receipts)!=1:raise Phase3Error('aggregate requires all requested accepted seeds')
                        receipt=receipts[0]; archive=Path(receipt['archive'])
                        pred=archive/receipt['validation']['prediction']; pred_metric=pred.parent/f'{seed}_metric.csv'
                        out=self.root/'evaluation'/key/f'seed{seed}'
                        done=out/'evaluation.json'
                        signature=identity({'receipt':receipt,'backtest':digest(self.repo/'backtest_qlib.py'),'args':BACKTEST_ARGS})
                        if done.exists():
                            saved=read(done)
                            if saved['signature']!=signature:raise Phase3Error('evaluation provenance conflict')
                            verify_inventory(out,saved['files'],exact=False)
                            portfolio=out/saved['metric']
                        else:
                            attempt=out/'attempts'/str(time.time_ns()); attempt.mkdir(parents=True)
                            with (attempt/'backtest.log').open('w') as log:
                                subprocess.run([self.python,str(self.repo/'backtest_qlib.py'),
                                                '--pred_path',str(pred),'--universe',r['universe'],
                                                '--output_dir',str(attempt),*BACKTEST_ARGS],
                                               cwd=self.repo,check=True,stdout=log,stderr=subprocess.STDOUT,
                                               env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
                            portfolio=attempt/'portfolio_metric.csv'
                            metric_csv(portfolio,'project_portfolio')
                            atomic(done,{'signature':signature,'metric':str(portfolio.relative_to(out)),
                                         'files':inventory(out)})
                    metrics=metric_csv(pred_metric)
                    metrics.update({'portfolio.'+k:v for k,v in metric_csv(portfolio,'project_portfolio').items()})
                    rows.append({'seed':seed,'metrics':metrics})
                summary={'experiment':key,'protocol':'fixed Stage 1; independent Stage 2 seeds',
                         'rows':rows,'aggregate':aggregate_rows(rows),'std_ddof':1}
                summaries[key]=summary
                atomic(self.root/'reports'/f'{key}.json',summary)
        atomic(self.root/'reports/summary.json',summaries)
        complete=not self.failures and all(v.get('status')!='incomplete' for v in summaries.values())
        self.event('batch-completed' if complete else 'batch-incomplete',
                   'batch completed' if complete else 'batch incomplete',json.dumps({'experiments':summaries,'failed':self.failures,
                   'reports':str(self.root/'reports'),'shutdown':{p.stem:read(p) for p in (self.root/'power').glob('*.json')}},indent=2))
        return summaries

    def run(self,retry=False):
        with lock(self.root/'coordinator.lock'):
            if (self.root/'manifest.json').exists():self.restore()
            else:self.prepare()
            try:
                with ThreadPoolExecutor(max_workers=len(self.batch['devices'])) as pool:
                    futures={}
                    for device in self.batch['devices']:
                        m=self.machines[device['machine']]; tr=self.factory(m)
                        if m['kind']=='local' and (tr.root.is_relative_to(self.repo/'artifacts') or tr.root==self.repo):
                            raise Phase3Error('local worker must use dedicated scratch outside canonical artifacts')
                        dest,job=self.build_job(device,tr)
                        remote_root=tr.root/self.batch['batch_id']
                        if all(self.load_receipt(t) for t in job['tasks']) and (self.root/'devices'/f'{device["machine"]}.json').exists():
                            continue  # evaluation recovery after remote power-off is fully local
                        # identity before staging. Input verification runs again before every fit.
                        if tr.identity()!=m['identity']:raise Phase3Error('machine identity mismatch')
                        tr.stage(dest,remote_root)
                        futures[pool.submit(self.device_loop,device,job,tr,remote_root,retry)]=device['machine']
                    for future in as_completed(futures):
                        name=futures[future]
                        try:future.result()
                        except Exception as e:
                            self.failures.append({'device':name,'error':str(e)})
                            self.event('attention-'+name,'attention required',str(e))
                if self.failures:
                    atomic(self.root/'failures.json',self.failures)
                    self.evaluate()  # evaluate complete experiments, report partial coverage honestly
                    raise Phase3Error('batch has incomplete devices; inspect state and resume')
                return self.evaluate()
            finally:
                self.protect_seed0()


def diagnostic(repo,machine_name,machines,batch_id,python):
    """Real transport+CPU fixture, same worker and acceptance, never formal outputs."""
    batch={'schema':1,'batch_id':batch_id,'local_python':python,'seeds':[1,2],
           'devices':[{'machine':machine_name,'experiments':[{'id':'fixtureA'},{'id':'fixtureB'}]}]}
    c=Coordinator(repo,batch,machines,fixture=True,poll_seconds=.5)
    tr=make_transport(machines[machine_name]); device=batch['devices'][0]
    with lock(c.root/'coordinator.lock'):
        dest=c.root/'deploy'/machine_name; (dest/'tool').mkdir(parents=True,exist_ok=True)
        for p in TOOL.glob('*.py'):
            target=dest/'tool'/p.name
            if target.exists() and digest(target)!=digest(p):raise Phase3Error('diagnostic tool changed; new batch id needed')
            if not target.exists():shutil.copyfile(p,target)
        tasks=[]
        for exp in device['experiments']:
            for seed in batch['seeds']:
                t={'experiment':exp['id'],'seed':seed,'commit':'fixture','stage1_sha256':'fixture',
                   'data_fingerprint':{'fixture':'synthetic CPU'}}
                t['task_id']=identity(t); tasks.append(t)
        files=inventory(dest); files.pop('job.json',None)
        job={'tasks':tasks,'fixture':True,'inputs':files,'gpu':'','machine_identity':machines[machine_name]['identity'],'device_lock':str(tr.root/'.device.lock')}
        sealed(dest/'job.json',job)
        if tr.identity()!=job['machine_identity']:raise Phase3Error('machine identity mismatch')
        remote_root=tr.root/'diagnostics'/batch_id
        tr.stage(dest,remote_root)
        c.device_loop(device,job,tr,remote_root)
        result={'fixture':True,'accepted':len(c.accepted),'events':[read(p)['kind'] for p in (c.root/'events').glob('*.json')],
                'root':str(c.root),'remote_root':str(remote_root),'shutdown':'dry_run_only'}
        atomic(c.root/'diagnostic.json',result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['dry-run','run','resume','status','diagnostic'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--batch',type=Path)
    p.add_argument('--machines',type=Path,required=True)
    p.add_argument('--notify',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--allow-shutdown',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--machine'); p.add_argument('--diagnostic-id',default='cpu-check')
    p.add_argument('--python',default=sys.executable)
    a=p.parse_args()
    try:
        if a.action=='diagnostic':
            import yaml
            if a.allow_shutdown or a.notify is True:raise Phase3Error('diagnostic cannot notify or shut down')
            machines=yaml.safe_load(a.machines.read_text())['machines']
            # Reuse machine/path validation with a syntactically formal temporary spec.
            result=diagnostic(a.repo,a.machine,machines,a.diagnostic_id,a.python)
        else:
            if not a.batch:raise Phase3Error('--batch required')
            b,m=load_specs(a.batch,a.machines)
            c=Coordinator(a.repo,b,m,notify=(a.notify is not False and a.action in ('run','resume')),allow_shutdown=a.allow_shutdown)
            if a.action=='dry-run':
                if a.allow_shutdown:raise Phase3Error('dry-run cannot enable shutdown')
                with lock(c.root/'coordinator.lock'):result=c.prepare(deep=False)
                result={'status':'dry_run','experiments':list(c.refs),'training':False,'remote_connections':False}
            elif a.action=='status':
                result={'receipts':[p.stem for p in (c.root/'receipts').glob('*.json')],
                        'devices':{p.stem:read(p) for p in (c.root/'devices').glob('*.json')},
                        'root':str(c.root)}
            else:
                result=c.run(retry=a.retry_failed)
        print(json.dumps(result,indent=2))
    except Exception as e:
        if a.action in ('run','resume') and 'c' in locals():
            c.event('coordinator-attention','attention required',f'{type(e).__name__}: {e}')
        print(f'Phase3: {type(e).__name__}: {e}',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
