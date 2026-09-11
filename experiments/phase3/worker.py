"""Detached per-device ordered worker. No mail, shutdown, or global scheduling."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

try:
    from .common import Phase3Error, atomic, digest, inventory, lock, read, safe_path, verify_inventory
except ImportError:
    from common import Phase3Error, atomic, digest, inventory, lock, read, safe_path, verify_inventory

RUNTIME = Path(__file__).with_name('runtime.py')


def machine_identity():
    return {'hostname': platform.node(), 'machine_id': Path('/etc/machine-id').read_text().strip()}


def check_identity(expected):
    if machine_identity() != expected:
        raise Phase3Error('machine identity mismatch')


def status(root):
    state_path = root/'worker_state.json'
    state = read(state_path) if state_path.exists() else {'status':'not_started','tasks':{}}
    try:
        with lock(root/'worker.lock'):
            active = False
    except Phase3Error:
        active = True
    busy = False
    if (root/'job.json').exists():
        try:
            with lock(read(root/'job.json')['device_lock']) as lease_fd:
                pass
        except Phase3Error:
            busy = True
    return {**state, 'active':active, 'device_busy':busy}


def run_helper(action, request, target, log, env, lease_fd=None):
    req = target.with_suffix('.request.json')
    atomic(req, request)
    with log.open('a') as f:
        subprocess.run([sys.executable, str(RUNTIME), action, '--request', str(req),
                        '--output', str(target)], check=True, stdout=f, stderr=subprocess.STDOUT, env=env,
                        pass_fds=() if lease_fd is None else (lease_fd,))
    return read(target)


def prepare_data(root, task, env):
    data = root/task['data']
    stamp = data/'fingerprint.json'
    code = root/task['code']
    if not data.exists():
        # Fresh generation directory. A failed partial generation is preserved and
        # must not be silently reused or overwritten on restart.
        with (root/'data-generation.log').open('a') as f:
            subprocess.run([sys.executable, '-m', 'dataset.get_dataset',
                            '--universe', task['universe'], '--output-dir', str(data),
                            '--jkp-path', str(root/task['jkp'])], cwd=code, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    paths = {k:str(data/v) for k,v in task['data_files'].items()}
    actual = run_helper('fingerprint', {'paths':paths,'code':str(code)},
                        root/('fingerprint-'+task['experiment']+'.json'), root/'data-generation.log',env)
    if actual != task['data_fingerprint']:
        raise Phase3Error('data fingerprint mismatch; training blocked')
    atomic(stamp, actual)


def wait_acceptance(root, job, task):
    tasks = [t for t in job['tasks'] if t['experiment'] == task['experiment']]
    if task['task_id'] != tasks[-1]['task_id']:
        return
    state = read(root/'worker_state.json')
    if not all(state['tasks'].get(t['task_id'], {}).get('status') == 'ready' for t in tasks):
        return
    ack = root/'acks'/f"{task['experiment']}.json"
    while not ack.exists():
        time.sleep(.2)
    if read(ack) != {'task_ids': [t['task_id'] for t in tasks]}:
        raise Phase3Error('acceptance acknowledgement mismatch')


def work(root, retry_failed=False):
    root = safe_path(root)
    with lock(root/'worker.lock'), lock(read(root/'job.json')['device_lock']) as lease_fd:
        job = read(root/'job.json')
        check_identity(job['machine_identity'])
        verify_inventory(root, job['inputs'], exact=False)
        state = read(root/'worker_state.json') if (root/'worker_state.json').exists() else {'tasks':{}}
        state.update(status='running', pid=os.getpid(), started=time.time())
        atomic(root/'worker_state.json',state)
        env = {**os.environ, 'WANDB_MODE':'offline', 'PYTHONHASHSEED':'0', 'PYTHONDONTWRITEBYTECODE':'1',
               'CUDA_VISIBLE_DEVICES': '' if job['fixture'] else job['gpu']}
        prepared = set()
        try:
            if not job['fixture']:
                try:
                    from .runtime import environment
                except ImportError:
                    from runtime import environment
                os.environ['CUDA_VISIBLE_DEVICES'] = job['gpu']
                atomic(root/'gpu-preflight.json', environment(gpu=True))
            for task in job['tasks']:
                key = task['task_id']
                old = state['tasks'].get(key,{})
                if old.get('status') == 'ready':
                    # A ready result is never retrained due to transfer/eval failure.
                    manifest = read(root/old['result']/'result_manifest.json')
                    verify_inventory(root/old['result'], manifest['files'], exact=False)
                    wait_acceptance(root, job, task)
                    continue
                if old.get('status') in ('failed','running') and not retry_failed:
                    state['tasks'][key] = {**old, 'status':'failed',
                        'error':old.get('error','interrupted attempt; explicit retry required')}
                    atomic(root/'worker_state.json',state)
                    continue
                attempt = old.get('attempt',0)+1
                out = root/'attempts'/key/str(attempt)
                out.mkdir(parents=True, exist_ok=False)
                state['tasks'][key] = {'status':'running','attempt':attempt,'result':str(out.relative_to(root))}
                atomic(root/'worker_state.json',state)
                try:
                    if not job['fixture']:
                        check_identity(job['machine_identity'])
                        verify_inventory(root, job['inputs'], exact=False)
                        if task['data'] not in prepared:
                            prepare_data(root,task,env)
                            prepared.add(task['data'])
                        if digest(root/task['stage1']) != task['stage1_sha256']:
                            raise Phase3Error('stage1 changed')
                    env['PYTHONHASHSEED'] = str(task['seed'])
                    run_helper('fixture' if job['fixture'] else 'train',
                               {'task':task,'root':str(root),'out':str(out)},
                               out/'execution.json',out/'stage2.log',env,lease_fd=lease_fd)
                    run_helper('inspect',{'task':task,'folder':str(out),'fixture':job['fixture']},
                               out/'validation.json',out/'validation.log',
                               {**env,'CUDA_VISIBLE_DEVICES':''})
                    validation = read(out/'validation.json')
                    export = out/'export'
                    export.mkdir()
                    required = [validation['checkpoint'], validation['prediction'],
                                'training_evidence.json', 'stage2.log', 'validation.json',
                                'execution.json', 'validation.log']
                    if not job['fixture']:
                        required.append(str(Path(validation['prediction']).with_name(f"{task['seed']}_metric.csv")))
                    for rel in required:
                        src = safe_path(out/rel)
                        dst = safe_path(export/rel)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst)
                    files = inventory(export)
                    atomic(export/'result_manifest.json',{'task_id':key,'files':files,'fixture':job['fixture']})
                    state['tasks'][key].update(status='ready', result=str(export.relative_to(root)))
                except Exception as e:
                    state['tasks'][key].update(status='failed',error=f'{type(e).__name__}: {e}')
                atomic(root/'worker_state.json',state)
                wait_acceptance(root, job, task)
            state['status'] = 'finished'
        except Exception as e:
            state.update(status='failed',error=f'{type(e).__name__}: {e}')
        finally:
            atomic(root/'worker_state.json',state)


def start(root,retry_failed=False):
    with lock(root/'launch.lock'):
        s = status(root)
        if s.get('device_busy') and not s['active']:
            raise Phase3Error('device busy or orphan child still holds lease; refuse duplicate launch')
        if s['active'] or (s['status']=='finished' and not retry_failed):
            return s
        with (root/'worker.log').open('a') as log:
            cmd = [sys.executable,str(Path(__file__).resolve()),'work','--root',str(root)]
            if retry_failed:
                cmd.append('--retry-failed')
            subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                             start_new_session=True,close_fds=True)
        # Launch is acknowledged only after the child acquires its durable lock.
        for _ in range(50):
            s = status(root)
            if s['active'] or s['status']=='finished':
                return s
            time.sleep(.1)
        raise Phase3Error('launch outcome uncertain; inspect status before any retry')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['identity','probe','start','work','status','verify'])
    p.add_argument('--root',type=Path)
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if a.action=='identity':
        value=machine_identity()
    elif a.action=='probe':
        try:
            from .runtime import environment
        except ImportError:
            from runtime import environment
        value={**machine_identity(),'environment':environment(),
               'disk_free':shutil.disk_usage(a.root if a.root.exists() else a.root.parent).free}
    elif a.action=='start':
        value=start(a.root,a.retry_failed)
    elif a.action=='work':
        work(a.root,a.retry_failed); value=status(a.root)
    elif a.action=='status':
        value=status(a.root)
    else:
        job=read(a.root/'job.json')
        check_identity(job['machine_identity'])
        verify_inventory(a.root,job['inputs'],exact=False)
        value={'verified':True, 'job_sha256':digest(a.root/'job.json')}
    print(json.dumps(value))

if __name__=='__main__':
    main()
