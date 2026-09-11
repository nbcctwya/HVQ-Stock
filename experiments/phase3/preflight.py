"""Resolve explicit batches against canonical metadata, never mutate queue."""
import ast
import io
import re
import subprocess
import tarfile
from pathlib import Path
import yaml
try:
    from .common import Phase3Error, atomic, digest, identity, inventory, read, safe_path, sealed, under
except ImportError:
    from common import Phase3Error, atomic, digest, identity, inventory, read, safe_path, sealed, under


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def load_specs(batch_file,machine_file):
    batch=yaml.safe_load(Path(batch_file).read_text())
    machines=yaml.safe_load(Path(machine_file).read_text())['machines']
    if batch.get('schema')!=1 or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',str(batch.get('batch_id',''))):
        raise Phase3Error('invalid batch schema/id')
    if not isinstance(batch.get('devices'),list) or not batch['devices']:
        raise Phase3Error('explicit nonempty devices required')
    devices=set(); experiments=set(); identities=set()
    for device in batch['devices']:
        name=device['machine']
        if name in devices or name not in machines:
            raise Phase3Error('duplicate/unknown device')
        devices.add(name)
        m=machines[name]
        if m.get('kind') not in ('local','remote'):
            raise Phase3Error('unknown machine kind')
        if m['kind']=='local' and m.get('shutdown','disabled')!='disabled':
            raise Phase3Error('local has no shutdown capability')
        for field in ('python','work_root'):
            if not Path(m[field]).is_absolute() or '..' in Path(m[field]).parts:
                raise Phase3Error('machine paths must be absolute')
        if m['work_root'] in ('/','/root','/home'):
            raise Phase3Error('dedicated work_root required')
        if m['kind']=='remote' and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',m.get('ssh_alias','')):
            raise Phase3Error('invalid SSH alias')
        if m.get('shutdown','disabled') not in ('disabled','ssh_poweroff'):
            raise Phase3Error('unsupported shutdown backend')
        if set(m.get('identity',{}))!={'hostname','machine_id'} or not all(m['identity'].values()):
            raise Phase3Error('pinned machine identity required')
        machine_key=tuple(sorted(m['identity'].items()))
        if machine_key in identities:raise Phase3Error('duplicate physical device assignment')
        identities.add(machine_key)
        if not re.fullmatch(r'[0-9]+',str(m.get('gpu','0'))):
            raise Phase3Error('one GPU ordinal required')
        if not device.get('experiments'):
            raise Phase3Error('empty device queue')
        for exp in device['experiments']:
            key=str(exp['id'])
            if not (key=='baseline' or re.fullmatch('[0-9]{3}',key)) or key in experiments:
                raise Phase3Error('duplicate assignment/invalid experiment id')
            experiments.add(key)
            seeds=exp.get('seeds',batch.get('seeds',[1,2,3,4]))
            if not isinstance(seeds,list) or not seeds or any(type(s)!=int or s<=0 for s in seeds) or len(set(seeds))!=len(seeds):
                raise Phase3Error('seeds must be ordered unique positive integers; seed0 is read-only')
    return batch,machines


def snapshot(repo,commit,dest):
    dest=safe_path(dest)
    if dest.exists():
        return
    data=git(repo,'archive','--format=tar',commit)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            p=under(dest,member.name)
            if member.isdir():p.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as out:out.write(archive.extractfile(member).read())
            else:raise Phase3Error('unsupported link/special file in frozen code')


def marker(path):
    return dict(line.split('=',1) for line in Path(path).read_text().splitlines() if '=' in line)


def stage1(repo,key,queue,seen=None):
    seen=set() if seen is None else seen
    if key in seen:raise Phase3Error('stage1 provenance cycle')
    seen.add(key)
    e=queue[key]; run=repo/'artifacts'/key/'run'
    m=marker(run/'.stage1.done')
    if m.get('commit')!=e['commit']:raise Phase3Error('stage1 marker commit mismatch')
    path=Path(m['best'])
    path=path if path.is_absolute() else run/'checkpoints'/path.name
    path=safe_path(path)
    if not path.is_file() or not path.stat().st_size:raise Phase3Error('missing stage1 checkpoint')
    src=str(e.get('stage1_source','self'))
    if src=='external':
        expected=Path(e['stage1_ckpt']); expected=expected if expected.is_absolute() else repo/expected
        if path!=expected.absolute() or m.get('source')!='external':raise Phase3Error('external stage1 mismatch')
    elif src!='self':
        other=stage1(repo,src,queue,seen)
        if m.get('source')!=src or digest(other)!=digest(path):raise Phase3Error('reused stage1 mismatch')
    elif m.get('reused')=='true':raise Phase3Error('self stage1 marked reused')
    return path


def protocol_supported(code):
    source=(code/'stage2.py').read_text()
    tree=ast.parse(source)
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    if ('trainer.fit' not in calls or 'pl.seed_everything' not in calls
            or 'train.seed' not in source or not (code/'dataset/schema.py').exists()
            or 'dataset_basename' not in source or 'apply_artifact_root' not in source):
        raise Phase3Error('unsupported: requires canonical VQ independent Stage 2 Trainer.fit protocol')
    # Runtime audit is the definitive fit/seed check; this rejects known inference-only families early.


def resolve(repo,batch,workspace):
    queue_data=yaml.safe_load(git(repo,'show','main:experiments/queue.yaml'))
    queue={e['id']:e for e in queue_data['experiments']}
    result={}
    for device in batch['devices']:
        for spec in device['experiments']:
            key=str(spec['id'])
            run=safe_path(repo/'artifacts'/key/'run')
            if key=='baseline':
                b=batch.get('baseline',{})
                commit=b.get('commit','')
                if not re.fullmatch('[0-9a-f]{40}',commit):
                    raise Phase3Error('baseline requires exact full commit; branch main is not provenance')
                report=b.get('compatibility_report')
                if not report:raise Phase3Error('baseline blocked: seed0/code/data compatibility report required')
                proof=read(repo/report)
                if proof.get('commit')!=commit or proof.get('status')!='pass' or not proof.get('seed0_tree_sha256'):
                    raise Phase3Error('baseline compatibility evidence incomplete')
                ckpt=safe_path(repo/b['stage1_ckpt'])
                record=repo/'baseline_results/CORRECTED_PROTOCOL.md'
                if not record.is_file():raise Phase3Error('baseline record missing')
            else:
                if key not in queue:raise Phase3Error('unknown experiment')
                e=queue[key]; commit=e['commit']
                if e['status']!='done' or git(repo,'rev-parse',e['branch']).decode().strip()!=commit:
                    raise Phase3Error('experiment not done/frozen')
                record=repo/'experiments/records'/f'{key}-{e["name"]}.md'
                text=record.read_text()
                if commit not in text or not re.search(r'Status: PASS',text) or not re.search(r'Status: DONE',text):
                    raise Phase3Error('record missing commit/smoke/seed0 DONE')
                for name in ('.stage2.done','.backtest.done'):
                    if marker(run/name).get('commit')!=commit:raise Phase3Error('seed0 marker mismatch')
                if marker(run/'.stage2.done').get('seed')!='0':raise Phase3Error('seed0 marker seed mismatch')
                ckpt=stage1(repo,key,queue)
            code=workspace/'code'/key
            snapshot(repo,commit,code)
            protocol_supported(code)
            cfg=yaml.safe_load((code/'configs/config.yaml').read_text())
            u=cfg['data']['universe']
            if u not in ('csi300','sp500') or cfg['data']['window_size']!=20 or cfg['vqvae']['predictor']['pred_len']!=10:
                raise Phase3Error('unsupported dataset contract')
            pred=list(run.glob('res/*/0_best.pkl'))
            if len(pred)!=1 or not (pred[0].parent/'0_metric.csv').is_file():raise Phase3Error('missing/ambiguous seed0 result')
            backtests=list(pred[0].parent.glob('backtest/seed0_top30_drop5/portfolio_metric.csv'))
            if len(backtests)!=1:raise Phase3Error('missing seed0 backtest')
            baseline_inventory=inventory(run,allow_symlinks=True)
            if key=='baseline' and identity(baseline_inventory)!=proof['seed0_tree_sha256']:
                raise Phase3Error('baseline proof does not match seed0 tree')
            base=f'{u}_20_h10'; region='CN' if u=='csi300' else 'US'
            paths={split:f'{region}/{base}_{split}.pkl' for split in ('train','valid','test')}
            local_data=Path(cfg['data']['data_path'])
            local_data=local_data if local_data.is_absolute() else repo/local_data
            if not all((local_data/p).is_file() for p in paths.values()):raise Phase3Error('reference data missing')
            jkp=repo/'dataset/jkpdata'/f'[{"chn" if u=="csi300" else "usa"}]_[all_themes]_[daily]_[vw_cap].csv'
            if not jkp.is_file():raise Phase3Error('raw JKP input missing')
            result[key]={'experiment':key,'commit':commit,'code_local':str(code),
                         'stage1_local':str(ckpt),'stage1_sha256':digest(ckpt),'jkp_local':str(jkp),
                         'jkp_sha256':digest(jkp),'universe':u,'data_files':paths,
                         'reference_data':str(local_data),'seed0_tree':baseline_inventory,
                         'seed0_prediction':str(pred[0]),'seed0_metric':str(pred[0].parent/'0_metric.csv'),
                         'seed0_backtest':str(backtests[0]),'record_sha256':digest(record),
                         'code_inventory':inventory(code)}
    return result
