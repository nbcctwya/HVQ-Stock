"""Subprocess helpers. Training audit observes Lightning; frozen sources stay untouched."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import runpy
import sys

try:
    from .common import Phase3Error, atomic, digest, identity, read
except ImportError:
    from common import Phase3Error, atomic, digest, identity, read


def environment(gpu=False):
    result = {'python': platform.python_version(), 'hostname': platform.node(),
              'packages': {n: importlib.metadata.version(n) for n in
                           ('torch', 'pytorch-lightning', 'pyqlib', 'numpy', 'pandas',
                            'hydra-core', 'omegaconf')}}
    if gpu:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        if not result['cuda_available']:
            raise Phase3Error('GPU unavailable; Phase 3 never starts a GPU instance')
        result['gpu'] = torch.cuda.get_device_name(0)
        result['cuda_build'] = torch.version.cuda
        result['gpu_bytes'] = torch.cuda.get_device_properties(0).total_memory
    return result


def fingerprint(paths, code):
    """Hash every actual sampled window, not pickle bytes or a sample of rows.

    CanonicalSampler overrides __getitem__ (market windows), so hashing data_arr
    alone misses real inputs. Batch-vectorized iteration bounds memory usage.
    NaNs and signed zero are normalized; index order and dtypes remain explicit.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(code).resolve()))
    result = {}
    for split, path in paths.items():
        with open(path, 'rb') as f:
            ds = pickle.load(f)
        index = ds.get_index()
        if not index.is_unique or len(index) != len(ds) or not len(ds):
            raise Phase3Error('invalid dataset index')
        columns = getattr(ds, 'columns', None)
        if columns is None:
            raise Phase3Error('unsupported dataset: missing column order metadata')
        h = hashlib.sha256()
        # pandas hash is stable within the pinned pandas version, recorded below.
        h.update(pd.util.hash_pandas_object(index, index=True).to_numpy(dtype='<u8').tobytes())
        metadata = {'n': len(ds), 'index_names': list(index.names),
                    'columns': [list(x) if isinstance(x, tuple) else str(x) for x in columns],
                    'pandas': pd.__version__, 'sample_shape': list(ds[0].shape),
                    'dtype': str(ds[0].dtype), 'split': split}
        h.update(identity(metadata).encode())
        for start in range(0, len(ds), 128):
            a = np.asarray(ds[list(range(start, min(start + 128, len(ds))))], dtype='<f8').copy()
            a[a == 0] = 0
            a[np.isnan(a)] = np.nan
            h.update(a.tobytes(order='C'))
        result[split] = {'sha256': h.hexdigest(), **metadata}
        del ds
    return result


def verify_stage1_state(model, checkpoint):
    """Reject permissive strict=False loads that silently left random Stage 1 weights."""
    import torch
    obj = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = obj.get('state_dict', obj)
    for name, prefix in (('encoder', 'vqvae.spatial_encoder.'),
                         ('quantizer', 'vqvae.quantizer.'), ('revin', 'vqvae.revin.')):
        module = getattr(model, name, None)
        if module is None:
            raise Phase3Error(f'unsupported Stage 1 module: {name}')
        current = module.state_dict()
        expected = {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}
        if set(current) != set(expected):
            raise Phase3Error(f'Stage 1 key mismatch: {name}')
        for k,v in current.items():
            if not torch.equal(v.detach().cpu(), expected[k].detach().cpu()):
                raise Phase3Error(f'Stage 1 tensor mismatch: {name}.{k}')
        if any(p.requires_grad for p in module.parameters()):
            raise Phase3Error(f'Stage 1 not frozen: {name}')
    return True


def train(task, root, out):
    import torch
    import pytorch_lightning as pl
    code = root / task['code']
    seed = task['seed']
    checkpoint = root / task['stage1']
    evidence = {'requested_seed': seed, 'actual_training_seed': None,
                'commit': task['commit'], 'stage1_sha256': digest(checkpoint),
                'data_fingerprint': task['data_fingerprint'], 'entrypoint': 'stage2.py',
                'task_id': task['task_id'], 'fixture': False,
                'environment': environment(gpu=True)}
    original_fit = pl.Trainer.fit
    fits = []

    def audited_fit(trainer, model, *args, **kwargs):
        cfg = model.config
        actual = int(cfg['train']['seed'])
        if (actual != seed or torch.initial_seed() != seed
                or os.environ.get('PL_GLOBAL_SEED') != str(seed)):
            raise Phase3Error('actual training seed mismatch')
        if kwargs.get('ckpt_path') is not None or len(args) > 2:
            raise Phase3Error('warm-start training is unsupported')
        if fits:
            raise Phase3Error('expected exactly one independent fit')
        fits.append(True)
        evidence['stage1_verified'] = verify_stage1_state(model, checkpoint)
        # Capture resolved configuration from the real module, not CLI filenames.
        evidence.update(actual_training_seed=actual, torch_initial_seed=torch.initial_seed(),
                        pl_global_seed=os.environ['PL_GLOBAL_SEED'], config=cfg,
                        T_max=task['T_max'] if 'T_max' in task else None,
                        fit_started=True, fit_completed=False)
        atomic(out / 'training_evidence.json', evidence)
        result = original_fit(trainer, model, *args, **kwargs)
        verify_stage1_state(model, checkpoint)
        best = Path(trainer.checkpoint_callback.best_model_path)
        if trainer.global_step <= 0 or not best.is_file() or not best.resolve().is_relative_to(out.resolve()):
            raise Phase3Error('fit did not produce a trained checkpoint')
        evidence.update(fit_completed=True, global_step=int(trainer.global_step),
                        best_checkpoint=str(best.relative_to(out)), checkpoint_sha256=digest(best))
        atomic(out / 'training_evidence.json', evidence)
        return result

    pl.Trainer.fit = audited_fit
    sys.path.insert(0, str(code))
    os.chdir(code)
    sys.argv = [str(code / 'stage2.py'), f'train.seed={seed}',
                f'artifact_root={out}', f'predictor.saved_model={json.dumps(str(checkpoint))}',
                f'data.data_path={root / task["data"]}',
                f'hydra.run.dir={out / "hydra"}']
    try:
        runpy.run_path(str(code / 'stage2.py'), run_name='__main__')
    finally:
        pl.Trainer.fit = original_fit
    if not evidence.get('fit_completed'):
        raise Phase3Error('unsupported: entrypoint never completed Trainer.fit')


def fixture(task, out):
    """Explicit CPU transport/lifecycle test only; forbidden in formal acceptance."""
    import torch
    import pandas as pd
    torch.manual_seed(task['seed'])
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        model(torch.ones(2, 2)).square().mean().backward()
        opt.step()
    torch.save({'state_dict': model.state_dict(), 'global_step': 2}, out / 'model.ckpt')
    idx = pd.MultiIndex.from_product([pd.date_range('2025-01-01', periods=2), ['A','B','C']],
                                    names=['datetime','instrument'])
    pd.DataFrame({'score': [1.,2.,3.,3.,2.,1.], 'label': [1.,2.,3.,3.,2.,1.]}, index=idx).to_pickle(out/'prediction.pkl')
    evidence = {'fixture': True, 'requested_seed': task['seed'], 'actual_training_seed': task['seed'],
                'torch_initial_seed': task['seed'], 'pl_global_seed': str(task['seed']),
                'task_id': task['task_id'], 'commit': task['commit'],
                'entrypoint': 'cpu_fixture', 'fit_started': True, 'fit_completed': True,
                'global_step': 2, 'best_checkpoint': 'model.ckpt',
                'checkpoint_sha256': digest(out/'model.ckpt'),
                'stage1_sha256': task['stage1_sha256'], 'data_fingerprint': task['data_fingerprint']}
    atomic(out/'training_evidence.json', evidence)


def inspect_result(task, folder, fixture_allowed=False):
    import torch
    import numpy as np
    import pandas as pd
    evidence = read(folder / 'training_evidence.json')
    for k in ('task_id','commit','stage1_sha256','data_fingerprint'):
        if evidence.get(k) != task[k]:
            raise Phase3Error(f'provenance mismatch: {k}')
    seed = task['seed']
    if (evidence.get('fixture', False) != fixture_allowed
            or evidence.get('requested_seed') != seed or evidence.get('actual_training_seed') != seed
            or evidence.get('torch_initial_seed') != seed or evidence.get('pl_global_seed') != str(seed)
            or not evidence.get('fit_completed') or not evidence.get('fit_started')
            or evidence.get('global_step', 0) <= 0):
        raise Phase3Error('fake/unsupported training seed provenance')
    if not fixture_allowed:
        if not evidence.get('stage1_verified') or evidence.get('entrypoint') != 'stage2.py' or evidence.get('config',{}).get('train',{}).get('seed') != seed:
            raise Phase3Error('actual module config seed mismatch')
    try:
        from .common import under
    except ImportError:
        from common import under
    ckpt = under(folder, evidence['best_checkpoint'])
    if digest(ckpt) != evidence['checkpoint_sha256']:
        raise Phase3Error('checkpoint evidence hash mismatch')
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    if not obj.get('state_dict') or obj.get('global_step', 0) <= 0:
        raise Phase3Error('checkpoint missing trained state')
    if not all(torch.isfinite(v).all().item() for v in obj['state_dict'].values() if torch.is_tensor(v)):
        raise Phase3Error('nonfinite checkpoint')
    pred = folder/'prediction.pkl' if fixture_allowed else None
    if pred is None:
        hits = list(folder.glob(f'res/*/{seed}_best.pkl'))
        if len(hits) != 1:
            raise Phase3Error('missing/ambiguous prediction')
        pred = hits[0]
        metric = pred.parent / f'{seed}_metric.csv'
        m = pd.read_csv(metric, index_col=0)
        if not {'IC','ICIR','RankIC','RankICIR'}.issubset(m.index) or not np.isfinite(m.loc[['IC','ICIR','RankIC','RankICIR']].to_numpy()).all():
            raise Phase3Error('invalid metrics')
    frame = pd.read_pickle(pred)
    if (not isinstance(frame, pd.DataFrame) or not {'score','label'}.issubset(frame.columns)
            or not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ['datetime','instrument']
            or not frame.index.is_unique or not len(frame) or not np.isfinite(frame['score']).all()):
        raise Phase3Error('invalid prediction schema')
    return {'checkpoint': str(ckpt.relative_to(folder)), 'prediction': str(pred.relative_to(folder)),
            'rows': len(frame), 'evidence_sha256': digest(folder/'training_evidence.json')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fingerprint','train','inspect','environment','fixture','compare_prediction'])
    p.add_argument('--request', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.action == 'environment':
        result = environment()
    else:
        req = read(a.request)
        if a.action == 'fingerprint':
            result = fingerprint(req['paths'], req['code'])
        elif a.action == 'compare_prediction':
            import pandas as pd
            x = pd.read_pickle(req['prediction']).sort_index()
            y = pd.read_pickle(req['reference']).sort_index()
            if not x.index.equals(y.index) or not x['label'].equals(y['label']):
                raise Phase3Error('seed0/new-seed prediction index or labels differ')
            result = {'matched': True, 'rows': len(x)}
        elif a.action == 'inspect':
            result = inspect_result(req['task'], Path(req['folder']), req.get('fixture',False))
        else:
            out = Path(req['out']).resolve()
            out.mkdir(parents=True, exist_ok=True)
            if a.action == 'train':
                train(req['task'], Path(req['root']).resolve(), out)
            else:
                fixture(req['task'], out)
            result = {'ok': True}
    if a.output:
        atomic(a.output, result)
    else:
        print(json.dumps(result))

if __name__ == '__main__':
    main()
