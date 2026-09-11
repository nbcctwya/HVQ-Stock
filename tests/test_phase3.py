"""Phase 3 invariants; all model work here is tiny synthetic CPU fixtures."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import yaml

from experiments.phase3.common import Phase3Error, atomic, digest, identity, inventory, publish, read, verify_inventory
from experiments.phase3.coordinator import Coordinator, aggregate_rows, diagnostic
from experiments.phase3.preflight import load_specs, protocol_supported
from experiments.phase3.runtime import fixture, inspect_result
from experiments.phase3.transport import Local, Remote
from experiments.phase3.worker import machine_identity


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.machine={'kind':'local','python':sys.executable,'work_root':str(self.root/'worker'),
                      'identity':machine_identity(),'gpu':'0','shutdown':'disabled'}
        self.batch={'schema':1,'batch_id':'test','local_python':sys.executable,'seeds':[1,2],
                    'devices':[{'machine':'local','experiments':[{'id':'010'},{'id':'019'}]}]}
    def tearDown(self):self.tmp.cleanup()
    def specs(self,b=None,m=None):
        bp=self.root/'batch.yaml';mp=self.root/'machines.yaml'
        bp.write_text(yaml.safe_dump(b or self.batch));mp.write_text(yaml.safe_dump({'machines':m or {'local':self.machine}}))
        return load_specs(bp,mp)
    def task(self,seed=1):
        t={'experiment':'010','seed':seed,'commit':'fixture','stage1_sha256':'fixture',
           'data_fingerprint':{'fixture':'synthetic CPU'}}
        t['task_id']=identity(t);return t
    def result(self,seed=1):
        t=self.task(seed);out=self.root/f'result{seed}';out.mkdir()
        fixture(t,out)
        atomic(out/'result_manifest.json',{'task_id':t['task_id'],'fixture':True,'files':inventory(out)})
        return t,out
    def test_ordered_spec_and_override(self):
        b=copy.deepcopy(self.batch);b['devices'][0]['experiments'][0]['seeds']=[4,1,3]
        got,_=self.specs(b)
        self.assertEqual(got['devices'][0]['experiments'][0]['seeds'],[4,1,3])
        self.assertEqual([e['id'] for e in got['devices'][0]['experiments']],['010','019'])
    def test_duplicate_assignment_and_seed0(self):
        b=copy.deepcopy(self.batch);b['devices'][0]['experiments'].append({'id':'010'})
        with self.assertRaisesRegex(Phase3Error,'duplicate'):self.specs(b)
        for seeds in ([0,1],[1,1],[True],[]):
            b=copy.deepcopy(self.batch);b['seeds']=seeds
            with self.assertRaises(Phase3Error):self.specs(b)
    def test_local_power_capability_absent(self):
        self.assertFalse(hasattr(Local(self.machine),'shutdown'))
        m={**self.machine,'shutdown':'ssh_poweroff'}
        with self.assertRaisesRegex(Phase3Error,'local'):self.specs(m={'local':m})
    def test_remote_identity_shutdown_guard(self):
        m={**self.machine,'kind':'remote','ssh_alias':'test','shutdown':'ssh_poweroff'}
        r=Remote(m)
        with patch.object(r,'identity',return_value={'hostname':'wrong','machine_id':'wrong'}), patch.object(r,'command') as command:
            with self.assertRaises(Phase3Error):r.shutdown(machine_identity(),dry_run=False)
            command.assert_not_called()
        with patch.object(r,'identity',return_value=machine_identity()):
            with self.assertRaises(Phase3Error):r.shutdown(machine_identity(),dry_run=True)
    def test_unsupported_inference_entry(self):
        c=self.root/'code';c.mkdir();(c/'stage2.py').write_text('print("inference only")')
        with self.assertRaisesRegex(Phase3Error,'unsupported'):protocol_supported(c)
    def test_publish_conflict_symlink_and_seed0_immutable(self):
        seed0=self.root/'artifacts/010/run';seed0.mkdir(parents=True);(seed0/'0_best.pkl').write_bytes(b'original')
        before=inventory(seed0)
        source=self.root/'src';source.mkdir();(source/'x').write_bytes(b'new')
        dest=self.root/'artifacts/010/phase3/b/seed1'
        publish(source,dest);publish(source,dest)
        (source/'x').write_bytes(b'different')
        with self.assertRaisesRegex(Phase3Error,'conflict'):publish(source,dest)
        linked=self.root/'linked';linked.symlink_to(seed0,target_is_directory=True)
        with self.assertRaisesRegex(Phase3Error,'symlink'):publish(source,linked/'seed1')
        self.assertEqual(before,inventory(seed0))
    def test_acceptance_and_resume_receipt(self):
        task,out=self.result()
        c=Coordinator(self.root,self.batch,{'local':self.machine},fixture=True)
        r=c.accept(task,out)
        mtime=(Path(r['archive'])/'model.ckpt').stat().st_mtime_ns
        other=Coordinator(self.root,self.batch,{'local':self.machine},fixture=True)
        self.assertTrue(other.load_receipt(task))
        other.accept(task,out)
        self.assertEqual(mtime,(Path(r['archive'])/'model.ckpt').stat().st_mtime_ns)
    def test_fake_seed_and_fixture_never_formal(self):
        t,out=self.result()
        with self.assertRaisesRegex(Phase3Error,'provenance'):inspect_result(t,out,False)
        e=read(out/'training_evidence.json');e['actual_training_seed']=2;atomic(out/'training_evidence.json',e)
        with self.assertRaisesRegex(Phase3Error,'provenance'):inspect_result(t,out,True)
    def test_hash_conflict_missing_file_not_accepted(self):
        t,out=self.result();(out/'model.ckpt').write_bytes(b'corrupt')
        c=Coordinator(self.root,self.batch,{'local':self.machine},fixture=True)
        with self.assertRaisesRegex(Phase3Error,'conflict'):c.accept(t,out)
        self.assertFalse((c.root/'receipts'/f'{t["task_id"]}.json').exists())
    def test_notification_failure_does_not_change_receipt(self):
        t,out=self.result();c=Coordinator(self.root,self.batch,{'local':self.machine},fixture=True)
        c.accept(t,out);before=inventory(c.root/'receipts')
        c.fixture=False;c.notify_enabled=True
        with patch('experiments.phase3.coordinator.subprocess.run',side_effect=OSError('SMTP down')):
            c.event('experiment-010','experiment multi-seed completed','accepted')
        self.assertEqual(before,inventory(c.root/'receipts'))
        self.assertEqual(read(c.root/'events/experiment-010.json')['status'],'failed')
    def test_aggregate_sample_std_and_n(self):
        out=aggregate_rows([{'metrics':{'IC':1}},{'metrics':{'IC':3}}])
        self.assertEqual(out['IC']['n'],2);self.assertAlmostEqual(out['IC']['std'],2**.5)
    def test_shutdown_requires_all_receipts(self):
        c=Coordinator(self.root,self.batch,{'remote':{**self.machine,'kind':'remote'}},fixture=True)
        remote=Remote({**self.machine,'kind':'remote','ssh_alias':'x'})
        with patch.object(remote,'shutdown') as power:
            with self.assertRaises(Phase3Error):c.finish_device('remote',{'tasks':[self.task()]},remote)
            power.assert_not_called()
    def test_two_devices_can_work_concurrently(self):
        gate=threading.Barrier(2);done=[]
        def run(name):gate.wait(timeout=3);done.append(name)
        with ThreadPoolExecutor(max_workers=2) as p:
            futures=[p.submit(run,n) for n in ('A','B')]
            for f in futures:f.result()
        self.assertEqual(set(done),{'A','B'})
    def test_real_local_worker_multiple_experiments_and_resume(self):
        result=diagnostic(self.root,'local',{'local':self.machine},'ordered',sys.executable)
        self.assertEqual(result['accepted'],4)
        run=Path(result['remote_root']);state=read(run/'worker_state.json');job=read(run/'job.json')
        self.assertEqual([(t['experiment'],t['seed']) for t in job['tasks']],
                         [('fixtureA',1),('fixtureA',2),('fixtureB',1),('fixtureB',2)])
        logs=[run/state['tasks'][t['task_id']]['result']/'training_evidence.json' for t in job['tasks']]
        times=[p.stat().st_mtime_ns for p in logs]
        self.assertEqual(times,sorted(times))
        again=diagnostic(self.root,'local',{'local':self.machine},'ordered',sys.executable)
        self.assertEqual(again['accepted'],4)
        self.assertEqual(times,[p.stat().st_mtime_ns for p in logs])
        events=[read(p)['kind'] for p in Path(result['root']).glob('events/*.json')]
        self.assertEqual(events.count('experiment multi-seed completed'),2)
        self.assertNotIn('remote shutdown',events)


class WindowDataset:
    """Synthetic dataset exercising the same vectorized sampler API."""
    def __init__(self):
        import numpy as np
        import pandas as pd
        self.values=np.arange(24,dtype='float32').reshape(4,2,3)
        self.columns=pd.MultiIndex.from_tuples([('feature','a'),('prior','b'),('label','c')])
        self.index=pd.MultiIndex.from_product([pd.date_range('2025-01-01',periods=2),['A','B']],names=['datetime','instrument'])
    def __len__(self):return len(self.values)
    def get_index(self):return self.index
    def __getitem__(self,i):return self.values[i]


class AdditionalTests(unittest.TestCase):
    setUp = Phase3Tests.setUp
    tearDown = Phase3Tests.tearDown
    task = Phase3Tests.task
    result = Phase3Tests.result

    def test_fingerprint_values_labels_index_columns_and_split(self):
        import pickle
        from experiments.phase3.runtime import fingerprint
        ds=WindowDataset();path=self.root/'data.pkl'
        def fp(obj,split='train'):
            path.write_bytes(pickle.dumps(obj))
            return fingerprint({split:str(path)},self.root)
        original=fp(ds)
        self.assertEqual(original,fp(ds))
        for mutate in ('feature','label','index','columns'):
            other=copy.deepcopy(ds)
            if mutate=='feature':other.values[1,0,0]+=1
            elif mutate=='label':other.values[1,1,2]+=1
            elif mutate=='index':other.index=other.index[::-1]
            else:other.columns=other.columns[::-1]
            self.assertNotEqual(original,fp(other))
        self.assertNotEqual(original,fp(ds,'valid'))

    def test_observed_real_fit_provenance_on_tiny_cpu_lightning_model(self):
        from experiments.phase3.runtime import train
        import textwrap
        code=self.root/'code';code.mkdir();out=self.root/'trained';out.mkdir()
        (self.root/'s1.ckpt').write_bytes(b'fixture-only-stage1')
        (code/'stage2.py').write_text(textwrap.dedent('''
            import sys
            from pathlib import Path
            import torch
            import pytorch_lightning as pl
            import pandas as pd
            from torch.utils.data import DataLoader,TensorDataset
            from pytorch_lightning.callbacks import ModelCheckpoint
            args=dict(x.split('=',1) for x in sys.argv[1:])
            seed=int(args['train.seed']);out=Path(args['artifact_root'])
            pl.seed_everything(seed,workers=True)
            class Tiny(pl.LightningModule):
                def __init__(self):
                    super().__init__();self.layer=torch.nn.Linear(2,1);self.config={'train':{'seed':seed}}
                def training_step(self,batch,batch_idx):return self.layer(batch[0]).square().mean()
                def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.01)
            cb=ModelCheckpoint(dirpath=out/'checkpoints',filename='best')
            trainer=pl.Trainer(max_epochs=1,accelerator='cpu',devices=1,logger=False,
                               enable_progress_bar=False,enable_model_summary=False,callbacks=[cb])
            trainer.fit(Tiny(),DataLoader(TensorDataset(torch.ones(4,2)),batch_size=2))
            res=out/'res/run';res.mkdir(parents=True)
            idx=pd.MultiIndex.from_product([pd.date_range('2025-01-01',periods=2),['A','B','C']],names=['datetime','instrument'])
            pd.DataFrame({'score':[1.,2.,3.,3.,2.,1.],'label':[1.,2.,3.,3.,2.,1.]},index=idx).to_pickle(res/f'{seed}_best.pkl')
            pd.DataFrame({'values':{'IC':.1,'ICIR':.2,'RankIC':.1,'RankICIR':.2}}).to_csv(res/f'{seed}_metric.csv')
        '''))
        t=self.task(3);t.update(code='code',stage1='s1.ckpt',data='data',stage1_sha256=digest(self.root/'s1.ckpt'))
        oldcwd=os.getcwd();argv=sys.argv[:];path=sys.path[:]
        try:
            with patch('experiments.phase3.runtime.environment',return_value={'fixture':'CPU test'}), \
                 patch('experiments.phase3.runtime.verify_stage1_state',return_value=True):
                train(t,self.root,out)
        finally:
            os.chdir(oldcwd);sys.argv=argv;sys.path[:]=path
        evidence=read(out/'training_evidence.json')
        self.assertEqual(evidence['actual_training_seed'],3)
        self.assertEqual(evidence['global_step'],2)
        self.assertTrue(evidence['fit_completed'])
        result=inspect_result(t,out,False)
        self.assertEqual(result['rows'],6)
        evidence['config']['train']['seed']=2;atomic(out/'training_evidence.json',evidence)
        with self.assertRaisesRegex(Phase3Error,'config seed'):inspect_result(t,out,False)

    def test_transfer_failure_then_resume_without_new_training_and_event_order(self):
        t,out=self.result();batch=copy.deepcopy(self.batch)
        batch['devices'][0]['experiments']=[{'id':'010','seeds':[1]}]
        c=Coordinator(self.root,batch,{'local':self.machine},fixture=True,poll_seconds=.01)
        deploy=c.root/'deploy/local';deploy.mkdir(parents=True);atomic(deploy/'job.json',{})
        actions=[]
        class Fake(Local):
            def __init__(self,m):super().__init__(m);self.launched=False;self.fail=True
            def worker(self,root,action,retry=False):
                if action=='verify':return {'job_sha256':digest(deploy/'job.json')}
                if action=='probe':return {**machine_identity(),'environment':{},'disk_free':10**12}
                if action=='start':
                    if not self.launched:actions.append('train');self.launched=True
                return {'active':False,'status':'finished','tasks':{t['task_id']:{'status':'ready','result':'result1','attempt':1}}}
            def pull(self,source,dest):
                actions.append('pull')
                if self.fail:self.fail=False;raise OSError('transfer interrupted')
                publish(out,dest)
            def stage(self,source,dest):actions.append('ack')
        tr=Fake(self.machine);job={'tasks':[t]}
        original_event=c.event
        def event(*args):actions.append('notify');return original_event(*args)
        c.event=event
        with self.assertRaises(OSError):c.device_loop(batch['devices'][0],job,tr,self.root)
        self.assertNotIn('notify',actions)
        c.device_loop(batch['devices'][0],job,tr,self.root)
        self.assertEqual(actions.count('train'),1)
        self.assertEqual(actions[-2:],['notify','ack'])
        self.assertEqual(len(c.accepted),1)

    def test_device_loops_concurrent_not_just_threadpool(self):
        tasks=[];sources={}
        for seed in (1,2):
            t,out=self.result(seed);tasks.append(t);sources[t['task_id']]=out
        gate=threading.Barrier(2);starts=[]
        c=Coordinator(self.root,self.batch,{'A':self.machine,'B':self.machine},fixture=True,poll_seconds=.01)
        class Fake(Local):
            def __init__(self,name,t):super().__init__(self_machine);self.name=name;self.t=t
            def worker(self,root,action,retry=False):
                if action=='verify':return {'job_sha256':digest(c.root/'deploy'/self.name/'job.json')}
                if action=='probe':return {**machine_identity(),'environment':{},'disk_free':10**12}
                if action=='start':starts.append(self.name);gate.wait(timeout=5)
                return {'active':False,'status':'finished','tasks':{self.t['task_id']:{'status':'ready','result':'x','attempt':1}}}
            def pull(self,source,dest):publish(sources[self.t['task_id']],dest)
            def stage(self,source,dest):pass
        self_machine=self.machine
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[]
            for name,t in zip(('A','B'),tasks):
                p=c.root/'deploy'/name;p.mkdir(parents=True);atomic(p/'job.json',{})
                d={'machine':name,'experiments':[{'id':'010','seeds':[t['seed']]}]}
                futures.append(pool.submit(c.device_loop,d,{'tasks':[t]},Fake(name,t),self.root))
            for f in futures:f.result()
        self.assertEqual(set(starts),{'A','B'})

    def test_shutdown_event_independent_and_dry_run_no_command(self):
        t,out=self.result();m={**self.machine,'kind':'remote','ssh_alias':'fake',
                              'identity':{'hostname':'remote','machine_id':'remote-id'},'shutdown':'ssh_poweroff'}
        c=Coordinator(self.root,self.batch,{'R':m},fixture=True);c.accept(t,out)
        c.fixture=False;c.event('experiment-010','experiment multi-seed completed','accepted; evaluation pending')
        tr=Remote(m)
        with patch.object(tr,'identity',return_value=m['identity']),patch.object(tr,'command') as command:
            c.finish_device('R',{'tasks':[t]},tr)
            command.assert_not_called()
        self.assertEqual(read(c.root/'power/R.json')['status'],'dry_run')
        c.allow_shutdown=True
        with patch.object(tr,'identity',return_value=m['identity']),patch.object(tr,'command',return_value=''):
            c.finish_device('R',{'tasks':[t]},tr)
        kinds=[read(p)['kind'] for p in (c.root/'events').glob('*.json')]
        self.assertIn('remote shutdown',kinds);self.assertIn('experiment multi-seed completed',kinds)
        self.assertEqual(read(c.root/'power/R.json')['status'],'requested_unconfirmed')

    def test_evaluation_recovery_is_local_after_device_accepted(self):
        t,out=self.result()
        c=Coordinator(self.root,self.batch,{'local':self.machine},fixture=True)
        c.accept(t,out)
        atomic(c.root/'devices/local.json',{'status':'accepted','tasks':[t['task_id']]})
        tr=Local(self.machine)
        with patch.object(c,'prepare'),patch.object(c,'build_job',return_value=(self.root,{'tasks':[t]})),\
             patch.object(tr,'identity',side_effect=AssertionError('offline machine contacted')),\
             patch.object(tr,'stage',side_effect=AssertionError('restaging accepted output')),\
             patch.object(c,'evaluate',side_effect=RuntimeError('backtest interrupted')):
            c.factory=lambda m:tr
            with self.assertRaisesRegex(RuntimeError,'backtest interrupted'):c.run()
        with patch.object(c,'prepare'),patch.object(c,'build_job',return_value=(self.root,{'tasks':[t]})),\
             patch.object(tr,'identity',side_effect=AssertionError('offline machine contacted')),\
             patch.object(c,'evaluate',return_value={'done':True}):
            self.assertEqual(c.run(),{'done':True})

    def test_remote_path_is_not_statted_locally(self):
        m={**self.machine,'kind':'remote','ssh_alias':'fake','work_root':'/root/unreadable/phase3'}
        remote=Remote(m)
        self.assertEqual(str(remote.result_path(remote.root,'attempts/a/1')),'/root/unreadable/phase3/attempts/a/1')
        with self.assertRaises(Phase3Error):remote.result_path(remote.root,'../../run')

    def test_seed0_wandb_symlink_is_fingerprinted_without_following(self):
        from experiments.phase3.common import safe_path
        tree=self.root/'run';tree.mkdir();(tree/'0_best.pkl').write_bytes(b'original')
        (tree/'debug.log').symlink_to('/does/not/exist')
        original=inventory(tree,allow_symlinks=True)
        verify_inventory(tree,original,allow_symlinks=True)
        self.assertEqual(original['debug.log'],{'symlink':'/does/not/exist'})
        with self.assertRaises(Phase3Error):inventory(tree)
        (tree/'debug.log').unlink();(tree/'debug.log').symlink_to('/different')
        with self.assertRaises(Phase3Error):verify_inventory(tree,original,allow_symlinks=True)

    def test_stage1_tensor_provenance_uses_real_checkpoint_prefixes(self):
        import torch
        from experiments.phase3.runtime import verify_stage1_state
        model=torch.nn.Module()
        state={}
        for attr,prefix in [('encoder','vqvae.spatial_encoder.'),('quantizer','vqvae.quantizer.'),('revin','vqvae.revin.')]:
            m=torch.nn.Linear(2,2);m.requires_grad_(False);setattr(model,attr,m)
            state.update({prefix+k:v.clone() for k,v in m.state_dict().items()})
        ck=self.root/'stage1.ckpt';torch.save({'state_dict':state},ck)
        self.assertTrue(verify_stage1_state(model,ck))
        with torch.no_grad():model.encoder.weight.add_(1)
        with self.assertRaisesRegex(Phase3Error,'tensor mismatch'):verify_stage1_state(model,ck)

    def test_worker_refuses_orphan_device_lease(self):
        from experiments.phase3.worker import start
        root=self.root/'job';root.mkdir()
        with patch('experiments.phase3.worker.status',return_value={'active':False,'device_busy':True,'status':'running'}), \
             patch('experiments.phase3.worker.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(Phase3Error,'orphan'):start(root,retry_failed=True)
            spawn.assert_not_called()

if __name__=='__main__':unittest.main()
