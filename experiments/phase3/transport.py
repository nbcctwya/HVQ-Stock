"""Local and SSH transports. Only Remote has a power-off request capability."""
import json
import os
from pathlib import Path
import shlex
import subprocess

try:
    from .common import Phase3Error, inventory, publish, safe_path, under, verify_inventory
    from .worker import machine_identity
except ImportError:
    from common import Phase3Error, inventory, publish, safe_path, under, verify_inventory
    from worker import machine_identity


class Local:
    def __init__(self, config):
        self.config=config
        self.python=config['python']
        self.root=safe_path(config['work_root'])

    def result_path(self, root, relative):
        return under(root,relative)

    def identity(self):
        return machine_identity()

    def command(self,args,timeout=60):
        r=subprocess.run(args,capture_output=True,text=True,check=True,timeout=timeout)
        return r.stdout

    def stage(self,source,dest):
        # Copy only missing immutable files. Existing different bytes are a conflict.
        source,dest=Path(source),safe_path(dest)
        dest.mkdir(parents=True,exist_ok=True)
        for rel,info in inventory(source).items():
            src=source/rel; dst=safe_path(dest/rel)
            if dst.exists():
                verify_inventory(dest,{rel:info},exact=False)
            else:
                dst.parent.mkdir(parents=True,exist_ok=True)
                with src.open('rb') as f, dst.open('xb') as out:
                    import shutil
                    shutil.copyfileobj(f,out)
        verify_inventory(dest,inventory(source),exact=False)

    def pull(self,source,dest):
        publish(source,dest)

    def worker(self,root,action,retry=False):
        args=[self.python,str(Path(root)/'tool/worker.py'),action,'--root',str(root)]
        if retry: args.append('--retry-failed')
        return json.loads(self.command(args))


class Remote(Local):
    def __init__(self,config):
        self.config=config
        self.python=config['python']
        self.root=Path(config['work_root'])  # remote path; never stat it on the coordinator
        self.alias=config['ssh_alias']
        self.ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                  '-o','ConnectTimeout=15',self.alias]

    def result_path(self,root,relative):
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise Phase3Error('remote result path escapes job root')
        return Path(root)/rel

    def command(self,args,timeout=60):
        r=subprocess.run([*self.ssh,shlex.join([str(x) for x in args])],
                         capture_output=True,text=True,check=True,timeout=timeout)
        return r.stdout

    def identity(self):
        script='import json,platform,pathlib; print(json.dumps({"hostname":platform.node(),"machine_id":pathlib.Path("/etc/machine-id").read_text().strip()}))'
        return json.loads(self.command([self.python,'-B','-c',script]))

    def stage(self,source,dest):
        # Dedicated namespace + ignore-existing, followed by worker hash verification.
        # Never --delete, --inplace, or a project-root upload.
        self.command([self.python,'-c','import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert not any(x.is_symlink() for x in [p,*p.parents]); p.mkdir(parents=True,exist_ok=True)',str(dest)])
        subprocess.run(['rsync','-r','--ignore-existing','--partial-dir=.rsync-partial','--protect-args','-e',
                        shlex.join(self.ssh[:-1]),str(source)+'/',self.alias+':'+str(dest)+'/'],
                       check=True,timeout=1800)

    def pull(self,source,dest):
        dest=safe_path(dest); dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-r','--ignore-existing','--partial-dir=.rsync-partial','--protect-args','-e',
                        shlex.join(self.ssh[:-1]),self.alias+':'+str(source)+'/',str(dest)+'/'],
                       check=True,timeout=1800)

    def shutdown(self,expected,dry_run=True):
        actual=self.identity()
        if actual!=expected or actual==machine_identity() or actual['machine_id']==machine_identity()['machine_id']:
            raise Phase3Error('remote shutdown identity mismatch/local target forbidden')
        if dry_run:
            return 'dry_run'
        if self.config.get('shutdown')!='ssh_poweroff':
            raise Phase3Error('remote shutdown capability not enabled')
        # A disconnect is NOT proof of power-off. No generic cloud provider status
        # API is assumed. Persist intent BEFORE invoking this method.
        try:
            self.command(['sudo','-n','/sbin/shutdown','-h','now'],timeout=20)
        except (subprocess.SubprocessError,OSError):
            return 'requested_unconfirmed'
        return 'requested_unconfirmed'


def make_transport(config):
    return Remote(config) if config['kind']=='remote' else Local(config)
