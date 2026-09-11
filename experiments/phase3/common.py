"""Small filesystem primitives shared by coordinator and detached worker."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


class Phase3Error(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_path(path):
    """Reject symlink ancestors too; resolve() alone hides them."""
    p = Path(os.path.abspath(path))
    for part in (p, *p.parents):
        if part.is_symlink():
            raise Phase3Error(f'symlink forbidden: {part}')
    return p


def under(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise Phase3Error(f'path escapes root: {relative}')
    p = safe_path(Path(root) / rel)
    if not p.is_relative_to(safe_path(root)):
        raise Phase3Error('path escapes root')
    return p


def atomic(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def lock(path):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise Phase3Error(f'already locked: {path}') from e
        try:
            yield f.fileno()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def inventory(root, allow_symlinks=False):
    root = safe_path(root)
    if not root.is_dir():
        raise Phase3Error(f'missing directory: {root}')
    result = {}
    for p in sorted(root.rglob('*')):
        if allow_symlinks and p.is_symlink():
            result[str(p.relative_to(root))] = {'symlink': os.readlink(p)}
            continue
        safe_path(p)
        if p.is_file():
            result[str(p.relative_to(root))] = {'sha256': digest(p), 'size': p.stat().st_size}
        elif not p.is_dir():
            raise Phase3Error(f'not a regular file: {p}')
    return result


def verify_inventory(root, expected, exact=True, allow_symlinks=False):
    if exact:
        if inventory(root, allow_symlinks=allow_symlinks) != expected:
            raise Phase3Error(f'hash/file inventory conflict: {root}')
    else:
        for rel, info in expected.items():
            p = under(root, rel)
            if not p.is_file() or p.stat().st_size != info['size'] or digest(p) != info['sha256']:
                raise Phase3Error(f'hash conflict: {p}')


def publish(source, destination):
    """No overwrite, including on a retry. Call only under coordinator/device lock."""
    source, destination = safe_path(source), safe_path(destination)
    expected = inventory(source)
    if destination.exists():
        verify_inventory(destination, expected)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.incoming-', dir=destination.parent))
    try:
        shutil.copytree(source, tmp, dirs_exist_ok=True)
        verify_inventory(tmp, expected)
        for p in tmp.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:
                    os.fsync(f.fileno())
        # Our callers hold the publication lock; never merge with existing output.
        if destination.exists():
            raise Phase3Error(f'publication conflict: {destination}')
        os.rename(tmp, destination)
        sync_dir(destination.parent)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def sealed(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise Phase3Error(f'immutable manifest conflict: {path}')
    else:
        atomic(path, value)
