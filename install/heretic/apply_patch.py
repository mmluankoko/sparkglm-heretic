#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Apply the complete Heretic engine patch to the exact pinned Atlas source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def apply(engine):
    lock = json.loads((HERE / 'sources.lock.json').read_text())
    def git(*args):
        return subprocess.check_output(['git', '-C', str(engine), *args], text=True).strip()
    if (git('rev-parse', 'HEAD') != lock['upstream_commit'] or
            git('rev-parse', 'HEAD^{tree}') != lock['upstream_tree']):
        raise RuntimeError('Heretic requires the pinned Atlas commit and tree')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Heretic patch requires a clean engine checkout')
    patch = HERE / lock['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock['patch_sha256']:
        raise RuntimeError('Heretic patch SHA256 mismatch')
    git('apply', '--check', str(patch))
    git('apply', str(patch))
    for name, expected in lock['modified_files'].items():
        path = engine / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Patched source SHA256 mismatch: {name}')
    for name in lock.get('deleted_files', []):
        if (engine / name).exists():
            raise RuntimeError(f'Patched source was not deleted: {name}')
    print(f"Heretic: verified {len(lock['modified_files'])} engine files ({lock['patch_sha256']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', type=Path)
    apply(parser.parse_args().engine.resolve())
