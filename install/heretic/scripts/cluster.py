#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Launch the shared-base Heretic service using an explicit local configuration."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parents[1]
NODE = HERE / 'scripts/node.py'
DEFAULT_CONFIG = HERE.parents[1] / 'heretic.local.json'
DEFAULT_PROFILE = 'heretic-dflash'


def config(path, profile=DEFAULT_PROFILE):
    if profile not in ('stock', 'heretic', 'heretic-dflash'):
        raise ValueError('Unknown Heretic profile')
    cfg = json.loads(Path(path).read_text())
    for key in ('image', 'worker', 'project', 'model_root', 'model', 'drafter',
                'adapter', 'model_config_sha256', 'nodes', 'api_key'):
        if not cfg.get(key):
            raise ValueError(f'Missing configuration: {key}')
    for key in ('project', 'model_root', 'model', 'drafter', 'adapter'):
        value = cfg[key]
        if not Path(value).is_absolute() or any(c in value for c in ',\n\r'):
            raise ValueError(f'{key} must be an absolute path without commas or newlines')
        cfg[key] = str(Path(value).resolve())
    for key in ('model', 'drafter'):
        if not Path(cfg[key]).is_relative_to(cfg['model_root']):
            raise ValueError(f'{key} must be inside model_root (mounted read-only)')
    if len(cfg['nodes']) != 2:
        raise ValueError('Exactly two nodes are required')
    for entry in cfg['nodes']:
        if not all(entry.get(key) for key in ('ip', 'interface', 'hcas')):
            raise ValueError('Each node needs ip, interface and hcas')
        if not isinstance(entry['hcas'], list) or not all(isinstance(h, str) for h in entry['hcas']):
            raise ValueError('Node hcas must be a list of device names')
    for key, default in (('api_port', 8002), ('worker_port', 8896), ('master_port', 29627)):
        cfg.setdefault(key, default)
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 65535:
            raise ValueError(f'Invalid {key}')
    cfg.setdefault('api_bind', '127.0.0.1')
    cfg['master_addr'] = cfg['nodes'][0]['ip']
    cfg['stock'] = profile == 'stock'
    cfg['mode'] = profile
    cfg.setdefault('namespace', 'sparkglm-heretic')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', cfg['namespace']):
        raise ValueError('Invalid container namespace')
    cfg['prefix'] = cfg['service'] = cfg['namespace']
    cfg['profile'] = json.loads((HERE / f'profiles/{profile}.json').read_text())
    cfg['patch_sha256'] = json.loads((HERE / 'sources.lock.json').read_text())['patch_sha256']
    cfg['adapter_sha256'] = {
        'adapter_config.json': 'eef40842e770da0d5aca2caea37dc5f74650bbafeca33293bdea333c9d1e3e08',
        'adapter_model.safetensors': '59fc0271b34a1f4f6957d5d4a6c371e1ee915b1e3e5330d76161622839c3c16d',
    }
    return cfg


def remote(cfg, *args):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', cfg['worker'], shlex.join(args)]


def node(action, rank, cfg):
    payload = json.dumps(cfg, separators=(',', ':'))
    if rank == 0:
        result = subprocess.run([sys.executable, str(NODE), action, str(rank), payload],
                                capture_output=True, text=True)
    else:
        result = subprocess.run(remote(cfg, 'python3', '-', action, str(rank), payload),
                                input=NODE.read_text(), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'rank {rank} {action}: {result.stderr.strip()}')
    return json.loads(result.stdout)


def stage_adapter(cfg):
    # Verify locally before copying, then verify again on both nodes at launch.
    spec = importlib.util.spec_from_file_location('heretic_node', NODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = cfg['adapter']
    for name, expected in cfg['adapter_sha256'].items():
        if module.hash_file(Path(path) / name) != expected:
            raise RuntimeError(f'Adapter hash mismatch: {name}')
    subprocess.run(remote(cfg, 'mkdir', '-p', path), check=True)
    subprocess.run(['rsync', '-a', '--checksum', '--protect-args',
                    f'{path}/adapter_config.json', f'{path}/adapter_model.safetensors',
                    f"{cfg['worker']}:{path}/"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status', 'preflight'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--profile', choices=['stock', 'heretic', 'heretic-dflash'], default=DEFAULT_PROFILE)
    parser.add_argument('--dry-run', action='store_true', help='print commands without SSH or Docker')
    args = parser.parse_args()
    cfg = config(args.config, args.profile)
    cfg['launch_id'] = str(uuid.uuid4())
    if args.dry_run:
        spec = importlib.util.spec_from_file_location('heretic_node', NODE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg['api_key'] = '<redacted>'
        print(json.dumps({'action': args.action, 'mode': cfg['mode'],
                          'rank_commands': [module.docker_argv(cfg, rank, cfg['launch_id'])
                                            for rank in (0, 1)]}, indent=2))
        return
    results = []
    if args.action == 'start':
        for rank in (0, 1):
            results.append(node('preflight', rank, cfg))
        if not cfg['stock']:
            stage_adapter(cfg)
        try:
            for rank in (0, 1):
                results.append(node('start', rank, cfg))
        except Exception:
            for rank in (1, 0):
                try:
                    node('rollback', rank, cfg)
                except Exception as error:
                    print(str(error), file=sys.stderr)
            raise
    else:
        action = 'preflight-adapter' if args.action == 'preflight' else args.action
        errors = []
        for rank in (1, 0):
            try:
                results.append(node(action, rank, cfg))
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))
    directory = Path(cfg['project']) / 'runtime' / cfg['mode']
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'last-{args.action}.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
