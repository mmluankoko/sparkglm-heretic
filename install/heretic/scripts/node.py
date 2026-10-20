#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Node-local operations. Ownership labels bound every destructive operation."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def run(argv, check=True):
    return subprocess.run(argv, check=check, capture_output=True, text=True)


def inspect(name):
    result = run(['docker', 'inspect', name], check=False)
    if result.returncode:
        if 'no such' in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def owned_container(cfg, rank, launch_id=None):
    obj = inspect(f"{cfg['prefix']}-rank{rank}")
    if obj is None:
        return None
    labels = obj['Config'].get('Labels') or {}
    if labels.get('io.onlineheretic.service') != cfg['service']:
        raise RuntimeError('Refusing container with mismatched ownership label')
    if launch_id and labels.get('io.onlineheretic.launch') != launch_id:
        raise RuntimeError('Refusing rollback of a different launch')
    return obj


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_drafter(cfg):
    if '--dflash' in cfg['profile']['server_argv']:
        for name in ['config.json', 'model.safetensors']:
            if not (Path(cfg['drafter']) / name).is_file():
                raise RuntimeError(f'Missing DFlash drafter: {name}')


def preflight(cfg, rank, adapter=False):
    host = cfg['nodes'][rank]
    if host.get('hostname') and socket.gethostname().split('.')[0] != host['hostname']:
        raise RuntimeError('Rank/hostname mismatch')
    image = json.loads(run(['docker', 'image', 'inspect', cfg['image']]).stdout)[0]
    if (cfg['image'].startswith('sha256:') and image['Id'] != cfg['image']) or image['Architecture'] != 'arm64':
        raise RuntimeError('Image ID or architecture mismatch')
    labels = image['Config'].get('Labels') or {}
    if not cfg['stock'] and labels.get('io.onlineheretic.patch-sha256') != cfg['patch_sha256']:
        raise RuntimeError('Image does not contain the expected Heretic engine patch')
    if inspect(f"{cfg['prefix']}-rank{rank}") is not None:
        raise RuntimeError('Project container already exists; use the matching stop script')
    live = run(['docker', 'ps', '--format', '{{.Names}}']).stdout.splitlines()
    if any('vllm' in name or name.startswith(('atlas-sparkglm-rank', 'onlineheretic-sparkglm-'))
           for name in live):
        raise RuntimeError('Another model service is running; it must be stopped explicitly first')
    model = Path(cfg['model'])
    for name in ['config.json', 'model.safetensors.index.json', 'tokenizer.json']:
        if not (model / name).is_file():
            raise RuntimeError(f'Missing model file: {model / name}')
    if hash_file(model / 'config.json') != cfg['model_config_sha256']:
        raise RuntimeError('Base model config hash mismatch')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        if not (model / name).is_file():
            raise RuntimeError(f'Missing model shard: {model / name}')
    require_drafter(cfg)
    if adapter and not cfg['stock']:
        for name, expected in cfg['adapter_sha256'].items():
            if hash_file(Path(cfg['adapter']) / name) != expected:
                raise RuntimeError(f'Adapter hash mismatch: {name}')
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', cfg['api_port'] if rank == 0 else cfg['worker_port']))
    if rank == 0:
        with socket.socket() as sock:
            sock.bind((cfg['master_addr'], cfg['master_port']))
    addresses = json.loads(run(['ip', '-j', 'addr', 'show', 'dev', host['interface']]).stdout)
    if not any(a.get('local') == host['ip'] for item in addresses for a in item['addr_info']):
        raise RuntimeError('Cluster interface/IP mismatch')
    for hca in host['hcas']:
        state = Path('/sys/class/infiniband') / hca / 'ports/1/state'
        if not state.exists() or 'ACTIVE' not in state.read_text():
            raise RuntimeError(f'RDMA HCA is not active: {hca}')
    return {'rank': rank, 'image': image['Id'], 'preflight': 'passed',
            'mode': cfg['mode'], 'model_config_sha256': cfg['model_config_sha256'],
            'model_shards_present': len(set(index['weight_map'].values())),
            'drafter_files_present': '--dflash' in cfg['profile']['server_argv'],
            'adapter_sha256': cfg['adapter_sha256'] if adapter and not cfg['stock'] else None,
            'interface': host['interface'], 'node_ip': host['ip'],
            'active_hcas': host['hcas'], 'api_or_worker_port_available': True,
            'master_port_available': True if rank == 0 else None,
            'running_containers': live}


def docker_argv(cfg, rank, launch_id):
    host = cfg['nodes'][rank]
    cache = str(Path(cfg['project']) / 'runtime' / cfg['mode'] / 'cache')
    args = ['docker', 'run', '--detach', '--pull', 'never', '--name', f"{cfg['prefix']}-rank{rank}",
            '--label', f"io.onlineheretic.service={cfg['service']}", '--label', f'io.onlineheretic.launch={launch_id}',
            '--gpus', 'all', '--network', 'host', '--ipc', 'host', '--device', '/dev/infiniband',
            '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1:-1', '--user', f'{os.getuid()}:{os.getgid()}',
            '--restart', 'no', '--init', '--security-opt', 'no-new-privileges',
            '--log-opt', 'max-size=30m', '--log-opt', 'max-file=3',
            '--mount', f"type=bind,src={cfg['model_root']},dst={cfg['model_root']},readonly",
            '--mount', f'type=bind,src={cache},dst=/cache', '--entrypoint', '/usr/local/bin/spark']
    if not cfg['stock']:
        args += ['--mount', f"type=bind,src={cfg['adapter']},dst=/adapter,readonly"]
    env = dict(cfg['profile']['environment'])
    env.update({'NCCL_SOCKET_IFNAME': host['interface'], 'NCCL_IB_HCA': ','.join(host['hcas']),
                'ATLAS_RDMA_RAILS': ','.join(host['hcas']), 'HF_HUB_OFFLINE': '1', 'HF_HOME': '/cache/huggingface'})
    for key, value in sorted(env.items()):
        args += ['--env', f'{key}={value}']
    server = [str(a).replace('${model}', cfg['model']).replace('${drafter}', cfg['drafter'])
              for a in cfg['profile']['server_argv']]
    server += [f'--rank={rank}', f"--master-addr={cfg['master_addr']}", f"--master-port={cfg['master_port']}",
               f"--bind={cfg['api_bind']}" if rank == 0 else '--bind=127.0.0.1',
               f"--port={cfg['api_port'] if rank == 0 else cfg['worker_port']}",
               '--model-name=glm53-base', '--require-auth', f"--auth-token={cfg['api_key']}"]
    return [*args, cfg['image'], *server]


def main():
    action, rank, payload = sys.argv[1:]
    rank, cfg = int(rank), json.loads(payload)
    launch_id = cfg.get('launch_id')
    if action in ('preflight', 'preflight-adapter'):
        result = preflight(cfg, rank, adapter=action == 'preflight-adapter')
    elif action == 'start':
        preflight(cfg, rank, adapter=True)
        (Path(cfg['project']) / 'runtime' / cfg['mode'] / 'cache').mkdir(parents=True, exist_ok=True)
        result = {'rank': rank, 'container_id': run(docker_argv(cfg, rank, launch_id)).stdout.strip()}
    elif action in ('stop', 'rollback'):
        obj = owned_container(cfg, rank, launch_id if action == 'rollback' else None)
        if obj:
            run(['docker', 'stop', '--time', '60', obj['Id']])
            run(['docker', 'rm', obj['Id']])
        result = {'rank': rank, 'stopped': obj is not None}
    elif action == 'status':
        obj = owned_container(cfg, rank)
        result = {'rank': rank, 'state': obj['State'] if obj else None}
    else:
        raise RuntimeError('Unknown action')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
