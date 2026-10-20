#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline regressions for the migrated shared-base launcher."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cluster, node = module('cluster'), module('node')


class Launcher(unittest.TestCase):
    def config(self, profile=None):
        if profile is None:
            return cluster.config(ROOT / 'cluster.example.json')
        return cluster.config(ROOT / 'cluster.example.json', profile)

    def test_explicit_heretic_keeps_eager_and_shared_base_mount(self):
        cfg = self.config('heretic')
        self.assertFalse(cfg['stock'])
        self.assertNotIn('--dflash', cfg['profile']['server_argv'])
        for rank in (0, 1):
            argv = node.docker_argv(cfg, rank, 'fixture')
            self.assertIn('--model-name=glm53-base', argv)
            self.assertIn('--lora-adapter=heretic=/adapter', argv)
            self.assertIn(f"type=bind,src={cfg['model_root']},dst={cfg['model_root']},readonly", argv)
            self.assertNotIn('--speculative', argv)
            self.assertFalse(any('${' in arg for arg in argv))

    def test_default_dflash_keeps_reviewed_single_owner_guards(self):
        cfg = self.config()
        self.assertEqual(cfg['mode'], 'heretic-dflash')
        for rank in (0, 1):
            argv = node.docker_argv(cfg, rank, 'fixture')
            for expected in ('--dflash', '--mtp-gate=force', '--dflash-gamma=8',
                             '--dflash-window-size=2048', '--max-prefill-tokens=2048',
                             '--max-num-seqs=1', 'ATLAS_GLM_HERETIC_DFLASH=1',
                             'ATLAS_GLM_LORA_COMPACT=1', 'ATLAS_GLM_CANONICAL_VERIFY=1',
                             'ATLAS_GLM_PREFILL_SP=0', 'ATLAS_GLM_KV_SHARD=0',
                             'ATLAS_DEBUG_NO_GRAPH=1', '--enable-prefix-caching=false'):
                self.assertIn(expected, argv)

    def test_custom_public_names_on_both_ranks_and_profiles(self):
        for profile in ('stock', 'heretic', 'heretic-dflash'):
            cfg = self.config(profile)
            cfg.update(served_model_name='org/my-base', served_adapter_name='org/my-heretic')
            for rank in (0, 1):
                argv = node.docker_argv(cfg, rank, 'fixture')
                self.assertIn('--model-name=org/my-base', argv)
                self.assertEqual('--lora-adapter=org/my-heretic=/adapter' in argv, profile != 'stock')
                self.assertNotIn('--lora-adapter=heretic=/adapter', argv)

    def test_rejects_invalid_or_colliding_public_names(self):
        source = json.loads((ROOT / 'cluster.example.json').read_text())
        for changes in ({'served_model_name': 'same', 'served_adapter_name': 'same'},
                        {'served_adapter_name': ''}, {'served_adapter_name': 'bad=name'},
                        {'served_model_name': 'bad name'}, {'served_adapter_name': None}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(dict(source, **changes)))
                with self.assertRaises(ValueError):
                    cluster.config(path)

    def test_stop_targets_same_namespace_across_profiles(self):
        prefixes = {self.config(p)['prefix'] for p in ('stock', 'heretic', 'heretic-dflash')}
        self.assertEqual(len(prefixes), 1)

    def test_drafter_required_only_for_dflash(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config('heretic')
            cfg['drafter'] = directory
            node.require_drafter(cfg)
            cfg = self.config('heretic-dflash')
            cfg['drafter'] = directory
            for name in ('config.json', 'model.safetensors'):
                with self.assertRaisesRegex(RuntimeError, name):
                    node.require_drafter(cfg)
                (Path(directory) / name).write_text('fixture')
            node.require_drafter(cfg)

    def test_rank_specific_fabric_and_stock_without_adapter(self):
        cfg = self.config('stock')
        cfg['nodes'][1].update(interface='worker-net', hcas=['worker-rail'])
        argv = node.docker_argv(cfg, 1, 'fixture')
        self.assertIn('NCCL_SOCKET_IFNAME=worker-net', argv)
        self.assertIn('ATLAS_RDMA_RAILS=worker-rail', argv)
        self.assertNotIn('--lora-adapter=heretic=/adapter', argv)
        self.assertFalse(any('dst=/adapter' in arg for arg in argv))

    def test_ownership_and_rollback_protect_other_launches(self):
        cfg = self.config()
        with patch.object(node, 'inspect', return_value={'Config': {'Labels': {}}}):
            with self.assertRaisesRegex(RuntimeError, 'ownership'):
                node.owned_container(cfg, 0)
        obj = {'Config': {'Labels': {'io.onlineheretic.service': cfg['service'],
                                    'io.onlineheretic.launch': 'different-launch'}}}
        with patch.object(node, 'inspect', return_value=obj):
            with self.assertRaisesRegex(RuntimeError, 'different launch'):
                node.owned_container(cfg, 0, 'this-launch')

    def test_rejects_model_outside_readonly_mount(self):
        source = json.loads((ROOT / 'cluster.example.json').read_text())
        source['model'] = '/somewhere/else/model'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'model must be inside model_root'):
                cluster.config(path)

    def test_invalid_adapter_rejected_before_ssh(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as directory:
            cfg['adapter'] = directory
            (Path(directory) / 'adapter_config.json').write_text('{}')
            with patch.object(cluster.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'Adapter hash mismatch'):
                    cluster.stage_adapter(cfg)
                run.assert_not_called()

    def test_root_entrypoint_dry_run_is_offline_and_redacts_key(self):
        repo = ROOT.parents[1]
        result = subprocess.run(['bash', str(repo / 'start.sh'), 'heretic', 'start',
                                 '--config', str(ROOT / 'cluster.example.json'), '--dry-run'],
                                cwd=repo, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(len(value['rank_commands']), 2)
        self.assertEqual(value['mode'], 'heretic-dflash')
        for argv in value['rank_commands']:
            self.assertIn('--dflash', argv)
            self.assertIn('--draft-model=' + self.config()['drafter'], argv)
        self.assertNotIn('REPLACE_WITH_YOUR_LOCAL_API_KEY', result.stdout)
        self.assertIn('--auth-token=<redacted>', result.stdout)


if __name__ == '__main__':
    unittest.main()
