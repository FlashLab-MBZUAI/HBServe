#!/usr/bin/env python3
"""Bounded config contract checks against an explicitly selected HBFSim checkout.

Run with --hbfsim-root and --simulator. This checks startup, not timing accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def values(path):
    return dict(line.split('#', 1)[0].strip().split('=', 1)
                for line in path.read_text().splitlines()
                if '=' in line.split('#', 1)[0])


class ConfigurationCompatibility(unittest.TestCase):
    def test_every_config_is_accepted_by_native_backend(self):
        # Partial overlays need a base; complete profiles override it themselves.
        base = ROOT / 'configs/systems/eight-stack-baseline.cfg'
        for path in sorted((ROOT / 'configs').rglob('*.cfg')):
            with self.subTest(config=str(path.relative_to(ROOT))):
                result = subprocess.run(
                    [str(SIMULATOR), '--describe-system', '--enable-hbf', 'true',
                     '--system-config', str(base), '--system-config', str(path)],
                    text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['schema']['name'],
                                 'hbfsim.resolved_system')

    def test_profiles_match_declared_upstream_values(self):
        receipt = json.loads((ROOT / 'configs/ocp-profile-source.json').read_text())
        for name, source in receipt['profiles'].items():
            with self.subTest(config=name):
                self.assertEqual(values(ROOT / 'configs' / name),
                                 values(HBFSIM / source['source']))

    def test_capacity_scaling_preserves_physics(self):
        for name in ('eight-stack-baseline.cfg', '4hbm-4hbf.cfg',
                     '6hbm-2hbf.cfg', '2hbm-6hbf.cfg'):
            full = values(ROOT / 'configs/systems' / name)
            mini = values(ROOT / 'configs/systems/miniquick' / name)
            differences = {k for k in full.keys() | mini.keys() if full.get(k) != mini.get(k)}
            self.assertEqual(differences, {'hbm-capacity-bytes', 'hbf-blocks-per-plane'})
            for row in (full, mini):
                banks = int(row['hbf-channels']) * int(row['hbf-dies-per-channel']) * int(row['hbf-planes-per-die'])
                self.assertEqual(banks, 256)
            # Preserve pre-migration raw HBF capacity: 512 GiB / 50 GiB per stack.
            self.assertEqual(256 * int(full['hbf-blocks-per-plane']) * 256 * 4096, 512 * 2**30)
            self.assertEqual(256 * int(mini['hbf-blocks-per-plane']) * 256 * 4096, 50 * 2**30)

    def test_window_system_paths_resolve_with_selected_client(self):
        from hbfsim_client.simulation_session import ResolvedSystemConfig
        for path in sorted((ROOT / 'configs/windows').glob('*.json')):
            for topology in json.loads(path.read_text())['topologies']:
                paths = [(path.parent / p).resolve() for p in topology['system_configs']]
                with self.subTest(experiment=path.name, topology=topology['id']):
                    self.assertTrue(all(p.is_file() for p in paths))
                    # 0h8f remains unsupported by the controller-HBM contract;
                    # parsing does not certify that topology can execute.
                    ResolvedSystemConfig.load(paths).resolve(SIMULATOR, enable_hbf=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hbfsim-root', type=Path, required=True)
    parser.add_argument('--simulator', type=Path, required=True)
    parser.add_argument('--client', choices=('bundled', 'external'), default='bundled')
    args, rest = parser.parse_known_args()
    HBFSIM, SIMULATOR = args.hbfsim_root.resolve(), args.simulator.resolve()
    sys.path[:0] = ([str(ROOT), str(HBFSIM)] if args.client == "bundled"
                    else [str(HBFSIM), str(ROOT)])
    import hbfsim_client
    import hbserve
    assert Path(hbfsim_client.__file__).resolve().is_relative_to(ROOT if args.client == "bundled" else HBFSIM)
    assert Path(hbserve.__file__).resolve().is_relative_to(ROOT)
    unittest.main(argv=[__file__, *rest])
