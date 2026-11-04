#!/usr/bin/env python3
"""2026-11-04 集成后补充的回归门禁（交付包离线单测未覆盖的确凿缺陷）。

覆盖：
- import_asset.py 的 mtllib 守卫可被前导空白 / 大写 / Tab 绕过；
- split_manifest.py --verify 对 campus / campus_assets 漏报跨 split 泄漏；
- split_for_seed / split_for_scene 对未知任务与版本错配 fail-open；
- rl_teacher 可选依赖缺失时给裸 traceback。
"""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asset_campus  # noqa: E402
import build_demo_assets  # noqa: E402
import campus  # noqa: E402
import import_asset  # noqa: E402
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class ImportAssetGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'uavrgb_assets' / 'meshes').mkdir(parents=True)
        (self.root / 'asset_catalog.json').write_text(
            json.dumps({'schema': 'uavrgb.assets.v1', 'types': {}}), encoding='utf-8')
        self.old_root = import_asset.ROOT
        import_asset.ROOT = self.root
        self.mesh = self.root / 'tree.obj'
        self.mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')

    def tearDown(self):
        import_asset.ROOT = self.old_root
        self.tmp.cleanup()

    def _register(self):
        return import_asset.register('tree', str(self.mesh), [1.0, 1.0, 1.0],
                                     'local test provenance', 'CC0')

    def test_clean_obj_accepted(self):
        self._register()
        self.assertTrue((self.root / 'uavrgb_assets' / 'meshes' / 'tree.obj').is_file())

    def test_mtllib_variants_rejected(self):
        for text in ('mtllib tree.mtl\n', '  mtllib tree.mtl\n',
                     'MTLLIB tree.mtl\n', 'mtllib\ttree.mtl\n'):
            with self.subTest(text=text):
                self.mesh.write_text(text + 'v 0 0 0\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'external mtllib'):
                    self._register()


def _leak_pairs(identity_fn, n=64):
    groups = {}
    for s in range(n):
        groups.setdefault(identity_fn(s), []).append(s)
    for seeds in groups.values():
        if len(seeds) >= 2:
            return [(seeds[0], 'train'), (seeds[1], 'test')]
    raise AssertionError('no shared map group found')


class SplitAuditTaskTests(unittest.TestCase):
    def test_base_default_unchanged(self):
        pairs = [(s, 'train') for s in range(8)]
        self.assertEqual(sm.cross_split_groups(pairs), {})
        self.assertEqual(sm.group_id_for_task(0, 'base'),
                         sm.geometry_gid(sm.sg.sample_pillars(0)[0]))

    def test_campus_assets_leak_detected(self):
        pairs = _leak_pairs(asset_campus.map_key)
        self.assertEqual(sm.cross_split_groups(pairs), {})
        bad = sm.cross_split_groups(pairs, task='campus_assets')
        self.assertEqual(len(bad), 1)

    def test_campus_leak_detected(self):
        pairs = _leak_pairs(campus.map_id)
        self.assertEqual(sm.cross_split_groups(pairs), {})
        bad = sm.cross_split_groups(pairs, task='campus')
        self.assertEqual(len(bad), 1)

    def test_unknown_task_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unknown task'):
            sm.split_for_seed(0, task='totally_unknown')

    def test_asset_campus_version_mismatch_fails_closed(self):
        meta = {'seed': 0, 'map_id': asset_campus.map_key(0),
                'geometry_version': 'wrong.version', 'pillars': []}
        with self.assertRaisesRegex(ValueError, 'mismatched geometry_version'):
            sm.split_for_scene(meta)


class OptionalDependencyTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3') is None,
                         'SB3 已安装时不测缺依赖分支')
    def test_train_ppo_reports_clear_error(self):
        r = subprocess.run([sys.executable, '-m', 'rl_teacher.train_ppo',
                            '--out', '/tmp/unused_rl_out', '--steps', '1'],
                           cwd=str(REPO), capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)
        self.assertIn('gymnasium/stable-baselines3', r.stderr)


class CatalogProvenanceCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'uavrgb_assets' / 'meshes').mkdir(parents=True)
        self.catalog = self.root / 'asset_catalog.json'
        self.catalog.write_text(json.dumps(
            {'schema': 'uavrgb.assets.v1', 'types': {}}), encoding='utf-8')
        self.old_root = import_asset.ROOT
        import_asset.ROOT = self.root
        mesh = self.root / 'src.obj'
        mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')
        import_asset.register('tree', str(mesh), [1.4, 1.4, 4.0],
                              'https://example.org/tree', 'CC0',
                              author='Test Author')

    def tearDown(self):
        import_asset.ROOT = self.old_root
        self.tmp.cleanup()

    def _check(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', ['import_asset.py', '--check']), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return import_asset.main(), out.getvalue(), err.getvalue()

    def test_check_passes_and_detects_tamper(self):
        obj = json.loads(self.catalog.read_text(encoding='utf-8'))
        self.assertEqual(import_asset.catalog_problems(obj, root=self.root), [])
        rc, out, _ = self._check()
        self.assertEqual(rc, 0)
        self.assertIn('PASS', out)
        # 改动已登记文件后必须失败，而不是静默使用旧哈希。
        (self.root / 'uavrgb_assets' / 'meshes' / 'tree.obj').write_text(
            'v 0 0 0\n', encoding='utf-8')
        problems = import_asset.catalog_problems(
            json.loads(self.catalog.read_text(encoding='utf-8')))
        self.assertTrue(any('sha256 不匹配' in p for p in problems), problems)
        rc, _, err = self._check()
        self.assertEqual(rc, 1)
        self.assertIn('sha256', err)


@unittest.skipUnless(importlib.util.find_spec('PIL') is not None, 'Pillow 未安装')
class DemoAssetGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'sim' / 'assets' / 'uavrgb_assets'
        self.root.mkdir(parents=True)
        self.catalog = self.root.parent / 'asset_catalog.json'
        self.old_root = build_demo_assets.ROOT
        build_demo_assets.ROOT = self.root

    def tearDown(self):
        build_demo_assets.ROOT = self.old_root
        self.tmp.cleanup()

    def test_demo_provenance_allows_rebuild_imported_does_not(self):
        self.catalog.write_text(json.dumps({'schema': 'uavrgb.assets.v1', 'types': {
            'tree': {'provenance': {'kind': 'imported'}}}}), encoding='utf-8')
        with self.assertRaises(SystemExit):
            build_demo_assets.main()
        # 导入的 ground 纹理同样不能被 demo 重建覆盖 provenance。
        self.catalog.write_text(json.dumps({
            'schema': 'uavrgb.assets.v1', 'types': {},
            'ground': {'provenance': {'url': 'https://example.org/ground'}}}),
            encoding='utf-8')
        with self.assertRaises(SystemExit):
            build_demo_assets.main()
        self.catalog.write_text(json.dumps({'schema': 'uavrgb.assets.v1', 'types': {
            'tree': {'provenance': {'kind': 'demo'}}}}), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            build_demo_assets.main()
        obj = json.loads(self.catalog.read_text(encoding='utf-8'))
        prov = obj['types']['tree']['provenance']
        self.assertEqual(prov['kind'], 'demo')
        self.assertTrue(prov['author'] and prov['license'] and prov['source'])
        self.assertEqual(import_asset.catalog_problems(obj, root=self.root.parent), [])


class SceneDefaultTests(unittest.TestCase):
    def _gen(self, out, *extra):
        return subprocess.run(
            [sys.executable, str(REPO / 'sim/tools/scene_gen.py'),
             '--seed', '0', '--out', str(out), *extra],
            capture_output=True, text=True, timeout=120)

    def test_cli_default_is_v1_and_legacy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            r = self._gen(tmp / 'v1.sdf')
            self.assertEqual(r.returncode, 0, r.stderr)
            v1 = (tmp / 'v1.sdf').read_text()
            m1 = json.loads((tmp / 'v1.json').read_text())
            self.assertEqual(m1['requested_appearance'], 'v1')
            self.assertEqual(m1['effective_appearance'], 'v1')
            self.assertEqual(m1['appearance_schema_version'],
                             sg.APPEARANCE_SCHEMA_VERSION)
            self.assertNotIn('name="landmark_', v1)
            self.assertIn('<scene>', v1)
            self.assertTrue(m1['appearance']['shadows'])
            r = self._gen(tmp / 'legacy.sdf', '--appearance', 'legacy')
            self.assertEqual(r.returncode, 0, r.stderr)
            legacy = (tmp / 'legacy.sdf').read_text()
            m0 = json.loads((tmp / 'legacy.json').read_text())
            self.assertEqual(m0['requested_appearance'], 'legacy')
            self.assertIn('name="landmark_', legacy)
            self.assertNotIn('<scene>', legacy)
            # 外观不影响几何/出生位姿。
            self.assertEqual(m1['pillars'], m0['pillars'])
            self.assertEqual(m1['spawn'], m0['spawn'])


if __name__ == '__main__':
    unittest.main()
