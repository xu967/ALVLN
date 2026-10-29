import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import scene_tool  # noqa: E402

MOTOR = ('<plugin filename="gz-sim-multicopter-motor-model-system" '
         'name="gz::sim::systems::MulticopterMotorModel"/>')
VELOCITY_ONLY = ('<plugin filename="gz-sim-multicopter-control-system" '
                 'name="gz::sim::systems::MulticopterVelocityControl"/>')


def smoke(world):
    return subprocess.run([str(ROOT / 'run.sh'), 'sim', 'smoke', '--world', str(world), '--dry-run'],
                          capture_output=True, text=True)


class V7Contract(unittest.TestCase):
    def test_static_scene_rejected_as_flight(self):
        (ROOT / 'artifacts').mkdir(exist_ok=True)  # do not depend on gitignored, pre-generated files
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
            r = smoke(scene_tool.generate('forest', 42, 24, pathlib.Path(d)))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('GZ_MOTOR_MODEL_MISSING', r.stderr)

    def test_velocity_controller_alone_is_not_motor_dynamics(self):
        with tempfile.TemporaryDirectory() as d:
            world = pathlib.Path(d) / 'w.sdf'
            world.write_text(f'<sdf version="1.9"><world name="w">{VELOCITY_ONLY}</world></sdf>')
            r = smoke(world)
        self.assertIn('GZ_MOTOR_MODEL_MISSING', r.stderr)

    def test_four_motor_plugins_pass_the_static_check(self):
        with tempfile.TemporaryDirectory() as d:
            world = pathlib.Path(d) / 'w.sdf'
            world.write_text(f'<sdf version="1.9"><world name="w">{MOTOR * 4}</world></sdf>')
            r = smoke(world)
        self.assertNotIn('GZ_MOTOR_MODEL_MISSING', r.stderr)

    def test_unimplemented_camera_cannot_false_pass(self):
        r = subprocess.run([str(ROOT / 'run.sh'), 'sim', 'camera'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('FEATURE_NOT_IMPLEMENTED', r.stderr)


if __name__ == '__main__':
    unittest.main()
