import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_openfly_annotations import audit


class TestAuditOpenFlyAnnotations(unittest.TestCase):
    def test_element_validation_preserves_published_negative_actions_and_4d_positions(self):
        valid = {"gpt_instruction": "fly up and down", "action": [-1, -2, 0],
                 "index_list": ["a", "b", "c"],
                 "pos": [[0, 0, 1, 0], [0, 0, 4, 0], [0, 0, 1, 0]], "yaw": [0, 0, 0]}
        patches = [{"action": [True, -2, 0]}, {"action": [1.5, -2, 0]},
                   {"index_list": [{}, "b", "c"]}, {"index_list": ["", "b", "c"]},
                   {"pos": [[0, 0, float("nan")]] * 3}, {"pos": [[0, 0, True]] * 3},
                   {"pos": [[0, 0]] * 3}, {"yaw": [0, float("inf"), 0]}]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            source.write_text(json.dumps([valid]))
            self.assertTrue(audit(source)["ok"])
            for patch in patches:
                with self.subTest(patch=patch):
                    source.write_text(json.dumps([{**valid, **patch}]))
                    report = audit(source)
                    self.assertFalse(report["ok"])
                    self.assertEqual(report["invalid_records"], 1)

    def test_cli_output_cannot_overwrite_or_enter_annotation_inputs(self):
        valid = {"gpt_instruction": "stop", "action": [0],
                 "index_list": ["a"], "pos": [[0, 0, 1]], "yaw": [0]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            source.write_text(json.dumps([valid]))
            before = source.read_bytes()
            alias = root / "alias.json"
            alias.symlink_to(source)
            output = root / "external_report.json"
            output.with_name(output.name + ".tmp").symlink_to(source)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "linked.json").symlink_to(source)
            cases = [(source, source), (source, alias), (source, output),
                     (root, root / "report.json"), (corpus, source), (corpus, output)]
            for input_path, output_path in cases:
                with self.subTest(input_path=input_path, output_path=output_path):
                    source.write_bytes(before)
                    result = subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("audit_openfly_annotations.py")),
                         "--input", str(input_path), "--output", str(output_path)],
                        capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(source.read_bytes(), before)

    def test_json_and_jsonl_are_counted_without_touching_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "train.json"
            first.write_text(json.dumps([{
                "gpt_instruction": "fly to the red roof",
                "action": [9, 9, 0],
                "index_list": ["a", "b", "c"],
                "pos": [[0, 0, 1], [3, 0, 1], [3, 4, 1]],
                "yaw": [0, 0, 1],
            }]), encoding="utf-8")
            second = root / "unseen.jsonl"
            second.write_text(json.dumps({
                "gpt_instruction": "turn left",
                "action": [2],
                "pos": [[0, 0, 1], [1, 0, 1]],
            }) + "\n", encoding="utf-8")
            before = first.read_bytes(), second.read_bytes()
            report = audit(root)
            self.assertEqual(report["files"], 2)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["records_with_action_and_pos"], 2)
            self.assertEqual(report["action_counts"], {"0": 1, "2": 1, "9": 2})
            self.assertEqual(report["trajectory_length_m"]["max"], 7.0)
            self.assertEqual(report["split_counts"], {"train": 1, "unseen": 1})
            self.assertEqual(before, (first.read_bytes(), second.read_bytes()))

    def test_mismatched_parallel_fields_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text(json.dumps({"action": [1], "index_list": ["a", "b"], "pos": [[0, 0]], "yaw": [0]}), encoding="utf-8")
            report = audit(p)
            self.assertEqual(report["length_mismatch_records"], {"action/index_list/pos/yaw": 1})
            self.assertFalse(report["ok"])
            self.assertEqual(report["invalid_records"], 1)

    def test_cli_rejects_missing_language_or_invalid_parallel_fields(self):
        valid = {"gpt_instruction": "fly to the red roof", "action": [0],
                 "index_list": ["a"], "pos": [[0, 0, 1]], "yaw": [0]}
        cases = [("valid", [valid], True), ("empty_dataset", [], False)]
        for field in ("gpt_instruction", "action", "index_list", "pos", "yaw"):
            missing = {key: value for key, value in valid.items() if key != field}
            empty = {**valid, field: "  " if field == "gpt_instruction" else []}
            cases.extend([(f"missing_{field}", [missing], False),
                          (f"empty_{field}", [empty], False)])
        cases.append(("mismatch", [{**valid, "yaw": [0, 1]}], False))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input.json", root / "report.json"
            for name, records, expected_ok in cases:
                with self.subTest(name=name):
                    source.write_text(json.dumps(records), encoding="utf-8")
                    before = source.read_bytes()
                    result = subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("audit_openfly_annotations.py")),
                         "--input", str(source), "--output", str(output)],
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0 if expected_ok else 2, result.stderr)
                    self.assertEqual(json.loads(output.read_text())["ok"], expected_ok)
                    self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
