from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.vln_fork_references import lookup_reference, read_reference_spans


RESOURCE = Path(__file__).with_name("data") / "vln_fork_reference_spans.json"
TEXT = "离开 Café，去𠮷野书桌旁的窗。"
GOAL = "window_demo"
DIGEST = "4d07c1ad50734dc683a9063d93557e1fcc016c5f171ae20c801b71917d726581"


class TestForkReferences(unittest.TestCase):
    def test_packaged_rows(self):
        rows = read_reference_spans(RESOURCE)
        self.assertEqual(len(rows), 80)
        self.assertEqual(Counter(r["status"] for r in rows.values()),
                         {"explicit": 74, "ambiguous": 3, "none": 3})

    def test_unicode_offsets_and_exact_key(self):
        rows = {DIGEST: {"status": "explicit", "spans": [[9, 13]], "note": "终点旁的桌子"}}
        self.assertEqual(lookup_reference(rows, TEXT, GOAL),
                         {"status": "explicit", "phrases": ["𠮷野书桌"],
                          "instance_ids": None, "note": "终点旁的桌子"})
        for text, goal in [(TEXT.lower(), GOAL), (TEXT + " ", GOAL),
                           (TEXT.replace(" ", "  "), GOAL), (TEXT, GOAL + "_2")]:
            with self.subTest(text=text, goal=goal), self.assertRaisesRegex(ValueError, "未在"):
                lookup_reference(rows, text, goal)

    def test_none_has_no_phrases_or_instance_ids(self):
        rows = {DIGEST: {"status": "none", "spans": []}}
        self.assertEqual(lookup_reference(rows, TEXT, GOAL),
                         {"status": "none", "phrases": [], "instance_ids": None})

    def test_rejects_duplicate_registration(self):
        row = {"sha256": DIGEST, "status": "none", "spans": []}
        data = {"schema": "uavrgb.vln_fork_reference_spans.v1", "rows": [row, row]}
        with patch.object(Path, "read_text", return_value=json.dumps(data)), self.assertRaisesRegex(ValueError, "重复"):
            read_reference_spans(RESOURCE)

    def test_rejects_invalid_intervals(self):
        for spans in ([[0, len(TEXT) + 1]], [[3, 3]], [[4, 10], [9, 12]], [[True, 3]]):
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, "区间无效"):
                lookup_reference({DIGEST: {"status": "explicit", "spans": spans}}, TEXT, GOAL)


if __name__ == "__main__":
    unittest.main()
