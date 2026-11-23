import hashlib
import importlib.util
import json
import os
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vln_grounder", os.path.join(HERE, "grounder.py"))
grounder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grounder)


class TestLexicalGrounder(unittest.TestCase):
    def setUp(self):
        self.g = grounder.LexicalGrounder()

    def test_matches_landmark_without_using_coordinates(self):
        result = self.g.ground("Go to the track light", ["track_light_1", "sofa_1"])
        self.assertEqual(result.target, "track_light_1")
        self.assertTrue(result.accepted)

    def test_relation_sentence_is_rejected_instead_of_selecting_reference_object(self):
        result = self.g.ground(
            "Go to the track light located directly above the decorative wall art",
            ["track_light_1", "wall_art_1"],
        )
        self.assertFalse(result.accepted)
        self.assertIsNone(result.target)
        self.assertEqual(result.reason, "unsupported_relation")

    def test_rejects_unknown_instruction(self):
        result = self.g.ground("Fly to the red plant", ["track_light_1", "sofa_1"])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "below_min_score")

    def test_rejects_ambiguous_candidates(self):
        result = self.g.ground("Go to the chair", ["chair_left", "chair_right"])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "ambiguous")

    def test_goal_phrase_beats_source_distractor(self):
        result = self.g.ground(
            "Leave the cabinet within the display area and proceed to the entry door",
            ["display_cabinet_1", "door_1"],
        )
        self.assertEqual(result.target, "door_1")
        self.assertTrue(result.accepted)

    def test_instance_numbers_do_not_change_matching(self):
        result = self.g.ground("Go to door 17", ["door_1", "sofa_1"])
        self.assertEqual(result.target, "door_1")

    def test_goal_prepositions_do_not_promote_source_objects(self):
        for preposition in ("to", "toward", "towards", "at", "into"):
            with self.subTest(preposition=preposition):
                result = self.g.ground(
                    f"Leave the chair and proceed {preposition} the sofa",
                    ["chair_1", "sofa_1"],
                )
                self.assertTrue(result.accepted)
                self.assertEqual(result.target, "sofa_1")

    def test_phrase_matches_whole_tokens_only(self):
        for instruction, candidate in (
                ("Fly to the outdoor sculpture", "door_1"),
                ("Go to the cart", "art_1"),
                ("Go to the carpet", "car_1")):
            with self.subTest(instruction=instruction):
                result = self.g.ground(instruction, [candidate])
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason, "below_min_score")

    def test_source_prepositions_cannot_override_movement_destination(self):
        for instruction in (
                "Start at the chair, fly to the sofa",
                "Start at the chair, fly slowly to the sofa",
                "From next to the chair, fly to the sofa"):
            with self.subTest(instruction=instruction):
                result = self.g.ground(instruction, ["chair_1", "sofa_1"])
                self.assertTrue(result.accepted)
                self.assertEqual(result.target, "sofa_1")
                missing_goal = self.g.ground(instruction, ["chair_1"])
                self.assertFalse(missing_goal.accepted)
                self.assertEqual(missing_goal.reason, "below_min_score")

    def test_multiple_sentences_cannot_hide_another_movement_goal(self):
        for instruction in (
                "Fly to the chair. Continue to the sofa",
                "Go to the chair. Continue slowly to the sofa",
                "Go next to the chair and fly to the sofa"):
            for candidates in (["chair_1", "sofa_1"], ["chair_1"], ["sofa_1"]):
                with self.subTest(instruction=instruction, candidates=candidates):
                    result = self.g.ground(instruction, candidates)
                    self.assertFalse(result.accepted)
                    self.assertEqual(result.reason, "unsupported_multiple_goals")

    def test_destination_keeps_relations_and_following_goals(self):
        cases = [
            ("Go next to the chair", "unsupported_relation"),
            ("Go adjacent to the chair", "unsupported_relation"),
            ("Go to the chair next to the door", "unsupported_relation"),
            ("Go to the door next to the chair", "unsupported_relation"),
            ("Go to the chair adjacent to the door", "unsupported_relation"),
            ("Go to the chair and fly to the door", "unsupported_multiple_goals"),
        ]
        for instruction, reason in cases:
            with self.subTest(instruction=instruction):
                result = self.g.ground(instruction, ["chair_1", "door_1"])
                self.assertFalse(result.accepted)
                self.assertIsNone(result.target)
                self.assertEqual(result.reason, reason)

    def test_relation_grounder_drops_attributes_and_room_location(self):
        result = grounder.RelationPhraseGrounder().ground(
            "Go to the decorative painting with artistic elements in the recreation room.",
            ["decorative_painting_1", "recreation_room_1"],
        )
        self.assertEqual(result.target, "decorative_painting_1")
        self.assertTrue(result.accepted)

    def test_unsupported_semantics_never_select_a_target(self):
        cases = [
            ("飞到 door", "unsupported_language"),
            ("Go to the door, not the sofa", "unsupported_negation"),
            ("Don't go to the door", "unsupported_negation"),
            ("Go to the door then the sofa", "unsupported_multiple_goals"),
            ("Go to the door and the sofa", "unsupported_multiple_goals"),
            ("Fly up to the door", "unsupported_vertical_command"),
            ("Descend to the door", "unsupported_vertical_command"),
            ("Go to the door on the left of the sofa", "unsupported_relation"),
        ]
        for instruction, reason in cases:
            with self.subTest(instruction=instruction):
                result = self.g.ground(instruction, ["door_1", "sofa_1"])
                self.assertFalse(result.accepted)
                self.assertIsNone(result.target)
                self.assertEqual(result.reason, reason)

    def test_empty_inputs_are_safe(self):
        result = self.g.ground("", [])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.target)

    def test_catalog_is_external_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "map.json")
            with open(source, "wb") as f:
                f.write(b'{"landmarks": []}')
            import hashlib
            with open(source, "rb") as f:
                source_hash = hashlib.sha256(f.read()).hexdigest()
            path = os.path.join(tmp, "landmarks.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "s",
                           "source": "map.json", "source_sha256": source_hash,
                           "frame_id": "scene_nav", "units": "m",
                           "landmarks": [{"name": "door_1", "x": 1, "y": 2}]}, f)
            catalog = grounder.load_catalog(path, scene_name="s")
        self.assertEqual(catalog, {"door_1": {"x": 1.0, "y": 2.0}})

    def test_catalog_rejects_scene_mismatch_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "landmarks.json")
            source = os.path.join(tmp, "map.json")
            with open(source, "wb") as f:
                f.write(b"map")
            import hashlib
            source_hash = hashlib.sha256(b"map").hexdigest()
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "other",
                           "source": "map.json", "source_sha256": source_hash,
                           "frame_id": "scene_nav", "units": "m",
                           "landmarks": [{"name": "door_1", "x": 1, "y": 2}]}, f)
            with self.assertRaisesRegex(ValueError, "scene"):
                grounder.load_catalog(path, scene_name="s")

    def test_catalog_rejects_missing_provenance_and_bool_or_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "map.json")
            with open(source, "wb") as f:
                f.write(b"map")
            import hashlib
            base = {"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "s",
                    "source": "map.json", "source_sha256": hashlib.sha256(b"map").hexdigest(),
                    "frame_id": "scene_nav", "units": "m",
                    "landmarks": [{"name": "door_1", "x": 1, "y": 2}]}
            for name, mutate, message in (
                    ("no-source", lambda d: d.pop("source"), "source"),
                    ("bad-hash", lambda d: d.update(source_sha256="bad"), "source_sha256"),
                    ("bool", lambda d: d["landmarks"][0].update(x=True), "bool"),
                    ("nan", lambda d: d["landmarks"][0].update(x="NaN"), "有限"),
            ):
                path = os.path.join(tmp, name + ".json")
                doc = json.loads(json.dumps(base))
                mutate(doc)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(doc, f)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    grounder.load_catalog(path, scene_name="s")

    def test_catalog_normalizes_names_and_rejects_normalized_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "landmarks.json")
            source = os.path.join(tmp, "map.json")
            with open(source, "wb") as f:
                f.write(b"map")
            import hashlib
            doc = {"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "s",
                   "source": {"path": "map.json", "sha256": hashlib.sha256(b"map").hexdigest()},
                   "frame_id": "scene_nav", "units": "m",
                   "landmarks": [{"name": " door_1 ", "x": 1, "y": 2},
                                  {"name": "door_1", "x": 3, "y": 4}]}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaisesRegex(ValueError, "重复"):
                grounder.load_catalog(path, scene_name="s")

    def test_catalog_source_cannot_point_at_scene_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = os.path.join(tmp, "scene.json")
            catalog_path = os.path.join(tmp, "landmarks.json")
            with open(os.path.join(tmp, "map.json"), "wb") as f:
                f.write(b"map")
            import hashlib
            with open(scene_path, "w", encoding="utf-8") as f:
                f.write("{}")
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump({"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "s",
                           "source": "scene.json", "source_sha256": hashlib.sha256(b"{}").hexdigest(),
                           "frame_id": "scene_nav", "units": "m",
                           "landmarks": [{"name": "door_1", "x": 1, "y": 2}]}, f)
            with self.assertRaisesRegex(ValueError, "scene.json"):
                grounder.load_catalog(catalog_path, scene_name="s", scene_path=scene_path)

    def test_catalog_source_hash_and_coordinate_frame_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "map.json")
            with open(source, "wb") as f:
                f.write(b"map")
            import hashlib
            doc = {"schema": "uavrgb.vln_landmark_catalog.v1", "scene": "s",
                   "source": "map.json", "source_sha256": hashlib.sha256(b"wrong").hexdigest(),
                   "frame_id": "gps", "units": "m",
                   "landmarks": [{"name": "door", "x": 1, "y": 2}]}
            path = os.path.join(tmp, "landmarks.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                grounder.load_catalog(path, scene_name="s")
            doc["source_sha256"] = hashlib.sha256(b"map").hexdigest()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaisesRegex(ValueError, "frame_id"):
                grounder.load_catalog(path, scene_name="s")


class TestVisualLanguageGrounder(unittest.TestCase):
    def test_ground_step_signature_is_strictly_zero_privilege(self):
        import inspect
        sig = inspect.signature(grounder.VisualLanguageGrounder.ground_step)
        self.assertEqual(list(sig.parameters), ["self", "rgb_history", "instruction"])
        init_sig = inspect.signature(grounder.VisualLanguageGrounder.__init__)
        forbidden = {"scene", "waypoint", "occ", "pose", "dyn", "state"}
        self.assertTrue(forbidden.isdisjoint(set(init_sig.parameters)))

    def test_parse_vlm_output_sectors_and_fallbacks(self):
        self.assertEqual(grounder.parse_vlm_output('["C",1,"window"]')[:3], (0, 1, 0.8))
        self.assertEqual(grounder.parse_vlm_output('["L",0,"sofa"]')[:2], (1, 0))
        self.assertEqual(grounder.parse_vlm_output('["R",0,"plant"]')[:2], (7, 0))
        # 搜索态 N(8) 必须强制将 near_goal 清零，防止盲停
        self.assertEqual(grounder.parse_vlm_output('["N",1,"occluded"]')[:2], (8, 0))
        self.assertEqual(grounder.parse_vlm_output('{"sector":"front_left","near":1,"conf":9,"why":"lamp"}')[:3],
                         (1, 1, 0.9))
        self.assertEqual(grounder.parse_vlm_output("broken non-json text")[:2], (8, 0))
        for text in ('["C",2,"chair"]', '["C","maybe",1,"chair"]',
                     '{"sector":"C","near":1,"conf":"invalid"}', '["C","N",1,"chair"]',
                     '["C","F",true,"sofa"]', '["C","F",0.5,"sofa"]',
                     '["C","F",0.8,"sofa"]', '["C","F",0.8055,"sofa"]',
                     '["C","S",0.8055,"bed', '["C","S",0',
                     '["C","F",1,"sofa', '["C","F",1,"sofa"',
                     '["C","F","1","sofa"]', '["C","F",1,0]'):
            self.assertEqual(grounder.parse_vlm_output(text)[:2], (8, 0))

    def test_parse_vlm_output_accepts_decimal_binary_near_goal(self):
        self.assertEqual(grounder.parse_vlm_output('["L","F",0.0,"sofa"]'),
                         (1, 0, 0.8, "sofa", "final"))
        near_goal = grounder.parse_vlm_output('["C","F",1.0,"sofa"]')[1]
        self.assertIs(type(near_goal), int)
        self.assertEqual(near_goal, 1)
        self.assertEqual(grounder.parse_vlm_output('["L","S",1.0,"doorway"]')[:2], (1, 0))

    def test_instruction_target_match_ignores_source_and_location_words(self):
        cases = (
            ("Please bring something from the bedroom bed to the bathroom tuyere.", "tuyere", True),
            ("Please bring something from the bedroom bed to the bathroom tuyere.", "bathroom", False),
            ("Take an ornament from the display cabinet and showcase it near the decorative wall.",
             "wall", True),
            ("Transfer a cleaning cloth from the entrance area for polishing the ornament.",
             "ornament", True),
            ("Set up for bedtime reading by adjusting the chandelier light in the master bedroom after your bath.",
             "chandelier light", True),
            ("Go to the decorative painting with artistic elements in the recreation room.",
             "painting", True),
            ("Go to the decorative painting with artistic elements in the recreation room.",
             "recreation room", False),
            ("Go to the sofa next to the chair.", "sofa", True),
            ("Go to the sofa next to the chair.", "chair", False),
            ("Go from the downlight close by to the pendant hanging along the perimeter.",
             "pendant", True),
            ("Go from the northwest bedroom to the central living room area.",
             "central living room area", True),
            ("Take a spare towel from the bathroom and hang it near the painting in the recreation area.",
             "painting", True),
            ("Go from the decorative painting area to the lighting fixtures zone.",
             "lighting fixtures zone", True),
            ("Fly to the sofa.", "door", False),
            ("Fly to the sofa.", "it", False),
            ("请从卧室床边飞到卫生间的排气口。", "排气口", True),
            ("请从卧室床边飞到卫生间的排气口。", "卧室床", False),
            ("请前往客厅茶几上的水果盘。", "水果盘", True),
        )
        for instruction, label, expected in cases:
            with self.subTest(instruction=instruction, label=label):
                self.assertEqual(grounder.instruction_target_matches(instruction, label), expected)

    def test_mock_worker_subprocess_roundtrip_and_cleanup(self):
        import sys
        import numpy as np
        mock_code = (
            "import json, sys\n"
            "sys.stdout.write(json.dumps({'status':'ready'}) + '\\n'); sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    if req.get('cmd') == 'close': break\n"
            "    ins = req['instruction'].lower()\n"
            "    raw = '[\"C\",1,\"sofa\"]' if 'sofa' in ins else ('[\"R\",0,\"plant\"]' if 'plant' in ins else '[\"N\",0,\"none\"]')\n"
            "    sys.stdout.write(json.dumps({'raw': raw, 'conf': 0.92, 'infer_ms': 5.0}) + '\\n'); sys.stdout.flush()\n"
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        with grounder.VisualLanguageGrounder(worker_cmd=[sys.executable, "-c", mock_code], timeout_s=10.0) as vg:
            r1 = vg.ground_step([frame], "Fly to the sofa.")
            self.assertEqual((r1.bearing_bin, r1.near_goal, r1.reason), (0, 1, "sofa"))
            r2 = vg.ground_step(frame, "Go to the green plant.")
            self.assertEqual((r2.bearing_bin, r2.near_goal, r2.reason), (7, 0, "plant"))
            r_zero = vg.ground_step([frame], "")
            self.assertEqual((r_zero.bearing_bin, r_zero.near_goal), (8, 0))

    def test_g3_subgoal_vs_final_parsing_and_memory_roundtrip(self):
        import sys
        import numpy as np

        # G3 四元组：SUBGOAL 即使误写 near=1 也必须强制清零 near_goal，绝不在门洞提前停机
        self.assertEqual(grounder.parse_vlm_output('["L","S",1,"doorway"]'), (1, 0, 0.8, "doorway", "subgoal"))
        self.assertEqual(grounder.parse_vlm_output('["C","F",0,"sofa"]'), (0, 0, 0.8, "sofa", "final"))
        self.assertEqual(grounder.parse_vlm_output('["C","F",1,"sofa"]'), (0, 1, 0.8, "sofa", "final"))
        self.assertEqual(grounder.parse_vlm_output('["N","N",1,"wall"]'), (8, 0, 0.8, "wall", "search"))

        mock_code = (
            "import json, sys\n"
            "sys.stdout.write(json.dumps({'status':'ready'}) + '\\n'); sys.stdout.flush()\n"
            "seen_mems = []\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    if req.get('cmd') == 'close': break\n"
            "    seen_mems.append(req.get('memory', ''))\n"
            "    raw = '[\"L\",\"S\",0,\"doorway\"]' if len(seen_mems) == 1 else '[\"C\",\"F\",0,\"mirror_cabinet\"]'\n"
            "    sys.stdout.write(json.dumps({'raw': raw, 'conf': 0.85, 'infer_ms': 3.5}) + '\\n'); sys.stdout.flush()\n"
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        with grounder.VisualLanguageGrounder(worker_cmd=[sys.executable, "-c", mock_code], timeout_s=10.0) as vg:
            s1 = vg.ground_step([frame], "Go to the mirror cabinet in the bathroom.")
            self.assertEqual((s1.bearing_bin, s1.near_goal, s1.goal_mode, s1.reason), (1, 0, "subgoal", "doorway"))
            vg.note_action("LEFT")
            s2 = vg.ground_step([frame], "Go to the mirror cabinet in the bathroom.")
            self.assertEqual((s2.bearing_bin, s2.near_goal, s2.goal_mode, s2.reason), (0, 0, "final", "mirror_cabinet"))

    def test_student_stop_requires_fresh_target_and_leaves_obstacle_gate_active(self):
        import sys
        import numpy as np
        root = os.path.dirname(os.path.dirname(HERE))
        if root not in sys.path:
            sys.path.insert(0, root)
        collect_3dgs = importlib.import_module("sim.3dgs.collect_3dgs")
        run_vln = importlib.import_module("sim.3dgs.run_vln")
        dyn_mod = importlib.import_module("sim.3dgs.dynamics")
        FORWARD_SLOW, SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT = (
            dyn_mod.FORWARD_SLOW, dyn_mod.SAFE_NEUTRAL, dyn_mod.YAW_LEFT, dyn_mod.YAW_RIGHT,
        )

        seq = [
            grounder.VisualGrounding(1, 0, 0.82, "doorway", 4.0, "subgoal"),
            grounder.VisualGrounding(0, 1, 0.78, "sofa", 4.0, "final"),
            grounder.VisualGrounding(8, 0, 0.0, "lost", 4.0, "search"),
            grounder.VisualGrounding(0, 1, 0.78, "sofa", 4.0, "final"),
            grounder.VisualGrounding(0, 1, 0.78, "sofa", 4.0, "final"),
        ]
        idx = {"k": 0}

        class _StubVlm:
            def reset_episode(self):
                pass

            def note_action(self, _act):
                pass

            def ground_step(self, _hist, _ins):
                out = seq[min(idx["k"], len(seq) - 1)]
                idx["k"] += 1
                return out

        tracker = run_vln.VlmBeliefTracker(_StubVlm(), "Move to the sofa", update_every=1)
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[:, :80] = 200  # 非均匀真实画面（与 --blind-rgb 区分）

        class _FakeStudent:
            prev_dim = 15
            last_stop_requested = False
            last_depth_sectors = (2.2, 0.85, 0.60)
            last_near_obs_prob = 0.68

            def reset(self):
                pass

            def act(self, _msg, **_kw):
                return FORWARD_SLOW, 1.0, [0.05, 0.85, 0.05, 0.05]

        # 1) 未锁定最终目标时近墙 (d_center=0.85 < 1.05, d_left=2.2 > d_right=0.60)：门控改向 YAW_LEFT 绕障，不喊停
        ad_avoid = collect_3dgs.StudentAdapter(_FakeStudent(), goal_fn=lambda _r, _s: (0, 0))
        act_avoid, _ = ad_avoid(rgb, None, None)
        self.assertEqual((act_avoid, ad_avoid.stop_requested), (YAW_LEFT, False))

        # 2) --blind-rgb 全黑画面：严禁门控伪造转向
        ad_blind = collect_3dgs.StudentAdapter(_FakeStudent(), goal_fn=lambda _r, _s: (0, 0),
                                               rgb_transform=lambda im: np.zeros_like(im))
        act_blind, _ = ad_blind(rgb, None, None)
        self.assertEqual((act_blind, ad_blind.stop_requested), (FORWARD_SLOW, False))

        model = _FakeStudent()
        model.last_stop_requested = True
        model.last_depth_sectors = (0.85, 0.85, 0.85)
        adapter = collect_3dgs.StudentAdapter(model, goal_fn=tracker)
        for _ in range(4):
            self.assertEqual(adapter(rgb, None, None)[0], YAW_LEFT)
            self.assertFalse(adapter.stop_requested)
        self.assertEqual(adapter(rgb, None, None)[0], SAFE_NEUTRAL)
        self.assertTrue(adapter.stop_requested)

        class _WrongTargetVlm:
            def reset_episode(self):
                pass

            def ground_step(self, _hist, _ins):
                return grounder.VisualGrounding(0, 1, 0.9, "door", 4.0, "final")

        wrong_target = run_vln.VlmBeliefTracker(_WrongTargetVlm(), "Move to the sofa", update_every=1)
        self.assertEqual(wrong_target(rgb, None), (0, 0))
        self.assertEqual(wrong_target(rgb, None), (0, 0))
        self.assertFalse(wrong_target.confirmed_target)


if __name__ == "__main__":
    unittest.main()
