# Frozen-Frame Language Grounding Probe

Date: 2026-11-23 15:46 CST
Status: updated 2026-11-23 16:35 CST; target-stop gate remains closed

## Question

On the same cached RGB frame, does the current VLM grounder change its visual direction when the instruction target changes, and does it mark a visible destination as near enough to stop?

## Input selection

The four RGB samples were read from the completed `r2-subgoal-trained-subgoal` evaluation. Each correct, swapped, and empty condition used identical frame bytes within that sample, a reset VLM history, and `memory=none`.

| Sample | Correct task / goal | Swapped task / goal | RGB frame |
| --- | --- | --- | --- |
| Branch to tuyere | `sage3d_841980/ts_2/3` / `tuyere_2` | `sage3d_841980/ts_3/2` / `fruit_plate_1` | `ts_2_3/rgb_00059.png` |
| Branch to fruit plate | `sage3d_841980/ts_3/2` / `fruit_plate_1` | `sage3d_841980/ts_2/3` / `tuyere_2` | `ts_3_2/rgb_00059.png` |
| Near door | `sage3d_839992/ti_3/1` / `door_1` | `sage3d_839992/ti_1/4` / `track_light_1` | `ti_3_1/rgb_00086.png` |
| Near wall design | `sage3d_839992/ti_0/0` / `wall_design_2` | `sage3d_839992/ti_3/1` / `door_1` | `ti_0_0/rgb_00045.png` |

The branch frames were initially selected at the earliest observed rollout pose separation greater than 0.5 m. A later CPU check found this was not the task-route fork. The `ts_2` and `ts_3` reference paths share their first 20 points, and both frame-59 poses are still near the bedroom spawn, well before their later route split. Their common `R` response cannot be scored as a language error. The two terminal frames come from successful subgoal-input routes. Images are 320x240. This probe used the VLM worker only. It did not run a student or closed-loop flight.

## Results

The original worker had a 12-token output limit. It returned `parse_fallback` for 4 of 8 nonempty queries. Captured raw strings showed decimal values such as `0.8` and `0.8055` in the binary near-goal slot, plus a truncated response. The parser safely rejected those outputs.

The worker prompt now requires an integer `0` or `1` in the third field and forbids confidence values there. The output limit is 20 tokens. The parser remains fail-closed. The same cached frames produced:

| Sample | Condition | Raw response | Parsed bearing / mode / near |
| --- | --- | --- | --- |
| Branch to tuyere | correct | `R, S, 0, bedroom` | `R / subgoal / 0` |
| Branch to tuyere | swap | `R, S, 0, word` | `R / subgoal / 0` |
| Branch to fruit plate | correct | `R, S, 0, word` | `R / subgoal / 0` |
| Branch to fruit plate | swap | `R, S, 0, bedroom` | `R / subgoal / 0` |
| Near door | correct | `R, F, 0, door` | `R / final / 0` |
| Near door | swap | `R, S, 0, word` | `R / subgoal / 0` |
| Near wall design | correct | `R, S, 0, doorway` | `R / subgoal / 0` |
| Near wall design | swap | `R, S, 0, door` | `R / subgoal / 0` |

All 8 nonempty outputs parsed after the prompt change. Correct and swapped conditions returned the same bearing on both branch samples. Only the correct door instruction returned `final`, and no query returned `near_goal=1`. Empty input returned SEARCH with `near_goal=0` for all four samples through the grounder's local fast path, without a VLM call.

The prompt adjustment fixes malformed output frequency in this small sample. It does not establish correct language-conditioned branch selection or destination stopping. The generic `reason` strings also do not provide a stable target-instance identity for the tracker's repeated-evidence check.

## Artifacts

- Before prompt adjustment: `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/language-grounding-probe-r2-subgoal-20261123/probe.json`
- After prompt adjustment: `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/language-grounding-probe-r2-subgoal-20261123/probe_after_prompt_v2.json`
- Student checkpoint associated with the source episodes: `513e95a7a26709ab38a0157453138ea28ef0dac663e8f267919c2c64e397e667`
- Both probe passes exposed physical GPU0 only. GPU1 was not used.

The result does not pass the language/target-stop gate. No G3 closed-loop evaluation or additional training was started.

## Follow-up 2026-11-23 16:35 CST

### Code and test changes

- `parse_vlm_output` now accepts only complete JSON. It no longer appends closing quotes/brackets or extracts an object from surrounding prose. The exact previously accepted truncated STOP prefix `["C","F",1,"sofa` now returns `parse_fallback` with `near_goal=0`.
- A VLM `near_goal` signal can confirm only when its visible label matches the destination phrase extracted from the current English or Chinese instruction. A repeated unrelated label such as `door` for `Move to the sofa` no longer confirms STOP. The worker prompt now specifies all four JSON field types and asks for the requested object label in final mode.
- The target-label check covers the 10 frozen development instructions' destination phrases, including object goals and area goals. It remains a text-label check. It does not establish pixel-region correspondence or distinguish two instances of the same class.
- `test_grounder.py` passes 27/27 in both the system Python and the VLM environment. `test_vln.py` passes 70/70 in the VLM environment. `py_compile` and `git diff --check` pass.

### Strict-schema probe

The same four RGB frames and correct/swap/zero conditions were queried with the 20-token decoding limit and the revised four-field prompt. This one-off stdin probe ran on physical GPU0 only and took 18.86 seconds including worker startup. The zero condition returned local SEARCH without a VLM call.

| Sample | Correct output | Swapped output | Interpretation |
| --- | --- | --- | --- |
| Branch frame 1 | `R / subgoal / 0 / none` | `R / subgoal / 0 / none` | Both goals share the first route segment here. |
| Branch frame 2 | `R / subgoal / 0 / none` | `R / subgoal / 0 / none` | Same shared-route limitation. |
| Near door | `R / final / 1 / entrance door` | `R / subgoal / 0 / none` | Correct-vs-swap mode changes on this frame. |
| Near wall design | `R / subgoal / 0 / none` | `R / subgoal / 0 / door` | No final-target STOP signal. |

All 8 nonempty responses parsed. One of four correct-task frames produced a target-matching `final` label and `near_goal=1`; the paired swapped instruction produced `subgoal` and `near_goal=0`. The positive frame's bearing was still `R`, while `VlmBeliefTracker` confirms `near_goal` only at center bearing, so this observation did not itself trigger STOP. This is a single-pair diagnostic, not a rate estimate.

The v3 prompt experiment produced 5/8 numeric fourth fields. The strict parser now rejects those malformed values. The v4 explicit field-type prompt produced 8/8 parseable responses in this small repeat. This does not establish reliable target visibility or stopping.

### Gate decision

No closed-loop evaluation or training followed. The branch samples are before the reference-route split, and the near-door result is only one positive/negative frozen-frame pair. Next, select same-pose images at an annotated target-dependent fork and expand the positive/negative target-stop set before reopening the G3 closed-loop gate.

Artifacts:

- Target-label prompt probe: `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/language-grounding-probe-r2-subgoal-20261123/probe_after_target_gate_v3.json`
- Strict four-field prompt probe: `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/language-grounding-probe-r2-subgoal-20261123/probe_after_target_gate_v4.json`
