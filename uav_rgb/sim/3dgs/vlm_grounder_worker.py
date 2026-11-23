#!/usr/bin/env python3
"""常驻零特权层次化视觉语言 Grounder 子进程服务（G3 阶段路标分解，stdin/stdout JSONL）。

仅接收机载 RGB 与自然语言指令（及 VLM 自身历史摘要），区分 FINAL（最终目的地可见）与
SUBGOAL（跨房间门洞/走廊过渡路标），严禁读取位姿、占据图或真值深度。
"""
import base64
import io
import json
import os
import sys
import time

# ========================= 顶部集中配置区 =========================
# 严禁向 /root/rivermind-fs（Inode 已满）写入任何缓存或临时文件
CACHE_ROOT = os.environ.get("UAV_RGB_CACHE_ROOT", "/root/rivermind-data/uav_rgb/cache")
os.environ["HF_HOME"] = os.path.join(CACHE_ROOT, "hf")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["TORCH_HOME"] = os.path.join(CACHE_ROOT, "torch")
os.environ["TMPDIR"] = os.path.join(CACHE_ROOT, "tmp")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_PATH = os.environ.get(
    "UAV_RGB_VLM_MODEL",
    "/root/rivermind-fs/data/Drone-PUR/models/UAV-PUR-4B",
)
INPUT_W = int(os.environ.get("UAV_RGB_VLM_W", "288"))
INPUT_H = int(os.environ.get("UAV_RGB_VLM_H", "192"))
MAX_NEW_TOKENS = int(os.environ.get("UAV_RGB_VLM_MAX_TOKENS", "20"))
MIN_FREE_VRAM_MIB = int(os.environ.get("UAV_RGB_VLM_MIN_VRAM_MIB", "6000"))
ALLOWED_REQ_KEYS = frozenset(("cmd", "frames_b64", "instruction", "memory"))
# =================================================================

PROMPT_TEMPLATE = (
    'Drone front camera (HFOV 60°).\n'
    'Instruction: "{ins}"\n'
    'Recent history: {mem}\n'
    'Understand instructions in English or Chinese. In Chinese, the destination after “到/去/前往” is the final goal; “从/离开” describes the start.\n'
    '1. Identify the FINAL destination (ignore starting "from ..." objects).\n'
    '2. If the FINAL destination itself is visible in this view, set mode="F" and direction="C" (ahead), "L" (left), or "R" (right).\n'
    '3. If the FINAL destination is NOT visible (in another room or behind walls), look for an open doorway, hallway, or room exit to advance, and set mode="S" (Subgoal) with direction "C", "L", or "R".\n'
    '4. If facing a blank wall/obstacle with neither final goal nor passage visible, output ["N","N",0,"wall"].\n'
    '5. The array has four fields: direction string, mode string, integer close flag, quoted string label.\n'
    '6. The third field is exactly integer 0 or 1. The fourth field is always a quoted string, never a number.\n'
    '7. In mode F, the last field names the requested destination object using its words from the instruction. Never name a room, doorway, route, or generic word as the final target. In mode S, name the visible passage or use "none".\n'
    '8. Set close=1 ONLY when mode="F" AND the final destination fills the foreground within ~1m; otherwise use 0.\n'
    'Output only this JSON array: ["C|L|R|N","F|S|N",0|1,"word"]'
)


def main():
    for sub in ("hf", "transformers", "torch", "tmp"):
        os.makedirs(os.path.join(CACHE_ROOT, sub), exist_ok=True)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("vlm_grounder_worker 需要 CUDA GPU")
    free_b, _total_b = torch.cuda.mem_get_info()
    free_mib = free_b // (1024 * 1024)
    if free_mib < MIN_FREE_VRAM_MIB:
        raise RuntimeError(f"GPU 空闲显存不足: {free_mib} MiB < {MIN_FREE_VRAM_MIB} MiB")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 16384, "longest_edge": INPUT_W * INPUT_H * 2}
    if free_mib >= 12000:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.float16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval().to(torch.float16)
    else:
        gpu_cap = f"{max(4800, free_mib - 2200)}MiB"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.float16, device_map="auto",
            max_memory={0: gpu_cap, "cpu": "32GiB"},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()

    tok = processor.tokenizer
    eos_ids = list({151645, 151643, tok.encode('"]')[0], tok.encode("]")[0]})
    cand_ids = [tok.encode(c, add_special_tokens=False)[0] for c in ("L", "C", "R", "N")]

    warm_im = Image.new("RGB", (INPUT_W, INPUT_H), (128, 128, 128))
    warm_msgs = [
        {"role": "user", "content": [{"type": "image", "image": warm_im}, {"type": "text", "text": PROMPT_TEMPLATE.format(ins="warmup", mem="none")}]},
        {"role": "assistant", "content": [{"type": "text", "text": '["'}]},
    ]
    warm_in = processor.apply_chat_template(
        warm_msgs, tokenize=True, continue_final_message=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    warm_in["pixel_values"] = warm_in["pixel_values"].to(torch.float16)
    with torch.inference_mode():
        model.generate(**warm_in, max_new_tokens=4, do_sample=False, eos_token_id=eos_ids)
    torch.cuda.synchronize()

    sys.stdout.write(json.dumps({"status": "ready", "model": MODEL_PATH}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if not isinstance(req, dict):
            raise ValueError("vlm_grounder_worker 请求必须是 JSON 对象")
        extra_keys = set(req.keys()) - ALLOWED_REQ_KEYS
        if extra_keys:
            raise ValueError(f"零特权边界拒绝未知或特权字段: {sorted(extra_keys)}")
        cmd = req.get("cmd")
        if cmd == "close":
            break
        if cmd is not None:
            raise ValueError(f"未知的 cmd={cmd!r}")
        ins = str(req.get("instruction") or "").strip()
        frames_b64 = req.get("frames_b64") or []
        if not ins or not frames_b64:
            sys.stdout.write(json.dumps({"raw": '["N","N",0,"empty"]', "conf": 0.0, "infer_ms": 0.0}) + "\n")
            sys.stdout.flush()
            continue

        mem = str(req.get("memory") or "none").strip()[:160] or "none"
        t0 = time.perf_counter()
        raw_bytes = base64.b64decode(frames_b64[-1])
        with Image.open(io.BytesIO(raw_bytes)) as im:
            cur_img = im.convert("RGB").resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR)
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": cur_img}, {"type": "text", "text": PROMPT_TEMPLATE.format(ins=ins, mem=mem)}]},
            {"role": "assistant", "content": [{"type": "text", "text": '["'}]},
        ]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, continue_final_message=True, return_dict=True, return_tensors="pt",
        ).to("cuda:0")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=eos_ids, return_dict_in_generate=True, output_scores=True,
            )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        conf = float(out.scores[0][0, cand_ids].float().softmax(dim=-1).max().item()) if out.scores else 0.5
        raw_txt = '["' + processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        del out, inputs

        sys.stdout.write(json.dumps({"raw": raw_txt, "conf": round(conf, 4), "infer_ms": round(ms, 2)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
