#!/usr/bin/env python3
"""常驻零特权视觉语言 Grounder 子进程服务（stdin/stdout JSONL，仅接收机载 RGB 与自然语言指令）。

运行方式（由 sim/3dgs/grounder.py::VisualLanguageGrounder 自动拉起）:
  /root/rivermind-data/uav_rgb/envs/vlm-venv/bin/python sim/3dgs/vlm_grounder_worker.py
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
)  # 只读挂载的无人机指代微调 Qwen3-VL-4B（FP16 显存约 8.5 GiB）
INPUT_W = int(os.environ.get("UAV_RGB_VLM_W", "288"))             # 288x192 对应 27 个视觉 token，兼顾细节与 <0.5s 延迟
INPUT_H = int(os.environ.get("UAV_RGB_VLM_H", "192"))
MAX_NEW_TOKENS = int(os.environ.get("UAV_RGB_VLM_MAX_TOKENS", "8"))  # 紧凑数组 ["L",0,"word"] 仅需 5~6 个 token
MIN_FREE_VRAM_MIB = int(os.environ.get("UAV_RGB_VLM_MIN_VRAM_MIB", "6000"))  # 启动前最小空闲显存门禁（共享卡自动分层）
# =================================================================

PROMPT_TEMPLATE = (
    'Drone front camera (HFOV 60°). Instruction: "{ins}"\n'
    "Identify ONLY the final destination goal in the instruction (ignore starting objects).\n"
    "Is that final goal visible in this view?\n"
    '- 1st item: "C" if goal is roughly ahead in the middle half of the view, "L" if near left edge, "R" if near right edge, or "N" if not visible\n'
    "- 2nd item: 1 if the goal is visible ahead and close (<=1.2m away, large in foreground, ready to stop), else 0\n"
    "- 3rd item: 1-word label of what you see\n"
    'Format: ["C|L|R|N",0|1,"word"]'
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
    if free_mib >= 10200:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.float16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval().to(torch.float16)
    else:
        # 显卡被其他训练任务共享时，预留 1.5 GiB 给 gsplat 与 C++ 学生推理，防并发 OOM
        gpu_cap = f"{max(4800, free_mib - 1500)}MiB"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.float16, device_map="auto",
            max_memory={0: gpu_cap, "cpu": "32GiB"},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()

    tok = processor.tokenizer
    eos_ids = list({151645, 151643, tok.encode('"]')[0], tok.encode("]")[0]})
    cand_ids = [tok.encode(c)[0] for c in ("L", "C", "R", "N")]

    # Warmup 一次把算子编译开销移出在线步
    warm_im = Image.new("RGB", (INPUT_W, INPUT_H), (128, 128, 128))
    warm_msgs = [
        {"role": "user", "content": [{"type": "image", "image": warm_im}, {"type": "text", "text": PROMPT_TEMPLATE.format(ins="warmup")}]},
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
        if req.get("cmd") == "close":
            break
        ins = str(req.get("instruction") or "").strip()
        frames_b64 = req.get("frames_b64") or []
        if not ins or not frames_b64:
            sys.stdout.write(json.dumps({"raw": '["N",0,"empty"]', "conf": 0.0, "infer_ms": 0.0}) + "\n")
            sys.stdout.flush()
            continue

        t0 = time.perf_counter()
        raw_bytes = base64.b64decode(frames_b64[-1])
        with Image.open(io.BytesIO(raw_bytes)) as im:
            cur_img = im.convert("RGB").resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR)
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": cur_img}, {"type": "text", "text": PROMPT_TEMPLATE.format(ins=ins)}]},
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
        conf = 0.5
        if out.scores:
            probs = out.scores[0][0, cand_ids].float().softmax(dim=-1)
            conf = float(probs.max().item())
        raw_txt = '["' + processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        sys.stdout.write(json.dumps({"raw": raw_txt, "conf": round(conf, 4), "infer_ms": round(ms, 2)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
