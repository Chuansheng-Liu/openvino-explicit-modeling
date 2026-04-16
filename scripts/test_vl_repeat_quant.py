#!/usr/bin/env python3
"""VL repetition quantization accuracy test.

Sends 20 identical VL pointing requests in one multi-turn conversation to test
whether the quantization mode produces extra trailing braces ('}') in JSON
output as prompt length grows.

This test was created after discovering that int4_sym quantization generates
extra '}' tokens starting around prompt=4600, while int4_asym is stable up to
prompt=13700+.  The root cause is int4_sym precision loss on the '}' token
logit in long-context repeated-VL scenarios.

Usage:
    python scripts/test_vl_repeat_quant.py
    python scripts/test_vl_repeat_quant.py --turns 30
    python scripts/test_vl_repeat_quant.py --url http://host:8080 --no-proxy
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── System prompt (same as car_stress_test.py) ──────────────────────

CAR_SYSTEM_PROMPT = """\
<instruction>
你是一位先进的汽车智能助手，假设现在用户正在车内的主驾驶位置与你对话，每次对话前我都会告诉你当前车辆状态，和摄像头拍摄的照片。请根据这些信息以及用户的输入，判断用户的意图或者与用户进行闲聊对话
- 如果用户问你看到了什么，则回答照片中的内容。**其他所有情况都不要回答出照片的内容**
- 所有支持的意图全部列举在<supported_intents>中，其他情况则全部判定为闲聊
- 如果判断用户想与你闲聊，对于你约束必须参考<chat_prompts>
- 在用户的提问中，"我"指发言人，"你"指你自己即联博士
- 如果在照片中，用户用手指着某个方向，则：手指着左侧方向为车辆主驾驶方向，手指着右侧方向为车辆副驾驶方向。这个方向可以用于意图识别时的方向输入。
- 通过<car_status>,你可以了解车机系统现在的状态，并结合用户输入<user_input>,准确判断用户的意图
- 位置关系代称：主驾驶（司机）位置在front_left，副驾驶位置在front_right，左后位置在rear_left，右后位置在rear_right，前排位置为front，后排位置为rear，所有位置为all。
- 除闲聊外，其他所有意图必须按<supported_intents>中example的格式，以有效的JSON格式输出，不要包含任何其他文字或解释，正确的输出如: {"intent": "xxxxx", "arguments": {"xxxx": "xxxxx", "xxxx": "xxxxx"}}
- 重要：即使用户发送了照片，只要能识别出意图，也必须以JSON格式输出，绝对不要用自然语言回复意图操作结果
</instruction>

<supported_intents>
# vehicle_door - 车门控制
- intent: vehicle_door
- description: 车门控制，支持打开和关闭，支持指定车门位置。不指定位置时默认操作所有车门。
- arguments: {action: [on, off], position: [front_left, front_right, rear_left, rear_right, front, rear, all]}
- example: {"intent": "vehicle_door", "arguments": {"action": "on", "position": "front_right"}}
</supported_intents>

<chat_prompts>
- 你的名字叫做联博士
- 如果用户打招呼，回复简短的问候
- 保持友好、简洁的对话风格
</chat_prompts>"""

CAR_STATUS = """\
<car_status>
空调: 关闭 | 温度: 24°C (主驾) / 24°C (副驾)
车窗: 全部关闭 | 车门: 全部关闭 | 后备箱: 关闭
座椅加热: 关闭 | 座椅通风: 关闭
灯光: 关闭 | 驾驶模式: 舒适
</car_status>"""


def image_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def chat(base_url: str, messages: list, *, max_tokens: int = 128) -> dict:
    payload = {
        "model": "qwen3.5",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser(description="VL repetition quantization accuracy test")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Server base URL")
    parser.add_argument("--turns", type=int, default=20, help="Number of VL turns (default: 20)")
    parser.add_argument("--no-proxy", action="store_true", help="Bypass proxy for localhost")
    args = parser.parse_args()

    if args.no_proxy:
        import os
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",127.0.0.1,localhost"

    base_url = args.url.rstrip("/")
    img_path = SCRIPT_DIR / "test_pointing_left.png"
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        return 1

    img_uri = image_to_data_uri(img_path)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║   VL Repetition Quantization Accuracy Test               ║")
    print(f"║   Server: {base_url:<47s} ║")
    print(f"║   Turns: {args.turns:<48d} ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    messages = [{"role": "system", "content": CAR_SYSTEM_PROMPT}]
    passed = 0
    failed = 0
    first_fail = None

    for i in range(args.turns):
        turn = i + 1
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_uri}},
            {"type": "text", "text": f"{CAR_STATUS}\n<user_input>关闭手指方向的车门</user_input>"},
        ]})

        t0 = time.time()
        resp = chat(base_url, messages)
        elapsed = time.time() - t0

        choice = resp["choices"][0]
        content = choice["message"]["content"]
        finish = choice["finish_reason"]
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens = usage.get("completion_tokens", 0)
        perf = usage.get("performance", {})
        ttft = int(perf.get("prefill_ms", 0))
        decode_ms = perf.get("decode_ms", 0)
        tps = round(gen_tokens / (decode_ms / 1000), 1) if decode_ms > 0 else 0

        ok = True
        detail = ""
        try:
            parsed = json.loads(content)
            got = parsed.get("intent", "")
            if got != "vehicle_door":
                ok = False
                detail = f"intent={got}, expected vehicle_door"
        except json.JSONDecodeError:
            ok = False
            extra_braces = content.count("}") - content.count("{")
            if extra_braces > 0:
                detail = f"extra trailing '}}' ×{extra_braces}"
            else:
                detail = f"not JSON: {content[:60]}"

        if ok:
            passed += 1
        else:
            failed += 1
            if first_fail is None:
                first_fail = turn

        status = "\033[92m✅\033[0m" if ok else "\033[91m❌\033[0m"
        cache_str = f"cache={usage.get('prefix_cache_hit', 0)}" if usage.get("prefix_cache_hit", 0) > 0 else "cold"
        print(f"  T{turn:2d}  {status}  prompt={prompt_tokens:5d}  gen={gen_tokens:2d}  "
              f"ttft={ttft:4d}ms  tps={tps:4.1f}  {cache_str:<16s}  "
              f"{'  ⚠ ' + detail if detail else ''}")

        messages.append({"role": "assistant", "content": content})

    print()
    print("═" * 64)
    print(f"  Result: {passed}/{args.turns} passed, {failed} failed")
    if first_fail:
        print(f"  First failure at turn {first_fail}")
    else:
        print("  No extra braces detected — quantization is stable ✓")
    print("═" * 64)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
