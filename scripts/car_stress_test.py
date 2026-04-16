#!/usr/bin/env python3
"""Car assistant random multi-turn stress test.

Runs a continuous multi-turn conversation with randomly selected user commands
for a specified duration. Reports aggregate TTFT, throughput, cache, and quality.

Usage:
    python scripts/car_stress_test.py --duration 600    # 10 minutes
    python scripts/car_stress_test.py --duration 300 --report stress_report.md
    python scripts/car_stress_test.py --no-proxy --verbose
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Helpers ──────────────────────────────────────────────────────────

def image_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def chat(base_url: str, messages: list, *, max_tokens: int = 128,
         model: str = "qwen3.5") -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


# ── System prompt (same as car_assistant_test.py) ────────────────────

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

# vehicle_window - 车窗控制
- intent: vehicle_window
- description: 车窗控制，支持打开和关闭，支持指定车窗位置。不指定位置时默认操作所有车窗。
- arguments: {action: [on, off], position: [front_left, front_right, rear_left, rear_right, front, rear, all]}
- example: {"intent": "vehicle_window", "arguments": {"action": "on", "position": "front_right"}}

# vehicle_trunk - 后备箱控制
- intent: vehicle_trunk
- description: 后备箱控制，支持打开和关闭
- arguments: {action: [on, off]}
- example: {"intent": "vehicle_trunk", "arguments": {"action": "on"}}

# vehicle_light - 车灯控制
- intent: vehicle_light
- description: 车灯控制，支持打开和关闭
- arguments: {action: [on, off]}
- example: {"intent": "vehicle_light", "arguments": {"action": "on"}}

# vehicle_drive_mode - 驾驶模式控制
- intent: vehicle_drive_mode
- description: 驾驶模式控制，支持在经济/舒适/运动三种模式之间切换
- arguments: {mode: [eco, comfort, sport]}
- example: {"intent": "vehicle_drive_mode", "arguments": {"mode": "sport"}}

# vehicle_color_change - 改变车辆颜色
- intent: vehicle_color_change
- description: 改变车辆颜色，支持的颜色有 橙/红/白/银/黑
- arguments: {color: [orange, red, white, silver, black]}
- example: {"intent": "vehicle_color_change", "arguments": {"color": "orange"}}

# hvac_action - 空调控制
- intent: hvac_action
- description: 空调控制，支持打开和关闭
- arguments: {action: [on, off]}
- example: {"intent": "hvac_action", "arguments": {"action": "on"}}

# hvac_temp - 空调温度控制
- intent: hvac_temp
- description: 空调温度控制，area可选主驾（左前）/副驾（右前）/全部，默认为全部，支持的温度为16到32度。当用户说升高/降低温度时，基于<car_status>中的空调温度进行相对调整（默认±2度）
- arguments: {area: [front_left, front_right, all], temp: int[16-32]}
- example: {"intent": "hvac_temp", "arguments": {"area": "all", "temp": 25}}

# hvac_seat_heating - 座椅加热控制
- intent: hvac_seat_heating
- description: 座椅加热控制，area可选左前和右前，默认level为3，level为0表示关闭
- arguments: {area: [front_left, front_right], level: int[0-3]}
- example: {"intent": "hvac_seat_heating", "arguments": {"area": "front_left", "level": 3}}

# hvac_seat_ventilation - 座椅通风控制
- intent: hvac_seat_ventilation
- description: 座椅通风控制，area可选左前和右前，默认level为3，level为0表示关闭
- arguments: {area: [front_left, front_right], level: int[0-3]}
- example: {"intent": "hvac_seat_ventilation", "arguments": {"area": "front_left", "level": 3}}

# music_play_action - 音乐播放控制
- intent: music_play_action
- description: 音乐播放控制，支持播放和暂停
- arguments: {action: [on, off]}
- example: {"intent": "music_play_action", "arguments": {"action": "on"}}

# music_up_down - 音乐切换
- intent: music_up_down
- description: 音乐切换，支持上一曲和下一曲
- arguments: {action: [prev, next]}
- example: {"intent": "music_up_down", "arguments": {"action": "prev"}}

# gui_go_home - 返回gui主页
- intent: gui_go_home
- description: 返回gui主页
- arguments: {}
- example: {"intent": "gui_go_home", "arguments": {}}

# gui_open_app - 打开应用
- intent: gui_open_app
- description: 在<car_status>中寻找已安装应用并打开
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_open_app", "arguments": {"app_name": "xxxxx"}}

# gui_close_app - 关闭应用
- intent: gui_close_app
- description: 关闭<car_status>中的已安装应用
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_close_app", "arguments": {"app_name": "xxxxx"}}
</supported_intents>

<chat_prompts>
你的名字叫联博士，是联想汽车的智能助手。你需要：
- 友好、专业地与用户交流
- 回答简洁明了
- 如果不确定用户意图，礼貌地请用户重新表述
</chat_prompts>"""


def make_car_status(hvac="开启", tl=24, tr=24):
    return (
        f"<car_status>\n空调状态: {hvac}\n空调温度: 主驾{tl}°C, 副驾{tr}°C\n"
        "车窗状态: 全部关闭\n车门状态: 全部关闭\n车灯状态: 关闭\n驾驶模式: 舒适模式\n"
        "后备箱: 关闭\n座椅加热: 关闭\n座椅通风: 关闭\n当前播放: 无\n"
        "已安装应用: YouTube, 爱奇艺, 网易云音乐, 高德地图, 微信\n</car_status>"
    )


# ── Test case pool ───────────────────────────────────────────────────
# Each entry: (label, user_text, type, expect_intent, expect_field, is_vl, max_tokens)
# is_vl: "pointing" uses pointing image, "scene" uses scene image, None = text only

TEXT_CASES = [
    ("打开主驾车窗", "text", "vehicle_window", {"action": "on", "position": "front_left"}),
    ("关闭所有车窗", "text", "vehicle_window", {"action": "off", "position": "all"}),
    ("打开副驾车窗", "text", "vehicle_window", {"action": "on", "position": "front_right"}),
    ("打开后排车窗", "text", "vehicle_window", {"action": "on", "position": "rear"}),
    ("关闭主驾车门", "text", "vehicle_door", {"action": "off", "position": "front_left"}),
    ("打开后备箱", "text", "vehicle_trunk", {"action": "on"}),
    ("关闭后备箱", "text", "vehicle_trunk", {"action": "off"}),
    ("开灯", "text", "vehicle_light", {"action": "on"}),
    ("关灯", "text", "vehicle_light", {"action": "off"}),
    ("切换到运动模式", "text", "vehicle_drive_mode", {"mode": "sport"}),
    ("切换到经济模式", "text", "vehicle_drive_mode", {"mode": "eco"}),
    ("切换到舒适模式", "text", "vehicle_drive_mode", {"mode": "comfort"}),
    ("把车换成红色", "text", "vehicle_color_change", {"color": "red"}),
    ("把车换成黑色", "text", "vehicle_color_change", {"color": "black"}),
    ("关闭空调", "text", "hvac_action", {"action": "off"}),
    ("打开空调", "text", "hvac_action", {"action": "on"}),
    ("升高温度", "text", "hvac_temp", {"area": "all", "temp": 26}),
    ("降低温度", "text", "hvac_temp", {"area": "all", "temp": 22}),
    ("把温度调到20度", "text", "hvac_temp", {"area": "all", "temp": 20}),
    ("打开座椅加热", "text", "hvac_seat_heating", {"area": "front_left", "level": 3}),
    ("关闭座椅加热", "text", "hvac_seat_heating", {"area": "front_left", "level": 0}),
    ("打开座椅通风", "text", "hvac_seat_ventilation", {"area": "front_left", "level": 3}),
    ("播放音乐", "text", "music_play_action", {"action": "on"}),
    ("暂停音乐", "text", "music_play_action", {"action": "off"}),
    ("下一首", "text", "music_up_down", {"action": "next"}),
    ("上一首", "text", "music_up_down", {"action": "prev"}),
    ("打开YouTube", "text", "gui_open_app", {"app_name": "YouTube"}),
    ("打开高德地图", "text", "gui_open_app", {"app_name": "高德地图"}),
    ("关闭爱奇艺", "text", "gui_close_app", {"app_name": "爱奇艺"}),
    ("回到桌面", "text", "gui_go_home", None),
]

CHAT_CASES = [
    "你好呀",
    "今天天气怎么样？",
    "你叫什么名字？",
    "给我讲个笑话",
    "现在几点了？",
    "帮我推荐一首歌",
    "你能做什么？",
    "谢谢你",
]

VL_POINTING_CASES = [
    ("打开手指方向的车窗", "vehicle_window", {"action": "on", "position": "front_left"}),
    ("关闭手指方向的车门", "vehicle_door", {"action": "off", "position": "front_left"}),
    ("打开手指方向的车门", "vehicle_door", {"action": "on", "position": "front_left"}),
]


# ── Main loop ────────────────────────────────────────────────────────

def run_stress(base_url: str, duration_sec: int, verbose: bool):
    pointing_img = SCRIPT_DIR / "test_pointing_left.png"
    pointing_uri = image_to_data_uri(pointing_img) if pointing_img.exists() else None
    scene_img = SCRIPT_DIR / "test_800x600.jpg"
    scene_uri = image_to_data_uri(scene_img) if scene_img.exists() else None

    messages = [{"role": "system", "content": CAR_SYSTEM_PROMPT}]
    car_status = make_car_status()

    # Stats
    turn_data = []  # list of dicts
    start_time = time.time()
    turn_num = 0
    conversation_num = 1
    max_conversation_turns = 30  # reset conversation after this many turns

    print(f"Starting stress test for {duration_sec}s...")
    print(f"{'─'*90}")

    while (time.time() - start_time) < duration_sec:
        turn_num += 1
        elapsed_total = time.time() - start_time

        # Reset conversation if too long (avoid unbounded prompt growth)
        if len(messages) > max_conversation_turns * 2 + 1:
            messages = [{"role": "system", "content": CAR_SYSTEM_PROMPT}]
            conversation_num += 1
            print(f"\n{'▓'*90}")
            print(f"  New conversation #{conversation_num} (turn {turn_num}, {elapsed_total:.0f}s elapsed)")
            print(f"{'▓'*90}")

        # Randomly pick a test case type: 60% text intent, 15% VL pointing, 10% VL scene, 15% chat
        r = random.random()
        if r < 0.60:
            # Text intent
            case = random.choice(TEXT_CASES)
            user_text, typ, expect_intent, expect_field = case
            user_content = f"{car_status}\n<user_input>{user_text}</user_input>"
            max_tok = 128
            is_vl = False
            label = f"text: {user_text}"
        elif r < 0.75 and pointing_uri:
            # VL pointing
            case = random.choice(VL_POINTING_CASES)
            user_text, expect_intent, expect_field = case
            user_content = [
                {"type": "text", "text": f"{car_status}\n<user_input>"},
                {"type": "image_url", "image_url": {"url": pointing_uri}},
                {"type": "text", "text": f"{user_text}</user_input>"},
            ]
            max_tok = 128
            is_vl = True
            typ = "vl"
            label = f"vl: {user_text}"
        elif r < 0.85 and scene_uri:
            # VL scene describe
            user_content = [
                {"type": "text", "text": f"{car_status}\n<user_input>"},
                {"type": "image_url", "image_url": {"url": scene_uri}},
                {"type": "text", "text": "你看到了什么？</user_input>"},
            ]
            max_tok = 256
            is_vl = True
            typ = "vl-describe"
            expect_intent = None
            expect_field = None
            label = "vl: 你看到了什么？"
        else:
            # Chat
            user_text = random.choice(CHAT_CASES)
            user_content = f"{car_status}\n<user_input>{user_text}</user_input>"
            max_tok = 128
            is_vl = False
            typ = "chat"
            expect_intent = None
            expect_field = None
            label = f"chat: {user_text}"

        messages.append({"role": "user", "content": user_content})

        t0 = time.time()
        try:
            resp = chat(base_url, messages, max_tokens=max_tok)
            req_time = time.time() - t0

            c = resp["choices"][0]
            content = c["message"].get("content", "") or ""
            u = resp.get("usage", {})
            perf = u.get("performance", {})
            ttft = perf.get("ttft_ms", 0)
            tps = perf.get("throughput_tps", 0)
            cached = perf.get("prefix_cached_tokens", 0)
            prompt_tokens = u.get("prompt_tokens", 0)
            gen_tokens = u.get("completion_tokens", 0)

            # Validate
            ok = True
            detail = ""
            clean = content.strip()
            if "<think>" in clean:
                clean = re.sub(r"<think>.*?</think>\s*", "", clean, flags=re.DOTALL).strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                clean = "\n".join(lines).strip()

            if expect_intent:
                try:
                    parsed = json.loads(clean)
                    got = parsed.get("intent", "")
                    if got != expect_intent:
                        ok = False
                        detail = f"intent={got}, expected {expect_intent}"
                except json.JSONDecodeError:
                    ok = False
                    detail = f"not JSON: {clean[:80]}"

            turn_info = {
                "turn": turn_num, "conv": conversation_num, "label": label,
                "type": typ, "is_vl": is_vl, "ttft": ttft, "tps": tps,
                "cached": cached, "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens, "ok": ok, "detail": detail,
                "elapsed": elapsed_total, "content": clean[:200],
            }
            turn_data.append(turn_info)

            # Append assistant response
            messages.append({"role": "assistant", "content": content})

            status = "\033[92m✅\033[0m" if ok else "\033[91m❌\033[0m"
            cache_str = f"cache={cached}" if cached > 0 else "cold"
            vl_str = " 🖼️" if is_vl else ""
            print(f"  T{turn_num:<4d} {status} {label:<30s}{vl_str}  ttft={ttft:>6.0f}ms  tps={tps:>5.1f}  {cache_str:<14s}  prompt={prompt_tokens}  [{elapsed_total:.0f}s]")
            if not ok:
                print(f"         ⚠ {detail}")
            if verbose:
                print(f"         → {clean[:120]}")

        except Exception as e:
            req_time = time.time() - t0
            print(f"  T{turn_num:<4d} \033[91mERROR\033[0m {label}  — {e}")
            turn_data.append({
                "turn": turn_num, "conv": conversation_num, "label": label,
                "type": typ, "is_vl": is_vl, "ttft": 0, "tps": 0,
                "cached": 0, "prompt_tokens": 0, "gen_tokens": 0,
                "ok": False, "detail": str(e), "elapsed": elapsed_total,
                "content": "",
            })
            messages.append({"role": "assistant", "content": ""})

    total_time = time.time() - start_time
    return turn_data, total_time, conversation_num


def print_and_write_report(turn_data, total_time, num_convs, report_path=None):
    total = len(turn_data)
    passed = sum(1 for d in turn_data if d["ok"])

    # Category breakdown
    cats = {}
    for d in turn_data:
        t = d["type"]
        if t not in cats:
            cats[t] = {"ttfts": [], "tps": [], "ok": 0, "total": 0}
        cats[t]["ttfts"].append(d["ttft"])
        cats[t]["tps"].append(d["tps"])
        cats[t]["ok"] += 1 if d["ok"] else 0
        cats[t]["total"] += 1

    cache_hits = sum(1 for d in turn_data if d["cached"] > 0)
    all_ttfts = [d["ttft"] for d in turn_data]
    all_tps = [d["tps"] for d in turn_data]

    # Console output
    print(f"\n\n{'═'*90}")
    print(f"STRESS TEST RESULTS — {total_time:.0f}s, {total} turns, {num_convs} conversations")
    print(f"{'═'*90}")

    print(f"\n{'Category':<20s} {'Count':>6s} {'Pass':>6s} {'TTFT avg':>9s} {'P50':>7s} {'P95':>7s} {'Min':>7s} {'Max':>7s} {'TPS avg':>8s}")
    print(f"{'─'*20} {'─'*6} {'─'*6} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for cat in sorted(cats.keys()):
        c = cats[cat]
        ttfts = sorted(c["ttfts"])
        avg = sum(ttfts) / len(ttfts)
        p50 = ttfts[len(ttfts)//2]
        p95 = ttfts[int(len(ttfts)*0.95)]
        mn, mx = min(ttfts), max(ttfts)
        avg_tps = sum(c["tps"]) / len(c["tps"])
        print(f"{cat:<20s} {c['total']:>6d} {c['ok']:>6d} {avg:>8.0f}ms {p50:>6.0f} {p95:>6.0f} {mn:>6.0f} {mx:>6.0f} {avg_tps:>7.1f}")

    # Overall
    avg_ttft = sum(all_ttfts) / len(all_ttfts)
    avg_tps_all = sum(all_tps) / len(all_tps)
    p50_all = sorted(all_ttfts)[len(all_ttfts)//2]
    p95_all = sorted(all_ttfts)[int(len(all_ttfts)*0.95)]

    print(f"\n  Total turns    : {total}")
    print(f"  Quality        : {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"  Cache hit rate : {cache_hits}/{total} ({cache_hits/total*100:.1f}%)")
    print(f"  Overall TTFT   : avg={avg_ttft:.0f}ms, p50={p50_all:.0f}ms, p95={p95_all:.0f}ms")
    print(f"  Overall TPS    : avg={avg_tps_all:.1f}")
    print(f"  Throughput     : {total/total_time:.1f} turns/min ({total_time:.0f}s)")
    print(f"  Conversations  : {num_convs}")

    # Failed turns
    failures = [d for d in turn_data if not d["ok"]]
    if failures:
        print(f"\n  Failed turns ({len(failures)}):")
        for d in failures[:20]:
            print(f"    T{d['turn']}: {d['label']} — {d['detail']}")

    # Markdown report
    if report_path:
        lines = []
        lines.append("# Car Assistant Stress Test Report\n")
        lines.append(f"- **Duration:** {total_time:.0f}s ({total_time/60:.1f} min)")
        lines.append(f"- **Total turns:** {total}")
        lines.append(f"- **Conversations:** {num_convs} (reset every 30 turns)")
        lines.append(f"- **Quality:** {passed}/{total} ({passed/total*100:.1f}%)")
        lines.append(f"- **Cache hit rate:** {cache_hits}/{total} ({cache_hits/total*100:.1f}%)")
        lines.append(f"- **Model:** Qwen3.5-9B (INT4 g128)")
        lines.append(f"- **Images:** 800×600\n")

        lines.append("## Category Summary\n")
        lines.append("| Category | Count | Pass | TTFT avg (ms) | P50 | P95 | Min | Max | TPS avg |")
        lines.append("|----------|-------|------|--------------|-----|-----|-----|-----|---------|")
        for cat in sorted(cats.keys()):
            c = cats[cat]
            ttfts = sorted(c["ttfts"])
            avg = sum(ttfts) / len(ttfts)
            p50 = ttfts[len(ttfts)//2]
            p95 = ttfts[int(len(ttfts)*0.95)]
            mn, mx = min(ttfts), max(ttfts)
            avg_tps = sum(c["tps"]) / len(c["tps"])
            lines.append(f"| {cat} | {c['total']} | {c['ok']} | {avg:.0f} | {p50:.0f} | {p95:.0f} | {mn:.0f} | {mx:.0f} | {avg_tps:.1f} |")

        lines.append(f"\n**Overall:** TTFT avg={avg_ttft:.0f}ms, p50={p50_all:.0f}ms, p95={p95_all:.0f}ms, TPS avg={avg_tps_all:.1f}\n")

        if failures:
            lines.append(f"## Failures ({len(failures)})\n")
            lines.append("| Turn | Label | Detail |")
            lines.append("|------|-------|--------|")
            for d in failures:
                lines.append(f"| T{d['turn']} | {d['label']} | {d['detail'][:100]} |")

        # Sample turns (first 20)
        lines.append("\n## Sample Turns (first 20)\n")
        lines.append("| # | Type | User Input | TTFT (ms) | TPS | Cache | Prompt | Result |")
        lines.append("|---|------|-----------|-----------|-----|-------|--------|--------|")
        for d in turn_data[:20]:
            cache_s = str(d["cached"]) if d["cached"] > 0 else "—"
            status = "✅" if d["ok"] else "❌"
            short_label = d["label"][:25]
            lines.append(f"| {d['turn']} | {d['type']} | {short_label} | {d['ttft']:.0f} | {d['tps']:.1f} | {cache_s} | {d['prompt_tokens']} | {status} |")

        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n📄 Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Car assistant random stress test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--report", type=str, default=None, help="Write markdown report")
    args = parser.parse_args()

    if args.no_proxy:
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",127.0.0.1,localhost"

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║       Car Assistant Random Multi-Turn Stress Test           ║")
    print(f"║  Server: {args.base_url:<51s} ║")
    print(f"║  Duration: {args.duration}s ({args.duration/60:.0f} min){' '*(42-len(str(args.duration)))}║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    turn_data, total_time, num_convs = run_stress(
        args.base_url, args.duration, args.verbose
    )
    print_and_write_report(turn_data, total_time, num_convs, args.report)

    passed = sum(1 for d in turn_data if d["ok"])
    sys.exit(0 if passed == len(turn_data) else 1)


if __name__ == "__main__":
    main()
