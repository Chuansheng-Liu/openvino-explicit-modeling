#!/usr/bin/env python3
"""Car assistant test suite for ov_serve.

Tests automotive intent recognition with VL (vision-language) and text inputs.
The system prompt defines a car assistant that recognizes intents from user
speech + camera images (e.g. hand pointing direction → window/door control).

Usage:
    python scripts/car_assistant_test.py                          # default localhost:8080
    python scripts/car_assistant_test.py --base-url http://host:8093
    python scripts/car_assistant_test.py --verbose
    python scripts/car_assistant_test.py --no-proxy
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json",
                                  "Authorization": "Bearer test"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat(base_url: str, messages: list, *, max_tokens: int = 256,
         model: str = "qwen3.5") -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return post_json(f"{base_url}/v1/chat/completions", payload)


# ── Result helpers ───────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str, resp: dict):
        self.name = name
        self.resp = resp
        c = resp["choices"][0]
        self.finish_reason = c["finish_reason"]
        self.msg = c["message"]
        self.content = self.msg.get("content", "") or ""
        u = resp.get("usage", {})
        self.prompt_tokens = u.get("prompt_tokens", 0)
        self.completion_tokens = u.get("completion_tokens", 0)
        perf = u.get("performance", {})
        self.tps = perf.get("throughput_tps", 0)
        self.ttft = perf.get("ttft_ms", 0)
        self.prefix_cached_tokens = perf.get("prefix_cached_tokens", 0)

    def summary(self, verbose: bool = False) -> str:
        lines = [
            f"  finish_reason : {self.finish_reason}",
            f"  tokens        : {self.prompt_tokens} prompt + {self.completion_tokens} gen",
            f"  throughput    : {self.tps:.1f} t/s, ttft: {self.ttft:.0f}ms",
        ]
        if self.prefix_cached_tokens > 0:
            lines.append(f"  prefix cache  : {self.prefix_cached_tokens} tokens reused")
        if verbose:
            text = self.content[:500] + ("..." if len(self.content) > 500 else "")
            lines.append(f"  content       : {text}")
        return "\n".join(lines)


# ── System prompt ────────────────────────────────────────────────────

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
# video_ui - 视频UI的sub_agent
- intent: video_ui
- description: 视频UI的sub_agent，当车机使用YouTube, 爱奇艺等应用播放视频时调用，将意图转发给video_ui处理。
- 视频播放器内（如YouTube）的所有操作都可由video_ui处理，如：视频搜索、视频播放/暂停/继续、打开页面、进入分栏、开/关字幕、全屏/退出全屏、关闭视频页面/迷你播放器、调节画质、点赞/踩、取消点赞/踩、倍速播放、订阅/关注、保存至稍后观看、上一个/下一个、发表/查看/关闭评论、打开作者主页、不感兴趣、查看视频简介、跳过广告、点击按钮（click xxx button）等
- arguments: {}
- example: {"intent": "video_ui", "arguments": {}}

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
- description: 音乐播放控制，支持播放和暂停。常见的表达有：播放音乐，Play a Music，暂停播放等
- arguments: {action: [on, off]}
- example: {"intent": "music_play_action", "arguments": {"action": "on"}}

# music_up_down - 音乐切换
- intent: music_up_down
- description: 音乐切换，支持上一曲和下一曲
- arguments: {action: [prev, next]}
- example: {"intent": "music_up_down", "arguments": {"action": "prev"}}

# gui_go_home - 返回gui主页
- intent: gui_go_home
- description: 返回gui主页，也就是把所有打开的应用都切换到后台。常见的表达有：回到桌面，回到主页等
- arguments: {}
- example: {"intent": "gui_go_home", "arguments": {}}

# gui_open_app - 打开应用
- intent: gui_open_app
- description: 在<car_status>中寻找已安装应用并打开。常见的表达有：Open YouTube，Fire up YouTube，打开网易云音乐等
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_open_app", "arguments": {"app_name": "xxxxx"}}

# gui_close_app - 关闭应用
- intent: gui_close_app
- description: 关闭<car_status>中的已安装应用。常见的表达有：Close YouTube，关闭网易云音乐等
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_close_app", "arguments": {"app_name": "xxxxx"}}
</supported_intents>

<chat_prompts>
你的名字叫联博士，是联想汽车的智能助手。你需要：
- 友好、专业地与用户交流
- 回答简洁明了
- 如果不确定用户意图，礼貌地请用户重新表述
</chat_prompts>"""

CAR_STATUS_TEMPLATE = """\
<car_status>
空调状态: {hvac_status}
空调温度: 主驾{temp_left}°C, 副驾{temp_right}°C
车窗状态: 全部关闭
车门状态: 全部关闭
车灯状态: 关闭
驾驶模式: 舒适模式
后备箱: 关闭
座椅加热: 关闭
座椅通风: 关闭
当前播放: 无
已安装应用: YouTube, 爱奇艺, 网易云音乐, 高德地图, 微信
</car_status>"""


def make_car_status(hvac_status="开启", temp_left=24, temp_right=24):
    return CAR_STATUS_TEMPLATE.format(
        hvac_status=hvac_status, temp_left=temp_left, temp_right=temp_right
    )


# ── Test definitions ─────────────────────────────────────────────────

def run_tests(base_url: str, verbose: bool) -> tuple[list[tuple[str, bool, str]], list]:
    results: list[tuple[str, bool, str]] = []
    all_trs: list[tuple[str, object]] = []  # (name, TestResult) for stats
    test_num = 0

    def run(name, messages, *, max_tokens=128,
            expect_intent=None, expect_field=None,
            expect_content_contains=None, expect_chat=False):
        nonlocal test_num
        test_num += 1
        label = f"[{test_num:2d}] {name}"
        t0 = time.time()
        try:
            resp = chat(base_url, messages, max_tokens=max_tokens)
            tr = TestResult(label, resp)
            elapsed = time.time() - t0
            print(f"\n{'─'*60}")
            print(f"{'✓' if True else '✗'} {label}  ({elapsed:.1f}s)")
            print(tr.summary(verbose=verbose))

            ok = True
            detail = ""

            # Parse JSON from content (strip think tags and markdown fences)
            content = tr.content.strip()
            # Remove <think>...</think> wrapper if present
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            # Remove ```json ... ``` markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                # Drop first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            if expect_intent:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Model may append extra text after JSON — extract first {...}
                    parsed = None
                    if "{" in content:
                        start = content.index("{")
                        depth = 0
                        for i in range(start, len(content)):
                            if content[i] == "{":
                                depth += 1
                            elif content[i] == "}":
                                depth -= 1
                                if depth == 0:
                                    try:
                                        parsed = json.loads(content[start:i+1])
                                    except json.JSONDecodeError:
                                        pass
                                    break
                    if parsed is None:
                        ok = False
                        detail = f"response is not valid JSON: {content[:200]}"

                if parsed is not None and expect_intent:
                    got_intent = parsed.get("intent", "")
                    if got_intent != expect_intent:
                        ok = False
                        detail = f"expected intent={expect_intent}, got {got_intent}"
                    if expect_field:
                        args = parsed.get("arguments", {})
                        for k, v in expect_field.items():
                            got = args.get(k)
                            if got != v:
                                ok = False
                                detail += f"; expected {k}={v}, got {got}"

            if expect_content_contains:
                if expect_content_contains.lower() not in content.lower():
                    ok = False
                    detail = f"expected content to contain '{expect_content_contains}'"

            if expect_chat:
                # For chat responses, just verify it's NOT valid JSON intent
                try:
                    parsed = json.loads(content)
                    if "intent" in parsed:
                        ok = False
                        detail = "expected chat response but got JSON intent"
                except json.JSONDecodeError:
                    pass  # Good — it's a chat response

            status = "PASS" if ok else "FAIL"
            color = "\033[92m" if ok else "\033[91m"
            reset = "\033[0m"
            msg = f"{color}{status}{reset} {label}"
            if detail:
                msg += f" — {detail}"
            print(msg)
            results.append((label, ok, detail))
            all_trs.append((name, tr))
            return tr

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n{'─'*60}")
            print(f"\033[91mERROR\033[0m {label}  ({elapsed:.1f}s) — {e}")
            results.append((label, False, str(e)))
            return None

    car_status = make_car_status()
    pointing_img = SCRIPT_DIR / "test_pointing_left.png"
    pointing_uri = image_to_data_uri(pointing_img) if pointing_img.exists() else None
    test_img = SCRIPT_DIR / "test_800x600.jpg"
    test_uri = image_to_data_uri(test_img) if test_img.exists() else None

    # ── Multi-turn conversation simulating a real car session ──
    # Each turn builds on the previous, so prefix cache benefits accumulate.
    # The conversation mixes text intents, VL intents, and chat.

    messages = [{"role": "system", "content": CAR_SYSTEM_PROMPT}]

    TURNS = [
        # (name, user_content, expect_intent, expect_field, expect_chat, expect_contains, max_tokens)
        # Turn 1: greeting (cold start, no cache)
        ("T1 chat: greeting",
         f"{car_status}\n<user_input>你好呀</user_input>",
         None, None, True, None, 128),

        # Turn 2: text intent — open driver window
        ("T2 text: open driver window",
         f"{car_status}\n<user_input>打开主驾车窗</user_input>",
         "vehicle_window", {"action": "on", "position": "front_left"}, False, None, 128),

        # Turn 3: text intent — turn off AC
        ("T3 text: AC off",
         f"{car_status}\n<user_input>关闭空调</user_input>",
         "hvac_action", {"action": "off"}, False, None, 128),

        # Turn 4: text intent — play music
        ("T4 text: play music",
         f"{car_status}\n<user_input>播放音乐</user_input>",
         "music_play_action", {"action": "on"}, False, None, 128),

        # Turn 5: text intent — sport mode
        ("T5 text: sport mode",
         f"{car_status}\n<user_input>切换到运动模式</user_input>",
         "vehicle_drive_mode", {"mode": "sport"}, False, None, 128),

        # Turn 6: VL — hand pointing left → open window
        ("T6 vl: open window (pointing left→front_left)",
         [{"type": "text", "text": f"{car_status}\n<user_input>"},
          {"type": "image_url", "image_url": {"url": pointing_uri}},
          {"type": "text", "text": "打开手指方向的车窗</user_input>"}] if pointing_uri else
         f"{car_status}\n<user_input>打开主驾方向的车窗</user_input>",
         "vehicle_window", {"action": "on", "position": "front_left"}, False, None, 128),

        # Turn 7: text intent — next song
        ("T7 text: next song",
         f"{car_status}\n<user_input>下一首</user_input>",
         "music_up_down", {"action": "next"}, False, None, 128),

        # Turn 8: text intent — open YouTube
        ("T8 text: open YouTube",
         f"{car_status}\n<user_input>打开YouTube</user_input>",
         "gui_open_app", {"app_name": "YouTube"}, False, None, 128),

        # Turn 9: text intent — raise temp
        ("T9 text: raise temp",
         f"{make_car_status(temp_left=24, temp_right=24)}\n<user_input>升高温度</user_input>",
         "hvac_temp", {"area": "all", "temp": 26}, False, None, 128),

        # Turn 10: VL — describe what camera sees
        ("T10 vl: what do you see?",
         [{"type": "text", "text": f"{car_status}\n<user_input>"},
          {"type": "image_url", "image_url": {"url": test_uri}},
          {"type": "text", "text": "你看到了什么？</user_input>"}] if test_uri else
         f"{car_status}\n<user_input>你看到了什么？</user_input>",
         None, None, False, None, 256),

        # Turn 11: text intent — seat heating
        ("T11 text: seat heating on",
         f"{car_status}\n<user_input>打开座椅加热</user_input>",
         "hvac_seat_heating", {"area": "front_left", "level": 3}, False, None, 128),

        # Turn 12: text intent — go home
        ("T12 text: go home screen",
         f"{car_status}\n<user_input>回到桌面</user_input>",
         "gui_go_home", None, False, None, 128),

        # Turn 13: VL — pointing left → close door
        ("T13 vl: close door (pointing left→front_left)",
         [{"type": "text", "text": f"{car_status}\n<user_input>"},
          {"type": "image_url", "image_url": {"url": pointing_uri}},
          {"type": "text", "text": "关闭手指方向的车门</user_input>"}] if pointing_uri else
         f"{car_status}\n<user_input>关闭主驾方向的车门</user_input>",
         "vehicle_door", {"action": "off", "position": "front_left"}, False, None, 128),

        # Turn 14: chat — ask about weather
        ("T14 chat: weather question",
         f"{car_status}\n<user_input>今天天气怎么样？</user_input>",
         None, None, True, None, 128),

        # Turn 15: text intent — change color
        ("T15 text: change car color to red",
         f"{car_status}\n<user_input>把车换成红色</user_input>",
         "vehicle_color_change", {"color": "red"}, False, None, 128),

        # Turn 16: text intent — close all windows
        ("T16 text: close all windows",
         f"{car_status}\n<user_input>把所有车窗关上</user_input>",
         "vehicle_window", {"action": "off", "position": "all"}, False, None, 128),

        # Turn 17: text intent — seat ventilation
        ("T17 text: seat ventilation on",
         f"{car_status}\n<user_input>打开座椅通风</user_input>",
         "hvac_seat_ventilation", {"area": "front_left", "level": 3}, False, None, 128),

        # Turn 18: VL — pointing left → open door
        ("T18 vl: open door (pointing left→front_left)",
         [{"type": "text", "text": f"{car_status}\n<user_input>"},
          {"type": "image_url", "image_url": {"url": pointing_uri}},
          {"type": "text", "text": "打开手指方向的车门</user_input>"}] if pointing_uri else
         f"{car_status}\n<user_input>打开主驾方向的车门</user_input>",
         "vehicle_door", {"action": "on", "position": "front_left"}, False, None, 128),

        # Turn 19: text intent — lower temp to specific value
        ("T19 text: set temp to 20",
         f"{make_car_status(temp_left=26, temp_right=26)}\n<user_input>把温度调到20度</user_input>",
         "hvac_temp", {"area": "all", "temp": 20}, False, None, 128),

        # Turn 20: chat — say goodbye
        ("T20 chat: goodbye",
         f"{car_status}\n<user_input>好的谢谢，再见</user_input>",
         None, None, True, None, 128),
    ]

    for (name, user_content, expect_intent, expect_field,
         expect_chat, expect_contains, max_tokens) in TURNS:
        messages.append({"role": "user", "content": user_content})

        tr = run(name, messages, max_tokens=max_tokens,
                 expect_intent=expect_intent, expect_field=expect_field,
                 expect_chat=expect_chat, expect_content_contains=expect_contains)

        # Append assistant response to build the conversation
        resp_content = tr.content if tr else ""
        messages.append({"role": "assistant", "content": resp_content})

    return results, all_trs


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Car assistant intent recognition test suite for ov_serve.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080",
                        help="Base URL of ov_serve (default: http://127.0.0.1:8080)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full response content")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Set no_proxy for localhost")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs (each run = full 20-turn conversation)")
    parser.add_argument("--report", type=str, default=None,
                        help="Write markdown report to this file path")
    args = parser.parse_args()

    if args.no_proxy:
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",127.0.0.1,localhost"

    num_runs = args.runs

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║       Car Assistant Intent Recognition Tests            ║")
    print(f"║  Server: {args.base_url:<47s} ║")
    print(f"║  Runs: {num_runs:<49d} ║")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # Collect data across all runs
    # per_turn_data[turn_idx] = {"name":..., "type":..., "ttfts":[], "tps_list":[], "caches":[], "passes":[], "contents":[]}
    per_turn_data: dict[int, dict] = {}
    all_run_results = []  # list of (passed, total) per run

    for run_idx in range(num_runs):
        print(f"\n{'▓'*60}")
        print(f"  Run {run_idx + 1}/{num_runs}")
        print(f"{'▓'*60}")

        results, all_trs = run_tests(args.base_url, verbose=args.verbose)
        passed = sum(1 for _, ok, _ in results if ok)
        total = len(results)
        all_run_results.append((passed, total))

        color = "\033[92m" if passed == total else "\033[91m"
        reset = "\033[0m"
        print(f"\n{color}Run {run_idx+1}: {passed}/{total} passed{reset}")

        for i, (name, tr) in enumerate(all_trs):
            if i not in per_turn_data:
                is_vl = "vl" in name.lower()
                is_chat = "chat" in name.lower()
                typ = "VL" if is_vl else ("Chat" if is_chat else "Text")
                per_turn_data[i] = {
                    "name": name, "type": typ,
                    "ttfts": [], "tps_list": [], "caches": [],
                    "passes": [], "contents": [],
                }
            d = per_turn_data[i]
            d["ttfts"].append(tr.ttft)
            d["tps_list"].append(tr.tps)
            d["caches"].append(tr.prefix_cached_tokens)
            d["passes"].append(results[i][1])
            d["contents"].append(tr.content.strip()[:200])

    # ── Aggregate Statistics ──
    print(f"\n\n{'═'*80}")
    print(f"AGGREGATE STATISTICS ({num_runs} runs × 20 turns)")
    print(f"{'═'*80}")

    # Per-turn table
    print(f"\n{'Turn':<45s} {'Type':<5s} {'TTFT avg':>8s} {'min':>6s} {'max':>6s} {'σ':>6s} {'TPS':>6s} {'Pass':>6s}")
    print(f"{'─'*45} {'─'*5} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    ttft_text_all, ttft_vl_all, ttft_chat_all = [], [], []
    tps_all_all = []
    total_passes = 0
    total_tests = 0

    for i in sorted(per_turn_data.keys()):
        d = per_turn_data[i]
        ttfts = d["ttfts"]
        tps_list = d["tps_list"]
        passes = d["passes"]
        avg_ttft = sum(ttfts) / len(ttfts)
        min_ttft = min(ttfts)
        max_ttft = max(ttfts)
        std_ttft = (sum((t - avg_ttft)**2 for t in ttfts) / len(ttfts)) ** 0.5
        avg_tps = sum(tps_list) / len(tps_list)
        pass_rate = f"{sum(passes)}/{len(passes)}"

        print(f"T{i+1:<2d} {d['name']:<42s} {d['type']:<5s} {avg_ttft:>7.0f}ms {min_ttft:>5.0f} {max_ttft:>5.0f} {std_ttft:>5.0f} {avg_tps:>5.1f} {pass_rate:>6s}")

        tps_all_all.extend(tps_list)
        total_passes += sum(passes)
        total_tests += len(passes)
        if d["type"] == "VL":
            ttft_vl_all.extend(ttfts)
        elif d["type"] == "Chat":
            ttft_chat_all.extend(ttfts)
        else:
            ttft_text_all.extend(ttfts)

    # Category summary
    print(f"\n{'─'*80}")
    print("Category Summary:")
    for label, data in [("Text Intent", ttft_text_all), ("VL Intent/Describe", ttft_vl_all), ("Chat", ttft_chat_all)]:
        if data:
            avg = sum(data) / len(data)
            mn = min(data)
            mx = max(data)
            std = (sum((t - avg)**2 for t in data) / len(data)) ** 0.5
            p50 = sorted(data)[len(data)//2]
            p95 = sorted(data)[int(len(data)*0.95)]
            print(f"  {label:<20s}: avg={avg:>6.0f}ms  p50={p50:>6.0f}ms  p95={p95:>6.0f}ms  min={mn:>5.0f}  max={mx:>5.0f}  σ={std:>5.0f}  (n={len(data)})")
    if tps_all_all:
        avg_tps = sum(tps_all_all) / len(tps_all_all)
        print(f"  {'Throughput':<20s}: avg={avg_tps:.1f} t/s")

    perfect_runs = sum(1 for p, t in all_run_results if p == t)
    print(f"\n  Quality: {total_passes}/{total_tests} correct ({total_passes/total_tests*100:.1f}%)")
    print(f"  Perfect runs: {perfect_runs}/{num_runs}")

    # ── Write markdown report ──
    if args.report:
        import statistics
        lines = []
        lines.append("# Car Assistant 20-Turn Multi-Modal Test Report\n")
        lines.append(f"- **Model:** Qwen3.5-9B (INT4 g128)")
        lines.append(f"- **Runs:** {num_runs}")
        lines.append(f"- **Turns per run:** {len(per_turn_data)}")
        lines.append(f"- **Images:** 800×600 resolution")
        lines.append(f"- **Quality:** {total_passes}/{total_tests} ({total_passes/total_tests*100:.1f}%)")
        lines.append(f"- **Perfect runs:** {perfect_runs}/{num_runs}\n")

        # Per-turn table
        lines.append("## Per-Turn Statistics\n")
        lines.append("| # | Turn | Type | TTFT avg (ms) | TTFT min | TTFT max | σ | TPS avg | Cache avg | Pass Rate |")
        lines.append("|---|------|------|--------------|----------|----------|---|---------|-----------|-----------|")
        for i in sorted(per_turn_data.keys()):
            d = per_turn_data[i]
            ttfts = d["ttfts"]
            avg_ttft = sum(ttfts) / len(ttfts)
            min_ttft = min(ttfts)
            max_ttft = max(ttfts)
            std_ttft = (sum((t - avg_ttft)**2 for t in ttfts) / len(ttfts)) ** 0.5
            avg_tps = sum(d["tps_list"]) / len(d["tps_list"])
            avg_cache = sum(d["caches"]) / len(d["caches"])
            pass_n = sum(d["passes"])
            cache_str = f"{avg_cache:.0f}" if avg_cache > 0 else "—"
            lines.append(f"| {i+1} | {d['name']} | {d['type']} | {avg_ttft:.0f} | {min_ttft:.0f} | {max_ttft:.0f} | {std_ttft:.0f} | {avg_tps:.1f} | {cache_str} | {pass_n}/{len(d['passes'])} |")

        # Category summary
        lines.append("\n## Category Summary\n")
        lines.append("| Category | Avg TTFT (ms) | P50 | P95 | Min | Max | σ | Samples |")
        lines.append("|----------|--------------|-----|-----|-----|-----|---|---------|")
        for label, data in [("Text Intent", ttft_text_all), ("VL", ttft_vl_all), ("Chat", ttft_chat_all)]:
            if data:
                avg = sum(data) / len(data)
                mn = min(data)
                mx = max(data)
                std = (sum((t - avg)**2 for t in data) / len(data)) ** 0.5
                p50 = sorted(data)[len(data)//2]
                p95 = sorted(data)[int(len(data)*0.95)]
                lines.append(f"| {label} | {avg:.0f} | {p50:.0f} | {p95:.0f} | {mn:.0f} | {mx:.0f} | {std:.0f} | {len(data)} |")

        if tps_all_all:
            avg_tps = sum(tps_all_all) / len(tps_all_all)
            min_tps = min(tps_all_all)
            max_tps = max(tps_all_all)
            lines.append(f"\n**Throughput:** avg={avg_tps:.1f} t/s, min={min_tps:.1f}, max={max_tps:.1f}\n")

        # Per-run summary
        lines.append("## Per-Run Results\n")
        lines.append("| Run | Passed | Total | Status |")
        lines.append("|-----|--------|-------|--------|")
        for ri, (p, t) in enumerate(all_run_results):
            status = "✅" if p == t else "❌"
            lines.append(f"| {ri+1} | {p} | {t} | {status} |")

        report_path = Path(args.report)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n📄 Report written to: {report_path}")

    sys.exit(0 if all(p == t for p, t in all_run_results) else 1)


if __name__ == "__main__":
    main()
