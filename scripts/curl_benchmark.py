#!/usr/bin/env python3
"""Quick curl-style e2e latency benchmark for car assistant intent recognition.

Uses unbuffered socket streaming for accurate per-token timing.
Two modes:
  1. Single-prompt (default): repeat same prompt N times, measure cold/warm TTFT.
  2. Prefix-cache (--prefix-cache): send N different user prompts sequentially,
     all sharing the same system prompt.  Measures prefix cache hit speedup.

Usage:
    python scripts/curl_benchmark.py                          # single-prompt mode
    python scripts/curl_benchmark.py --host 192.168.1.100 --port 8093
    python scripts/curl_benchmark.py --runs 5 --verbose
    python scripts/curl_benchmark.py --prompt "turn on the AC"
    python scripts/curl_benchmark.py --prefix-cache            # prefix-cache mode
    python scripts/curl_benchmark.py --prefix-cache --num-prompts 5
"""

from __future__ import annotations

import argparse
import json
import socket
import time


# ── System prompt (same as car_assistant_test.py) ────────────────────

SYSTEM_PROMPT = """\
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
- arguments: {}
- example: {"intent": "video_ui", "arguments": {}}

# vehicle_door - 车门控制
- intent: vehicle_door
- description: 车门控制，支持打开(on)和关闭(off)，支持指定车门位置。不指定位置时默认操作所有车门。打开=on，关闭=off。
- arguments: {action: [on, off], position: [front_left, front_right, rear_left, rear_right, front, rear, all]}
- example: {"intent": "vehicle_door", "arguments": {"action": "on", "position": "front_right"}}

# vehicle_window - 车窗控制
- intent: vehicle_window
- description: 车窗控制，支持打开(on)和关闭(off)，支持指定车窗位置。不指定位置时默认操作所有车窗。打开=on，关闭=off。
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
- description: 空调温度控制，area可选主驾（左前）/副驾（右前）/全部，默认为全部，支持的温度为16到32度。
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
- description: 打开已安装应用
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_open_app", "arguments": {"app_name": "xxxxx"}}

# gui_close_app - 关闭应用
- intent: gui_close_app
- description: 关闭已安装应用
- arguments: {app_name: xxxxx}
- example: {"intent": "gui_close_app", "arguments": {"app_name": "xxxxx"}}
</supported_intents>

<chat_prompts>
你的名字叫联博士，是联想汽车的智能助手。你需要：
- 友好、专业地与用户交流
- 回答简洁明了
- 如果不确定用户意图，礼貌地请用户重新表述
- **重要：闲聊时必须用自然语言回复，绝对不要输出JSON格式！**
</chat_prompts>"""

CAR_STATUS = """\
<car_status>
空调状态: 开启
空调温度: 主驾24°C, 副驾24°C
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


def stream_request(host: str, port: int, messages: list,
                   max_tokens: int = 200) -> dict:
    """Send streaming request via raw socket, return per-token timing."""
    payload = json.dumps({
        "model": "default",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Authorization: Bearer test\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"\r\n"
    ).encode() + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect((host, port))

    t0 = time.perf_counter()
    sock.sendall(request)

    raw = b""
    first_token_time = None
    last_token_time = None
    tokens: list[str] = []
    header_done = False

    while True:
        try:
            chunk = sock.recv(1)
            if not chunk:
                break
        except socket.timeout:
            break

        raw += chunk

        if not header_done:
            if b"\r\n\r\n" in raw:
                header_done = True
                _, raw = raw.split(b"\r\n\r\n", 1)
            continue

        while b"\n" in raw:
            line, raw = raw.split(b"\n", 1)
            line = line.decode(errors="replace").strip()
            if line == "data: [DONE]":
                sock.close()
                total = (time.perf_counter() - t0) * 1000
                ttft = (first_token_time - t0) * 1000 if first_token_time else total
                last_t = (last_token_time - t0) * 1000 if last_token_time else total
                decode_ms = last_t - ttft if len(tokens) > 1 else 0
                tps = (len(tokens) - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
                return {
                    "ttft_ms": ttft, "last_token_ms": last_t, "e2e_ms": total,
                    "tokens": len(tokens), "tps": tps,
                    "output": "".join(tokens),
                }

            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                    choices = obj.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                            tokens.append(content)
                except Exception:
                    pass

    sock.close()
    total = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": (first_token_time - t0) * 1000 if first_token_time else total,
        "last_token_ms": total, "e2e_ms": total,
        "tokens": len(tokens), "tps": 0,
        "output": "".join(tokens),
    }


PREFIX_CACHE_PROMPTS = [
    ("开灯", "vehicle_light"),
    ("打开空调", "hvac_action"),
    ("把温度调到26度", "hvac_temp"),
    ("打开车门", "vehicle_door"),
    ("播放音乐", "music_play_action"),
    ("切换到运动模式", "vehicle_drive_mode"),
    ("打开后备箱", "vehicle_trunk"),
    ("关闭车窗", "vehicle_window"),
    ("下一首歌", "music_up_down"),
    ("打开YouTube", "gui_open_app"),
]


def run_single_prompt_benchmark(args):
    """Original mode: repeat the same prompt N times, measure cold/warm."""
    user_content = args.prompt
    if not args.no_car_status:
        user_content = f"{CAR_STATUS}\n<user_input>{args.prompt}</user_input>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    print(f"{'═'*65}")
    print(f"  Car Assistant E2E Latency Benchmark — Single Prompt")
    print(f"  Server: {args.host}:{args.port}  |  Runs: {args.runs}")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"{'═'*65}")
    print(f"\n  {'Run':<10s} {'TTFT':>8s} {'Last Token':>11s} {'E2E':>8s} {'Tokens':>7s} {'TPS':>7s}")
    print(f"  {'─'*10} {'─'*8} {'─'*11} {'─'*8} {'─'*7} {'─'*7}")

    ttfts, e2es = [], []
    for i in range(args.runs):
        label = "cold" if i == 0 else "warm"
        r = stream_request(args.host, args.port, messages)
        tag = f"Run {i+1} ({label})"
        print(f"  {tag:<10s} {r['ttft_ms']:>7.0f}ms {r['last_token_ms']:>10.0f}ms {r['e2e_ms']:>7.0f}ms {r['tokens']:>7d} {r['tps']:>6.1f}")
        if args.verbose:
            print(f"    Output: {r['output'][:200]}")
        ttfts.append(r["ttft_ms"])
        e2es.append(r["e2e_ms"])
        time.sleep(0.5)

    print(f"\n  {'─'*55}")
    if len(ttfts) > 1:
        warm_ttfts = ttfts[1:]
        warm_e2es = e2es[1:]
        avg_ttft = sum(warm_ttfts) / len(warm_ttfts)
        avg_e2e = sum(warm_e2es) / len(warm_e2es)
        print(f"  Cold:  TTFT={ttfts[0]:.0f}ms  E2E={e2es[0]:.0f}ms")
        print(f"  Warm:  TTFT avg={avg_ttft:.0f}ms  E2E avg={avg_e2e:.0f}ms  (n={len(warm_ttfts)})")
    else:
        print(f"  TTFT={ttfts[0]:.0f}ms  E2E={e2es[0]:.0f}ms")
    print(f"{'═'*65}")


def _run_prefix_phase(args, prompts, phase_name):
    """Run a list of prompts and return results + print table."""
    print(f"\n  {'#':<4s} {'Prompt':<22s} {'TTFT':>8s} {'E2E':>8s} {'Tokens':>7s} {'TPS':>7s} {'Output'}")
    print(f"  {'─'*4} {'─'*22} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*30}")

    results = []
    quality_issues = []
    for i, (prompt, expected_intent) in enumerate(prompts):
        user_content = f"{CAR_STATUS}\n<user_input>{prompt}</user_input>"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        r = stream_request(args.host, args.port, messages)

        # Strip <think>...</think> tags for validation
        import re as _re
        clean = _re.sub(r"<think>.*?</think>\s*", "", r["output"], flags=_re.DOTALL).strip()
        # Remove markdown code fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        output_short = clean.replace("\n", " ")[:40]

        # Proper JSON intent validation
        intent_ok = False
        if not clean:
            quality_issues.append(f"  ⚠ #{i+1} '{prompt}': empty output")
        else:
            try:
                parsed_json = json.loads(clean)
                intent_ok = parsed_json.get("intent") == expected_intent
            except json.JSONDecodeError:
                # Try extracting first {...} block
                if "{" in clean:
                    start = clean.index("{")
                    depth = 0
                    for idx in range(start, len(clean)):
                        if clean[idx] == "{":
                            depth += 1
                        elif clean[idx] == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    pj = json.loads(clean[start:idx+1])
                                    intent_ok = pj.get("intent") == expected_intent
                                except Exception:
                                    pass
                                break

        marker = "✅" if intent_ok else "❌"

        print(f"  {i+1:<4d} {prompt:<22s} {r['ttft_ms']:>7.0f}ms {r['e2e_ms']:>7.0f}ms {r['tokens']:>7d} {r['tps']:>6.1f} {marker} {output_short}")
        results.append(r)
        time.sleep(0.3)

    ttfts = [r["ttft_ms"] for r in results]
    print(f"\n  {'─'*60}")
    avg_ttft = sum(ttfts) / len(ttfts)
    min_ttft = min(ttfts)
    max_ttft = max(ttfts)
    print(f"  {phase_name}: TTFT avg={avg_ttft:.0f}ms  min={min_ttft:.0f}ms  max={max_ttft:.0f}ms  (n={len(ttfts)})")

    # Perf consistency warning
    if len(ttfts) >= 3:
        for j, t in enumerate(ttfts):
            if t > avg_ttft * 3 and t > 500:
                print(f"  ⚠ PERF OUTLIER: #{j+1} TTFT={t:.0f}ms > 3x avg ({avg_ttft:.0f}ms)")

    for qi in quality_issues:
        print(qi)

    return results


def run_prefix_cache_benchmark(args):
    """Prefix-cache benchmark with two phases:

    Phase 1 — Batch (one session): All N prompts sent sequentially in one
    invocation.  The first few requests build the prefix snapshot; later
    requests should hit the cache and show lower TTFT.

    Phase 2 — Standalone (one prompt per invocation): Each prompt is sent
    individually.  Since the prefix snapshot persists in the server session,
    every request should hit the cache.
    """
    prompts = PREFIX_CACHE_PROMPTS[:args.num_prompts]

    print(f"{'═'*70}")
    print(f"  Prefix-Cache Benchmark")
    print(f"  Server: {args.host}:{args.port}  |  Prompts: {len(prompts)}")
    print(f"  Shared prefix: system_prompt + car_status (~{len(SYSTEM_PROMPT) + len(CAR_STATUS)} chars)")
    print(f"{'═'*70}")

    # Phase 1: all prompts in one batch
    print(f"\n  ▶ Phase 1: Batch — all {len(prompts)} prompts in one session")
    batch_results = _run_prefix_phase(args, prompts, "Batch")

    # Phase 2: each prompt as a standalone request
    print(f"\n  ▶ Phase 2: Standalone — one prompt per invocation")
    standalone_results = _run_prefix_phase(args, prompts, "Standalone")

    # Combined summary
    batch_ttfts = [r["ttft_ms"] for r in batch_results]
    standalone_ttfts = [r["ttft_ms"] for r in standalone_results]
    # For batch, skip warmup requests (first 3) for "cached" avg
    cached_start = min(3, len(batch_ttfts) - 1)
    batch_cached = batch_ttfts[cached_start:] if cached_start < len(batch_ttfts) else batch_ttfts
    batch_cached_avg = sum(batch_cached) / len(batch_cached) if batch_cached else 0
    standalone_avg = sum(standalone_ttfts) / len(standalone_ttfts)

    print(f"\n  {'━'*60}")
    print(f"  Summary")
    print(f"  {'━'*60}")
    print(f"  Batch cached (req {cached_start+1}+):  TTFT avg={batch_cached_avg:.0f}ms")
    print(f"  Standalone (all {len(prompts)}):      TTFT avg={standalone_avg:.0f}ms")
    print(f"{'═'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick e2e latency benchmark for car assistant (streaming, unbuffered socket)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for single-prompt mode (default: 3)")
    parser.add_argument("--prompt", default="open the door", help="User prompt for single-prompt mode")
    parser.add_argument("--verbose", action="store_true", help="Print model output")
    parser.add_argument("--no-car-status", action="store_true", help="Skip car_status in user message")
    parser.add_argument("--prefix-cache", action="store_true",
                        help="Run prefix-cache benchmark: sequential different prompts sharing system prompt")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts for prefix-cache mode (default: 10, max: 10)")
    args = parser.parse_args()
    args.num_prompts = min(args.num_prompts, len(PREFIX_CACHE_PROMPTS))

    if args.prefix_cache:
        run_prefix_cache_benchmark(args)
    else:
        run_single_prompt_benchmark(args)


if __name__ == "__main__":
    main()
