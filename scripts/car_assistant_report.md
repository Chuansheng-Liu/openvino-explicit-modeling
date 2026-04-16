# Car Assistant 20-Turn Multi-Modal Test Report

- **Model:** Qwen3.5-9B (INT4 g128)
- **Runs:** 20
- **Turns per run:** 20
- **Images:** 800×600 resolution
- **Quality:** 400/400 (100.0%)
- **Perfect runs:** 20/20

## Per-Turn Statistics

| # | Turn | Type | TTFT avg (ms) | TTFT min | TTFT max | σ | TPS avg | Cache avg | Pass Rate |
|---|------|------|--------------|----------|----------|---|---------|-----------|-----------|
| 1 | T1 chat: greeting | Chat | 1280 | 999 | 2440 | 304 | 19.7 | — | 20/20 |
| 2 | T2 text: open driver window | Text | 213 | 153 | 754 | 134 | 19.6 | 2004 | 20/20 |
| 3 | T3 text: AC off | Text | 157 | 147 | 202 | 15 | 19.8 | 2158 | 20/20 |
| 4 | T4 text: play music | Text | 154 | 147 | 164 | 4 | 19.8 | 2304 | 20/20 |
| 5 | T5 text: sport mode | Text | 160 | 149 | 206 | 12 | 19.7 | 2450 | 20/20 |
| 6 | T6 vl: open window (pointing left→front_left) | VL | 383 | 334 | 473 | 32 | 19.6 | 2597 | 20/20 |
| 7 | T7 text: next song | Text | 158 | 151 | 193 | 9 | 19.5 | 3230 | 20/20 |
| 8 | T8 text: open YouTube | Text | 168 | 153 | 232 | 20 | 19.7 | 3376 | 20/20 |
| 9 | T9 text: raise temp | Text | 156 | 149 | 172 | 5 | 19.7 | 3523 | 20/20 |
| 10 | T10 vl: what do you see? | VL | 397 | 337 | 810 | 99 | 19.6 | 3676 | 20/20 |
| 11 | T11 text: seat heating on | Text | 356 | 155 | 4079 | 854 | 19.6 | 4128 | 20/20 |
| 12 | T12 text: go home screen | Text | 158 | 151 | 210 | 12 | 19.5 | 4501 | 20/20 |
| 13 | T13 vl: close door (pointing left→front_left) | VL | 427 | 346 | 996 | 136 | 19.1 | 4643 | 20/20 |
| 14 | T14 chat: weather question | Chat | 161 | 150 | 185 | 7 | 19.0 | 5276 | 20/20 |
| 15 | T15 text: change car color to red | Text | 168 | 154 | 186 | 10 | 19.3 | 5449 | 20/20 |
| 16 | T16 text: close all windows | Text | 161 | 157 | 177 | 5 | 19.3 | 5597 | 20/20 |
| 17 | T17 text: seat ventilation on | Text | 163 | 159 | 183 | 5 | 19.1 | 5749 | 20/20 |
| 18 | T18 vl: open door (pointing left→front_left) | VL | 406 | 348 | 528 | 52 | 19.2 | 5906 | 20/20 |
| 19 | T19 text: set temp to 20 | Text | 164 | 155 | 215 | 13 | 19.4 | 6539 | 20/20 |
| 20 | T20 chat: goodbye | Chat | 184 | 158 | 268 | 30 | 18.9 | 6696 | 20/20 |

## Category Summary

| Category | Avg TTFT (ms) | P50 | P95 | Min | Max | σ | Samples |
|----------|--------------|-----|-----|-----|-----|---|---------|
| Text Intent | 180 | 158 | 194 | 147 | 4079 | 246 | 260 |
| VL | 403 | 384 | 516 | 334 | 996 | 91 | 80 |
| Chat | 541 | 178 | 1357 | 150 | 2440 | 551 | 60 |

**Throughput:** avg=19.5 t/s, min=17.8, max=20.2

## Per-Run Results

| Run | Passed | Total | Status |
|-----|--------|-------|--------|
| 1 | 20 | 20 | ✅ |
| 2 | 20 | 20 | ✅ |
| 3 | 20 | 20 | ✅ |
| 4 | 20 | 20 | ✅ |
| 5 | 20 | 20 | ✅ |
| 6 | 20 | 20 | ✅ |
| 7 | 20 | 20 | ✅ |
| 8 | 20 | 20 | ✅ |
| 9 | 20 | 20 | ✅ |
| 10 | 20 | 20 | ✅ |
| 11 | 20 | 20 | ✅ |
| 12 | 20 | 20 | ✅ |
| 13 | 20 | 20 | ✅ |
| 14 | 20 | 20 | ✅ |
| 15 | 20 | 20 | ✅ |
| 16 | 20 | 20 | ✅ |
| 17 | 20 | 20 | ✅ |
| 18 | 20 | 20 | ✅ |
| 19 | 20 | 20 | ✅ |
| 20 | 20 | 20 | ✅ |

## Sample Run (1 complete 20-turn conversation)

Below is the full output of a single run, showing the multi-turn conversation flow,
model responses, prefix cache reuse, and per-turn TTFT/throughput.

```
Turn  1 — [Chat] 你好呀 (cold start, no cache)
  prompt: 1987 tokens | ttft: 1088ms | 19.6 t/s
  → 你好呀！我是联博士，你的智能助手。有什么可以帮你的吗？                              ✅ PASS

Turn  2 — [Text Intent] 打开主驾车窗
  prompt: 2135 tokens | ttft: 164ms | 19.8 t/s | cache: 2004 tokens reused
  → {"intent": "vehicle_window", "arguments": {"action": "on", "position": "front_left"}}  ✅ PASS

Turn  3 — [Text Intent] 关闭空调
  prompt: 2287 tokens | ttft: 153ms | 19.8 t/s | cache: 2158 tokens reused
  → {"intent": "hvac_action", "arguments": {"action": "off"}}                              ✅ PASS

Turn  4 — [Text Intent] 播放音乐
  prompt: 2433 tokens | ttft: 159ms | 19.7 t/s | cache: 2304 tokens reused
  → {"intent": "music_play_action", "arguments": {"action": "on"}}                         ✅ PASS

Turn  5 — [Text Intent] 切换到运动模式
  prompt: 2580 tokens | ttft: 156ms | 19.9 t/s | cache: 2450 tokens reused
  → {"intent": "vehicle_drive_mode", "arguments": {"mode": "sport"}}                       ✅ PASS

Turn  6 — [VL Intent] 🖼️ 打开手指方向的车窗 (pointing left image)
  prompt: 3207 tokens | ttft: 516ms | 19.3 t/s | cache: 2597 tokens reused
  → {"intent": "vehicle_window", "arguments": {"action": "on", "position": "front_left"}}  ✅ PASS

Turn  7 — [Text Intent] 下一首
  prompt: 3359 tokens | ttft: 156ms | 19.3 t/s | cache: 3230 tokens reused
  → {"intent": "music_up_down", "arguments": {"action": "next"}}                           ✅ PASS

Turn  8 — [Text Intent] 打开YouTube
  prompt: 3505 tokens | ttft: 160ms | 19.4 t/s | cache: 3376 tokens reused
  → {"intent": "gui_open_app", "arguments": {"app_name": "YouTube"}}                       ✅ PASS

Turn  9 — [Text Intent] 升高温度
  prompt: 3652 tokens | ttft: 156ms | 19.3 t/s | cache: 3523 tokens reused
  → {"intent": "hvac_temp", "arguments": {"area": "all", "temp": 26}}                      ✅ PASS

Turn 10 — [VL Describe] 🖼️ 你看到了什么？ (scene photo)
  prompt: 4285 tokens | ttft: 350ms | 19.5 t/s | cache: 3676 tokens reused
  → 这张照片拍摄于一个户外广场，背景是晴朗的蓝天。画面中央有一个巨大的地球仪雕塑，     ✅ PASS
    上面有金色的"UNIVERSAL"字样，这是环球影城的标志性建筑。周围聚集了许多游客...

Turn 11 — [Text Intent] 打开座椅加热
  prompt: 4487 tokens | ttft: 157ms | 19.4 t/s | cache: 4357 tokens reused
  → {"intent": "hvac_seat_heating", "arguments": {"area": "front_left", "level": 3}}       ✅ PASS

Turn 12 — [Text Intent] 回到桌面
  prompt: 4642 tokens | ttft: 167ms | 19.3 t/s | cache: 4513 tokens reused
  → {"intent": "gui_go_home", "arguments": {}}                                             ✅ PASS

Turn 13 — [VL Intent] 🖼️ 关闭手指方向的车门 (pointing left image, cached)
  prompt: 5265 tokens | ttft: 536ms | 19.5 t/s | cache: 4655 tokens reused
  → {"intent": "vehicle_door", "arguments": {"action": "off", "position": "front_left"}}   ✅ PASS

Turn 14 — [Chat] 今天天气怎么样？
  prompt: 5419 tokens | ttft: 156ms | 19.6 t/s | cache: 5288 tokens reused
  → 关于天气情况，我目前无法直接获取外部天气信息。如果您想了解天气，                     ✅ PASS
    建议您查看手机上的天气应用...

Turn 15 — [Text Intent] 把车换成红色
  prompt: 5678 tokens | ttft: 168ms | 19.1 t/s | cache: 5546 tokens reused
  → {"intent": "vehicle_color_change", "arguments": {"color": "red"}}                      ✅ PASS

Turn 16 — [Text Intent] 把所有车窗关上
  prompt: 5825 tokens | ttft: 160ms | 19.2 t/s | cache: 5695 tokens reused
  → {"intent": "vehicle_window", "arguments": {"action": "off", "position": "all"}}        ✅ PASS

Turn 17 — [Text Intent] 打开座椅通风
  prompt: 5977 tokens | ttft: 162ms | 19.3 t/s | cache: 5847 tokens reused
  → {"intent": "hvac_seat_ventilation", "arguments": {"area": "front_left", "level": 3}}   ✅ PASS

Turn 18 — [VL Intent] 🖼️ 打开手指方向的车门 (pointing left image, cached)
  prompt: 6614 tokens | ttft: 365ms | 18.6 t/s | cache: 6004 tokens reused
  → {"intent": "vehicle_door", "arguments": {"action": "on", "position": "front_left"}}    ✅ PASS

Turn 19 — [Text Intent] 把温度调到20度
  prompt: 6770 tokens | ttft: 171ms | 18.5 t/s | cache: 6637 tokens reused
  → {"intent": "hvac_temp", "arguments": {"area": "all", "temp": 20}}                      ✅ PASS

Turn 20 — [Chat] 好的谢谢，再见
  prompt: 6925 tokens | ttft: 164ms | 19.0 t/s | cache: 6794 tokens reused
  → 不客气！再见，祝您用车愉快！                                                           ✅ PASS
```

**Result: 20/20 passed** — All intents correctly recognized, VL pointing direction correct,
prefix cache hit on 19/20 turns (only cold start misses).
