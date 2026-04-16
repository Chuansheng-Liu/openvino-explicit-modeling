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

| # | Type | User Input | Prompt | TTFT (ms) | TPS | Cache | Model Output | Result |
|---|------|-----------|--------|-----------|-----|-------|-------------|--------|
| 1 | Chat | 你好呀 | 1987 | 1088 | 19.6 | — | 你好呀！我是联博士，你的智能助手。有什么可以帮你的吗？ | ✅ |
| 2 | Text | 打开主驾车窗 | 2135 | 164 | 19.8 | 2004 | `{"intent":"vehicle_window","arguments":{"action":"on","position":"front_left"}}` | ✅ |
| 3 | Text | 关闭空调 | 2287 | 153 | 19.8 | 2158 | `{"intent":"hvac_action","arguments":{"action":"off"}}` | ✅ |
| 4 | Text | 播放音乐 | 2433 | 159 | 19.7 | 2304 | `{"intent":"music_play_action","arguments":{"action":"on"}}` | ✅ |
| 5 | Text | 切换到运动模式 | 2580 | 156 | 19.9 | 2450 | `{"intent":"vehicle_drive_mode","arguments":{"mode":"sport"}}` | ✅ |
| 6 | 🖼️ VL | 打开手指方向的车窗 | 3207 | 516 | 19.3 | 2597 | `{"intent":"vehicle_window","arguments":{"action":"on","position":"front_left"}}` | ✅ |
| 7 | Text | 下一首 | 3359 | 156 | 19.3 | 3230 | `{"intent":"music_up_down","arguments":{"action":"next"}}` | ✅ |
| 8 | Text | 打开YouTube | 3505 | 160 | 19.4 | 3376 | `{"intent":"gui_open_app","arguments":{"app_name":"YouTube"}}` | ✅ |
| 9 | Text | 升高温度 | 3652 | 156 | 19.3 | 3523 | `{"intent":"hvac_temp","arguments":{"area":"all","temp":26}}` | ✅ |
| 10 | 🖼️ VL | 你看到了什么？ | 4285 | 350 | 19.5 | 3676 | 这张照片拍摄于一个户外广场...巨大的地球仪雕塑，上面有"UNIVERSAL"字样，这是环球影城的标志性建筑... | ✅ |
| 11 | Text | 打开座椅加热 | 4487 | 157 | 19.4 | 4357 | `{"intent":"hvac_seat_heating","arguments":{"area":"front_left","level":3}}` | ✅ |
| 12 | Text | 回到桌面 | 4642 | 167 | 19.3 | 4513 | `{"intent":"gui_go_home","arguments":{}}` | ✅ |
| 13 | 🖼️ VL | 关闭手指方向的车门 | 5265 | 536 | 19.5 | 4655 | `{"intent":"vehicle_door","arguments":{"action":"off","position":"front_left"}}` | ✅ |
| 14 | Chat | 今天天气怎么样？ | 5419 | 156 | 19.6 | 5288 | 关于天气情况，我目前无法直接获取外部天气信息。建议您查看手机上的天气应用... | ✅ |
| 15 | Text | 把车换成红色 | 5678 | 168 | 19.1 | 5546 | `{"intent":"vehicle_color_change","arguments":{"color":"red"}}` | ✅ |
| 16 | Text | 把所有车窗关上 | 5825 | 160 | 19.2 | 5695 | `{"intent":"vehicle_window","arguments":{"action":"off","position":"all"}}` | ✅ |
| 17 | Text | 打开座椅通风 | 5977 | 162 | 19.3 | 5847 | `{"intent":"hvac_seat_ventilation","arguments":{"area":"front_left","level":3}}` | ✅ |
| 18 | 🖼️ VL | 打开手指方向的车门 | 6614 | 365 | 18.6 | 6004 | `{"intent":"vehicle_door","arguments":{"action":"on","position":"front_left"}}` | ✅ |
| 19 | Text | 把温度调到20度 | 6770 | 171 | 18.5 | 6637 | `{"intent":"hvac_temp","arguments":{"area":"all","temp":20}}` | ✅ |
| 20 | Chat | 好的谢谢，再见 | 6925 | 164 | 19.0 | 6794 | 不客气！再见，祝您用车愉快！ | ✅ |

**Result: 20/20 passed** — All intents correctly recognized, VL pointing direction correct,
prefix cache hit on 19/20 turns (only cold start misses).
