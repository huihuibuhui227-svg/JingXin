# M2.5 设计:时间基修复(让系统里的「秒」是真的)

- 状态:**待使用者审阅**(审通过后才调 writing-plans 出实施计划)
- 日期:2026-09-25
- 前置:spec §9 原本把「上游时间基修复」排为 M2,但**实际执行的 M2 是「读侧按 `session_id` 对齐」**(另一件事),这条因此从里程碑序列里掉出来了。本设计把它接回来,编号 M2.5(沿用 M1.5 的插位命名)。

---

## 0. 目标(一句话)

让系统里所有时间量**只有一个来源、且与真实时间一致** —— 实时路径用服务端实测的流逝毫秒,离线路径用 `帧序号 × FRAME_SKIP / src_fps`,两条路径共用同一个**显式入参**契约。

---

## 1. 范围

### 1.1 在内

| 模块 | 文件 |
|---|---|
| face | `face_expression/pipeline/video_pipeline.py`、`face_expression/pipeline/detector.py`、`face_expression/core/analysis/micro_expression.py` |
| face 服务 | `face_expression/api/app.py`(不再相信 `?fps=`) |
| gesture | `gesture_analysis/core/detectors.py` |
| 离线 | `experiments/extract_features.py` |
| 测试 | 新增合成时间戳单测(§7.1) |

### 1.2 不在内(明确划界)

- **前端一行不动。** 实时路径**忽略** `?fps=`(记一次警告)。
- **特征 schema 不动。** `measured_fps` **不进 CSV 列** —— 进列属于 M3。
- M3 的 L0 逐列重构;跟进项 15 / 16(见 §9)。
- `report_frontend/evidence_thresholds.json`。

---

## 2. 决策记录

| # | 决定 | 依据 |
|---|---|---|
| **D1** | 时间戳改为**显式入参** `process_frame(frame, timestamp_ms)` | 两条路径共用一个契约;单测可直接喂受控时间戳,不必 monkeypatch 时钟 |
| **D2** | 实时路径用**服务端实测墙钟**(`time.monotonic()` 差值) | 实时流里墙钟**就是**真实时间。spec 反对墙钟是针对**离线批处理** —— 那里处理慢于实时,实测中位是真时长的 **2.32 倍**、相关仅 0.666 |
| **D3** | `measured_fps` 逐会话记录,**不进 CSV 列** | 不碰特征 schema(那是 M3 的地盘) |
| **D4** | 微表情窗从「帧数窗」改「时间窗」 | 实时 fps 是实测且可变的,帧数窗在那里语义不唯一(对 spec 措辞的偏离,见 §5.3) |
| **D5** | 离线 `extract_features.py` 一并修 | 同一个 bug、同一个修法;不影响合并门(不重抽历史数据) |

---

## 3. 实测事实(2026-09-25,是设计的地基)

### 3.1 实时路径错 **30 倍**

| 环节 | 实际 |
|---|---|
| 前端 `useCamera({frameRate: 5})` | `setInterval(…, 1000/5)` → 每 **200 ms** 采一帧 |
| `onFrame` 里 `frameCountRef.current % 5 !== 0 → return` | 每 5 帧发 1 次 → **1 帧/秒** |
| `src/services/api.ts:53` | ``withSession(`${FACE_API_URL}/analyze?fps=30`)``,注释写着「fps 固定 30」 |
| `face_expression/api/app.py:227` | `get_or_create_pipeline(session_id, fps)` ← 这个 30 一路传到底 |
| `face_expression/pipeline/detector.py:101` | `detect_for_video(image, int(self._frame_index * 1000 / self.fps))` |

服务端被告知 **30 fps**、实收 **1 帧/秒** → 时间量差 **30 倍**。方向是**时间被压缩**,于是任何「每秒多少次」的率被放大 30 倍。

### 3.2 离线路径错 **3 倍**

`experiments/extract_features.py:45` `FRAME_SKIP = 3`,而 `:251` 构造 `FaceExtractor(fps=30)`(`:84` → `VideoPipeline(fps=30)`)。
跳过 3 帧后有效 ≈10 fps,管线仍按 30 计 → 每个提交帧被当成 33 ms,真实是 100 ms。

### 3.3 `fps` 现在**同时**是「元数据」与「计时依据」(改动面的全貌)

`video_pipeline.py` 里由 `fps` 派生的**时间量**共 5 处:

| 行 | 现在 | 问题 |
|---|---|---|
| `:37` | `au_history = deque(maxlen=int(3 * fps))` | 想表达「最近 3 秒」,实际是「最近 90 帧」—— 实时 1 fps 下那是 **90 秒** |
| `:70` | `eye_closed_duration += 1 / self.fps` | 累加的是**假想**帧长,不是真实间隔 |
| `:261` | `duration_min = (len(au_history) / self.fps) / 60` | 同上 |
| `:265` | `duration_sec = frame_count / self.fps` | 同上 |
| `:30` / `:34` | 传给 `FaceDetector(fps=)` / `MicroExpressionDetector(fps=)` | 下游各自再派生时间 |

另外两处**挂钟**(同一段代码里):

| 行 | 现在 | 问题 |
|---|---|---|
| `:59` | `current_time = time.time()` | 离线批处理下是**处理时间**,不是视频时间 |
| `:62` / `:66` | 眨眼去抖 0.3 s、60 秒窗,都用 `current_time` | 离线路径里「每分钟眨眼次数」算的是**处理时间**里的次数 |

### 3.4 `is_blink` 三列恒 0 的原因(位置已定位)

`video_pipeline.py:75-78` 把 `is_blink` / `blink_rate_per_min` / `eye_closed_sec` 写进了
`au_for_history`(**深拷贝**,进 `self.au_history` 供时序统计),而**序列化用的是 `current_au`**
(`:172` `au_features=current_au`)—— 那个对象从来没拿到这三个字段。

### 3.5 gesture 比 face 更隐蔽

`gesture_analysis/api/app.py` 里 `fps` **零命中** —— gesture 连参数都没有,直接用
`core/detectors.py:95` 的默认 `fps: int = 30`(`:103` 是同一个 `_timestamp_ms` 公式)。
所以「客户端至少还申报了一个 fps」这件事在 gesture 上都不成立。

### 3.6 改 `timestamp` 语义**不波及报告层**(已核)

- `report_frontend/feature_engine.py:12` —— `timestamp` 在 `_SKIP_COLS` 里,不进特征(`:174`/`:280` 也排除)。
- `report_frontend/data_loader.py:387` —— 只用它 `sort_values`。新旧两种基准都**单调**,排序结果一致。
- 唯二用 `t_start`/`t_end` 的地方(`:421-422`)在 `__main__` 的调试打印块里,**不在生产路径**。

### 3.7 ★ 一处连锁:时间基一变,M1.5 的位移表就过期

喂给 mediapipe 的时间戳**直接参与 VIDEO 模式的跟踪与平滑**(`detector.py:101`、`detectors.py:103`)。
**改时间戳 ⟹ 探测器输出会变** ⟹ M1.5 spec §3.4 那张位移表(29 个 face 特征里 11 个相对位移 >10%)
是在**当前这个错时间基**上量的,**不能直接拿来定阈值**。见 §9。

---

## 4. 架构(一帧怎么走)

```
调用方(实时 / 离线)
   │  自己算 timestamp_ms
   ▼
process_frame(frame, timestamp_ms)
   ├─ detect_for_video(image, timestamp_ms)        ← mediapipe 内部时钟 = 真实时间
   ├─ 眨眼去抖 / 60s 窗 / eye_closed_sec            ← 全用 timestamp_ms 的差值
   ├─ micro_detector.detect(au, timestamp_ms)      ← 1.5 秒时间窗
   └─ AnalysisFrameResult(timestamp=timestamp_ms)
```

**时间戳的两个来源**

| 路径 | 公式 |
|---|---|
| 实时 | `int((time.monotonic() - session_start) * 1000)`;`session_start` 由服务端在**建管线时**记下 |
| 离线 | `round(k * 1000 * FRAME_SKIP / src_fps)`;`k` = 从 0 起的**已提交**帧序号;`src_fps` 取 `cap.get(cv2.CAP_PROP_FPS)`(已有,`:276`) |

---

## 5. 接口契约

### 5.1 `VideoPipeline.process_frame(frame, timestamp_ms: int)`

- `timestamp_ms` **必须单调不减**;出现回退时抛 `ValueError`(而不是静默产出乱序历史)。
- 构造参数 `fps` 保留,但**降级为纯元数据**:它不再参与任何时间计算。
  元信息里 `fps` 的值改为**到本帧为止的滑动实测值** `n_submitted / elapsed_sec`
  —— 逐帧都在变,所以只能给滑动值;会话收尾时的终值即 §5.5 的 `measured_fps`。
  ⚠️ 它是**元信息字段,不是特征列**:`feature_engine` 不读它(见 §3.6 / §5.5)。
- `session_start` 的语义:`reset()` 时一并归零(否则下一帧时间戳回退,与 M1.5 §6.4 的同一类问题)。

### 5.2 §3.3 表里那 5 处全部改由时间戳派生

- `au_history`:`deque(maxlen=int(3 * fps))` → **3 秒时间窗**(与 §5.3 同一机制)
- `eye_closed_sec`:相邻两帧 `timestamp_ms` 的**真实差值**累加
- `duration_sec` / `duration_min`:`(末帧 timestamp_ms − 首帧 timestamp_ms) / 1000`

### 5.3 微表情窗:时间窗(**本设计对 spec 的唯一偏离**)

spec §4.1 写「窗口由 fps 导出」。本设计改成**按 `timestamp_ms` 维护的 1.5 秒窗**
(`micro_expression.py:8` 的 `maxlen=15`)。

理由:实时路径的 fps 是实测且逐会话可变的,「帧数窗」在 1 fps 下等于 15 秒、在 30 fps 下等于
0.5 秒 —— 同一个参数两种含义。**时间窗在两条路径下语义唯一。**

> 使用者已于 2026-09-25 口头同意此偏离,记此存档。

### 5.4 实时路径忽略 `?fps=`

`face_expression/api/app.py:227` 不再把 query 的 fps 传进管线;收到时记一次 `logger.warning`。
**这是契约变更,但前端不动** —— `?fps=30` 会继续被发过来,它只是不再被相信。

### 5.5 `measured_fps`(协变量,**不进 CSV**)

会话级定义:`n_submitted / elapsed_sec`。落**服务日志**(`logger.info`,在 TTL 回收与 `/reset` 收尾时各记一行)。

> **收窄(写实施计划时改的,2026-09-25)**:原设计写的是「落两处」,第二处是
> `~/shared/jingxin_recordings/<session_id>/session.json`。做计划时发现那一半要付的代价不对:
> 清单由 `voice_interaction/asr/transcript_store.py:186` 的 `ensure_manifest` 独占,而它的语义是
> 「会话开始写一次、已存在即不动」—— face/gesture 要写进去就得引入**跨模块 import** 并动
> `refresh_manifest`,而后者至今**零生产调用方**(账本 §3 第 3 条,M2 刻意没碰)。
> 为一个本轮**不进特征列**的协变量付这个耦合不划算 → 收窄为「只进服务日志」,
> 清单那一半留到 M3(协变量升为特征列时一并处理)。

记账:这是 spec §4.5「机器负载会影响的量要显式输出为协变量」的**第一条实例**。
本轮只**记录**,不进特征列 —— 进列属于 M3。

---

## 6. 错误处理(写死,不留含糊)

| 情形 | 行为 |
|---|---|
| `timestamp_ms` 回退 | 抛 `ValueError`(**不静默**) |
| 会话起始时刻缺失(管线复用但未记 start) | 视为新会话,重置并记 `logger.warning` |
| 离线 `src_fps <= 0` | 沿用 `:277-278` 的回退 30.0,但**记一次 warning**(不静默) |
| `?fps=` 被忽略 | 记一次 `logger.warning`,不报错(前端不改,报错会打断采集) |

---

## 7. 测试与验收

### 7.1 合成信号单测(新增,喂受控时间戳)

| 断言 | 输入 | 期望 |
|---|---|---|
| 眨眼率 | 0/500/1000/… ms,EAR 交替 | `blink_rate_per_min` = 真实次数换算值 |
| `eye_closed_sec` | 不等间隔 0/300/900/1000 ms | = 真实闭眼时长,**≠** `帧数/30` |
| 微表情窗 | 跨 2 秒的帧 | 1.5 秒窗内只有窗内的帧参与 |
| `duration_sec` | 0 … 9500 ms | = **9.5**,不是 `帧数/30` |
| 时间戳回退 | 单调性被破坏 | 抛 `ValueError` |

### 7.2 反向复现(账本 §4.2 的标准动作)

撤掉本次生产改动 → §7.1 每一条必须**红**,且红在预期的位置。逐条记录到账本。

### 7.3 端到端

跑一场真会话(前端 1 帧/秒),断言:

- `measured_fps` 落在 **[0.5, 2.0]**;
- 行时间戳跨度与墙钟跨度差 **< 10%**。

### 7.4 不回归

- 全量 `pytest` **224 个仍全过**。
- 合并门 `--verify-legacy` 仍 **0/2835510**。
  ⚠️ **这条是「确认没有误伤」,不是本设计的证据** —— 合并门复现的是**已有 CSV**,不重抽,
  采集代码怎么改它都绿。别把它当成验收依据(账本 §4.8)。

---

## 8. 已知未满足项与风险

1. **★ 1 fps 下「时间窗」会让时序族退化。** 这是本设计最要命的副作用,必须记账:
   `au_history` 现在是 `maxlen=int(3*30)=90` 帧 —— 在真实 1 fps 下那是 **90 秒**的历史;
   改成 3 秒时间窗后只剩 **~3 帧**。帧数一少,时序统计(方差/趋势类)就不可用。
   **也就是说:本设计把「时间」修对了,但会让实时路径的时序族在 1 fps 下变得*更*退化。**
   这不是本设计的缺陷,而是**暴露了采集率本身太低** —— 账本里「一场会话采更多帧
   (现在只有 1 帧,时序类的量至少要几十帧才有意义)」那一条,因此从「提高覆盖率的手段之一」
   **升级为时序族可用的前提**。
2. **VIDEO 模式等价性门(M1.5 §10.4 / 账本跟进项 15)仍未验**,且本设计让它的基线**再次移动**
   (§3.7)。本设计的验收**不含**它。
3. `measured_fps` 是**会话级平均**,不含抖动与丢帧分布。更细的分布留给 M3。
4. 前端仍会发 `?fps=30`(一个不再被相信的谎)。清理它属于前端仓,不在本轮。

---

## 9. 与后续里程碑的关系

- **M3(L0 逐列重构)依赖本设计**:率类列(`*_per_sec`)、眨眼时长、微表情都要一个可信的「秒」。
- **跟进项 15(VIDEO 等价性门)与 16(阈值/量程重登记)必须在本设计落地后、在新时间基上重测。**
  M1.5 §3.4 那张位移表按 §3.7 已过期。本设计落地时应**同步改账本 §3 第 15 / 16 条的依赖说明**。
- **「一场会话采更多帧」**按 §8.1 升级为时序族的前提,与 M3 并列而非从属。

---

## 10. 待办(由本设计引出,不属本轮)

1. 前端去掉 `?fps=30`(前端仓)。
2. **提高实时采集率** —— 前端 `frameRate: 5` 叠加 `% 5` 抽稀,上限就是 1 帧/秒(前端仓)。
3. `measured_fps` 提升为特征列(M3)。
4. 抖动 / 丢帧分布统计(M3)。
