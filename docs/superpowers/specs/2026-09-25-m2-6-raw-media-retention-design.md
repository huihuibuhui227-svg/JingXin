# M2.6 设计:原始媒体留存(把不可逆的那一步先做掉)

- 状态:**待使用者审阅**(审通过后才调 writing-plans 出实施计划)
- 日期:2026-09-25
- 前置:本设计由 **M3 的 brainstorming**(2026-09-25)得出。brainstorming 期间发现 M3 的验收素材
  **在盘上不存在、也没有任何代码能产出**,而 M3 自己那份 spec 把「回归保护」当成了现成的
  (见 §3.1,那句假设是空的)。故按 brainstorming 的规矩先分解、再走第一个子项目 —— 这个
  子项目就是 M2.6,编号沿用 M1.5 / M2.5 的插位命名。

---

## 0. 目标(一句话)

**让「这一场」的原始素材不再只存在于内存里** —— 三路服务把真正收到/解码过的字节原样落盘到
仓库外的共享盘,前端另存一份原生帧率的音视频,从而使 3–5 场录制**可以开始**,并使 M3 的 L0 列
**事后来得及重抽**。

---

## 1. 范围

### 1.1 在内

| 面 | 文件 |
|---|---|
| 新的共用件 | `media_retention.py`(仓库根,**新增**) |
| 帧留存(face) | `face_expression/api/app.py`(`/analyze`) |
| 帧留存(gesture) | `gesture_analysis/api/app.py`(`/analyze`) |
| 音频留存 | `voice_interaction/api/app.py`(`/asr`、`/interview/answer_audio`、`/research/answer_audio`) |
| 原生视频上传端点 | `voice_interaction/api/app.py`(`POST /session/{sid}/media`,**新增端点**) |
| 前端录制 | `~/JingXin-frontend`:`src/hooks/useCamera.ts`、`src/services/api.ts` |
| 重抽回放器 | `experiments/replay_retained.py`(**新增**;供 §7.1 的验收用。单测级的回放件放 `tests/`) |

### 1.2 不在内(明确划界)

| 不做 | 为什么 |
|---|---|
| **不改任何 L0 列的定义** | 那是 M3。M2.6 只增加落盘,不碰一个特征值 |
| **不动 `report_frontend/`** | 报告层与 `evidence_thresholds.json` 的对齐归 M3.4(§9) |
| **不动 `code_data_supplement/`** | 实测:`git ls-files code_data_supplement/` → **0 个文件**。它是**别人的论文复现包**(CueCoT supplement),不是本仓代码。那两处重复的 `extract_features.py` / `voice_interaction/` 是它带来的 |
| **不迁移离线 `GestureExtractor`** | M2.5 已裁定延后(它仍直连已删除的 `mp.solutions`)。M2.6 的重抽只覆盖 face 与语音 |
| **不做 VAD / 不做停顿切分** | 归 M3(§9) |
| **不落盘原始文本** | M1 的既定边界:原句只进 `~/shared`,不进仓库。M2.6 维持不变 |

---

## 2. 决策记录(2026-09-25,使用者逐条裁定)

| # | 问题 | 裁定 | 归属 |
|---|---|---|---|
| R1 | M3 靠什么自证? | **并行**:先补留存,M3 边做边等录制。理由:留存是**唯一不可逆**的一步,且录制是长周期、要使用者出镜的那条腿 | M3 |
| R2 | 留存存什么保真度? | **两者都要**:服务端存帧(管线所见等价)+ 前端录原生视频(帧率上限不被钉死) | M2.6 |
| R3 | M3 改完列后报告层跟不跟? | **同步对齐**(封停名单重对齐到 M3 处置表)。注:L1 是 M4,M3 的新列在 L1 公式之前本就不出分 | M3 |
| R4 | M3 的内部切分 | 路线 **A**:留存摘成独立的 M2.6 → M3.1 仪器标定(跟进项 15/16)→ M3.2 面部 / M3.3 手势 / M3.4 语音 → M3.5 收尾(报告对齐 + 阈值重登记) | 两个里程碑 |
| R5 | `camera.webm` 带不带声音? | **带**(`useCamera` 的 `getUserMedia` 由 `audio: false` 改 `true`)。前提是实测两路 `getUserMedia` 的音频流不打架(见 §8.2) | M2.6 |

---

## 3. 实测事实(2026-09-25,是设计的地基)

### 3.1 ★ M3 那份 spec 假设的「回归保护」对 M3 是空转

`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md` §8.1 写:
「回归保护(已建立,直接复用)……**改采集代码后必须仍然通过**」。

实测不成立:`experiments/duration_audit/reaggregate_normalized.py` 读的是
`FEATURES_DIR.glob("vid_*")` 的**存量 CSV**(`:294`/`:413` `pd.read_csv`),**不重抽**。
M2.5 账本 Task 9 Step 2 已自行裁定过这一点。**结论:合并门 0/2835510 复现的是已有 CSV,
采集代码怎么改都绿 —— 它对 M3 不构成任何证据。**

### 3.2 ★ 盘上零原始媒体,且没有任何代码在写

- `~/shared/jingxin_recordings/` 全部会话目录里**只有** `session.json` / `transcript.json`。
- 全仓检查:**无** `VideoWriter`、生产路径**无** `cv2.imwrite`、**无** `sf.write`;
  出现的 `wave.open` **全部是 `'rb'`**(读入即弃)。
- `data/input/` 是空的;全仓无 `*.mp4` / `*.wav`。

⟹ 录制需求文档
(`2026-09-22-jingxin-recording-requirements.md`)§2 前提 2 的论断因此在今天**完全成立**:
*没有原始媒体 → 缺陷被永久烘进数据,事后无法重抽*。

### 3.3 真实会话目前只有 1–5 帧

`data/logs/face_au_log_20260925_*.csv` 实测 **6 行**(表头 + 5 行);9 月 24 日那批是 **2 行**。
与「一场会话 1 帧」的记录一致。**3–5 场录制会是本系统第一次拿到多帧真会话。**

### 3.4 采集端的真实节奏与形态(决定「原始」是什么意思)

| 事实 | 位置 |
|---|---|
| `useCamera` 默认 `frameRate = 1` | `src/hooks/useCamera.ts:8` |
| **`AssessmentPage` 传 `frameRate: 5`** | `src/components/assessment/AssessmentPage.tsx:120` |
| `RealtimeAnalysis` 传 5,但 `% 10 !== 0` 就 return → **0.5 fps** | `src/pages/RealtimeAnalysis.tsx:89`、`:23` |
| 取流 `video: {1280x720}, audio: false` | `src/hooks/useCamera.ts:20-22` |
| 逐帧 `canvas.toBlob(..., 'image/jpeg', 0.8)` | `src/hooks/useCamera.ts:150` |

⚠️ **口径更正**:M2.5 账本 F8 写「实时演示页 `% 10` 实际是 **0.1 fps**」,那是按 `useCamera` 的
**默认 `frameRate = 1`** 算的;而 `RealtimeAnalysis.tsx:89` 实际传的是 **5** ⟹ 真值是 **0.5 fps**。
两处数字不一样,以本表为准(账本里的历史记录不改,§4.6)。

⟹ **架构上不存在「原始视频文件」这个东西**:浏览器把摄像头画面经 canvas **重编码为 JPEG(q0.8)** 后逐帧 POST。
所以录制需求 §3.1「每路摄像头一份原始文件,**不做任何转码/裁剪**」**在当前架构下无法字面满足**
—— 采集端**已经在**转码(**已经在**降帧)。这正是 R2 要「两者都要」的原因:服务端那一腿留的是
**管线真正解码过的那些字节**(重抽等价性的基准),前端那一腿留的才是**未被 5 fps 钉死的原生帧率**。

### 3.5 两个 CV 服务的收帧形态是现成的钩子

`face_expression/api/app.py:234-237` 与 `gesture_analysis/api/app.py:254-257` **都是**
`@app.post("/analyze")` + `file: UploadFile = File(...)`,且 M1 已让它们**从 query 或 form** 认
`session_id`。⟹ 留存钩子有现成落点,不需要改前端就能拿到逐帧字节。

### 3.6 `~/shared` 落在 Windows 盘上(§4.7 的坑正好绕开)

`findmnt -T ~/shared` → `/mnt/d`,`fstype 9p`,`D:\` 477G / **可用 127G**。
⟹ 写这里**不会撑 WSL 的 `ext4.vhdx`**(§4.7:WSL 内删文件不还 Windows 磁盘)。
录制需求 §4 要求的路径也正是这里。

### 3.7 既有可复用的三件东西(避免重造)

`voice_interaction/asr/transcript_store.py` 已经有了留存所需的全部语义:

| 原语 | 位置 | 复用方式 |
|---|---|---|
| `DEFAULT_ROOT = ~/shared/jingxin_recordings` | `:17` | 同一默认根 |
| `JINGXIN_RECORDINGS_DIR` 环境变量覆盖 | `:83` | 同一开关(**§7.3 的测试靠它**) |
| `validate_session_id`(路径穿越守卫) | `:26` | 同一守卫 |
| `_session_lock`(逐会话锁) | `:64` | 同一并发语义 |
| `recording_dir(session_id, root)` | `:86` | 同一目录推导 |

且仓库根已有**扁平共用件**的约定:`logging_config.py`、`session_clock.py`(由 M2.5 建立),
服务侧写作 `from session_clock import SessionClock`(`face_expression/api/app.py:12`)。
⟹ `media_retention.py` 照这个样造,**不跨服务 import**(不让 face 去 import voice 的包)。

---

## 4. 架构(一段素材怎么走)

```
写侧(三路服务,各自独立)
  face :8000   /analyze   ← JPEG 字节 ─┐
  gesture :8002 /analyze  ← JPEG 字节 ─┤  ① 先落盘(原样字节)
                                        │  ② 再送进 mediapipe(照旧走临时文件)
  voice :8001  /asr            ← webm ──┤  ├ 存原始 webm
                                        │  └ 另存 ffmpeg 转出的 _converted.wav ← 提取器真正读的 PCM
               /interview/answer_audio  ← wav ─┤ 存原始
               /research/answer_audio   ← wav ─┘
               /session/{sid}/media     ← webm ── 前端摄像头原生音视频(R5)

落盘(~shared/jingxin_recordings/{session_id}/,9p → D:\)
  media/
    face/000001.jpg …      原始字节,不转码不缩放不裁剪
    gesture/000001.jpg …
    audio/0001.webm …      到达的原始上传字节
    audio/0001_converted.wav   提取器真正读的 PCM
    camera.webm            前端 MediaRecorder 原生音视频(含音轨,R5)
  retention.jsonl          每件一行,见 §5.2
```

**关键顺序**:① 落盘 → ② 分析。落盘用**同一份 bytes**,不再读一次请求体
(face 侧 `request.form()` 有「不二次消费请求体」的既有讲究,见 `app.py:145-155` 的注释)。
原有的 `tempfile.NamedTemporaryFile(suffix='.jpg')` 解码路径**保持不变** —— 留存是**旁路新增的一次写**,
不是替换解码源。这样留存关掉时行为与今天逐字节相同(正交、可证)。

---

## 5. 接口契约

### 5.1 `media_retention.py`(仓库根,新增)

```python
DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"   # 与 transcript_store 同一默认

def root() -> Path                                  # 认 JINGXIN_RECORDINGS_DIR
def enabled() -> bool                               # 认 JINGXIN_RETAIN_MEDIA,默认 True

def preflight(session_id: str) -> None              # 见下「调用时机」;不可写 → 抛(不是 warning)
def retain_frame(session_id: str, modality: str,
                 data: bytes, seq: int,
                 declared_ts: int | None) -> dict   # 返回写下的那一行,形状见 §5.2
def retain_audio(session_id: str, kind: str,       # kind: "raw" | "converted"
                 data: bytes, seq: int) -> dict
def retain_uploaded_video(session_id: str, data: bytes) -> dict
def degraded_reasons(session_id: str) -> list[str]  # 收尾时读;空 = 无降级
```

- `modality ∈ {"face", "gesture"}`;文件按 `seq` 零填充命名(`000001.jpg`),**字典序即时间序**。
- **`kind="raw"` 保留到达时的容器原样**(`/asr` 与 `answer_audio` 的原始上传分别是 webm 与 wav),
  扩展名由到达容器决定并记进 `retention.jsonl`;`kind="converted"` 固定是 `/asr` 转出的 `.wav`。
- **调用时机**:`preflight` 在**每个会话的第一帧**(会话首次创建、时钟建立的那一处)调用一次;
  同一会话后续帧不再重复预检。
- 写盘用**同一把逐会话锁**(语义同 `transcript_store._session_lock`),避免多帧并发交错。
- `session_id` 一律过 `validate_session_id`;**非法直接 400**,不拿去当目录名。

### 5.2 `retention.jsonl`(每件素材一行,JSON Lines)

```json
{"kind":"frame","modality":"face","seq":1,"file":"media/face/000001.jpg",
 "bytes":84213,"sha256":"…","received_at_wall":1.7588e9,"declared_ts":0,
 "source_endpoint":"/analyze","session_id":"20260925_124346_6e4a"}
```

| 字段 | 为什么必须有 |
|---|---|
| `sha256` | **证明落盘的 == 收到的**。这是「原样」唯一可证的形式 |
| `bytes` | 与文件大小对账 |
| `received_at_wall` | 粗对齐(与 `declared_ts` 一起给出「管线所见的时间」与「墙钟时间」两条线) |
| `declared_ts` | **M2.5 的会话相对时钟值**。重抽时必须用它,不能用墙钟 |
| `source_endpoint` | 同一份目录可能混入多路来源,得知道它从哪个端点来 |

### 5.3 新端点 `POST /session/{session_id}/media`(voice :8001)

- 为什么放语音服务:铸号(`/interview/start`、`/research/start`)、`session.json`、
  `transcript.json` 都归它,`~/shared` 的写入也只有它做过。**不新起服务**。
- 收 `UploadFile`,原样写 `media/camera.webm`,记一行 `retention.jsonl`(`kind:"video"`)。
- 非法 `session_id` → 400(走同一守卫)。

### 5.4 前端改动(三个文件)

| 文件 | 改什么 |
|---|---|
| `src/hooks/useCamera.ts` | ① `getUserMedia` 的 `audio: false` → `true`(R5);② 在同一个 `mediaStream` 上挂 `MediaRecorder`,`stopCapture` 时把 blob 交给调用方 |
| `src/services/api.ts` | 加一个上传函数(把 blob POST 到 `/session/{sid}/media`) |
| (可能)`src/hooks/useAudioRecorder.ts` | 仅在 §8.2 的实测显示两路音频打架时才动 |

**既有模式可抄**:`src/hooks/useAudioRecorder.ts:18-19` 已经在用
`new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' })` 并发 blob。

---

## 6. 错误处理(写死,不留含糊)

| 时机 | 行为 | 理由 |
|---|---|---|
| **会话开始(首帧)** | `preflight` 不可写 → **大声失败**(抛/500) | 半路才发现等于**已经丢了一半素材**,而这一场是人重跑不回来的 |
| **中途某件写失败** | **不中断分析**,记 `retention_degraded` + 原因;会话收尾报出来 | 中断也换不回已丢的字节,还白耗人的时间;但**绝不允许静默** |
| **留存关闭** | 只有 `JINGXIN_RETAIN_MEDIA=0` 才关;**默认开** | 「忘了开」正是丢素材的那条路径 |
| **落盘位置** | **只写 `~/shared`**(→ `/mnt/d`);不写仓库、不写 WSL ext4.vhdx | §3.6;§4.7 |
| **路径穿越** | `validate_session_id` 拦下 → 400 | 与既有守卫同一口径 |

**不与 D2(`log_prosody` 写失败改为抛出)冲突**:D2 抛是因为那条日志是**指标的载体**(丢了就出不了值);
留存丢的是**旁路证据**,分析值仍然成立。所以这里「预检严、中途宽但留痕」,不是双标。

---

## 7. 测试与验收

### 7.1 ★ 重抽等价性(核心判据,也是 M2.6 存在的理由)

**做法**:取一场留存下来的会话,用 `retention.jsonl` 里的 `declared_ts` **回放**留存帧,走
`VideoPipeline.process_frame(frame, timestamp_ms)`(M2.5 已把时间戳改成显式入参),
与**当场活跑**写出的 `data/logs/face_au_log_{sid}.csv` **逐格比对**。

- **必须用 `declared_ts`,不能用墙钟** —— 服务的 `SessionClock` 是按会话生成的,回放时要喂**当时那个值**。
- **反向复现**:把「原样字节」改成重编码一次 JPEG → 该测试**必红**。红了才证明它钉住了东西。
- 覆盖范围:**face 与语音**。手势侧的重抽回放**不在本轮**(离线 `GestureExtractor` 仍含
  `mp.solutions`,M2.5 已裁定延后)。

### 7.2 sha256 对账

`retention.jsonl` 里每行的 `sha256` vs 盘上文件重算的哈希,**逐行相等**;`bytes` vs
`os.path.getsize` 相等。撤掉记账 → 红。

### 7.3 预检真的会拦

把 `JINGXIN_RECORDINGS_DIR` 指向不可写路径 → **首帧必须抛异常**,而不是 warning。
(复用 §3.7 那个环境变量,不新造开关。)

### 7.4 中途失败留痕

注入一次写失败(monkeypatch 写函数抛一次)→ `degraded_reasons(sid)` 非空、`retention.jsonl`
出现 `retention_degraded`、且**分析结果仍然产出**(不因留存失败而丢掉这一场)。

### 7.5 路径穿越

`sid='../x'` / `sid='a/b'` → 400;盘上**不得出现**越界目录。(照既有的 `validate_session_id` 钉子。)

### 7.6 端到端(与录制的第一次真会话合并验)

一场真会话后,逐条打勾(录制需求 §3.4 的原话):

- [ ] `media/face/` 的帧数 == 服务端实际收到的帧数(以 `retention.jsonl` 的行数为准)
- [ ] 素材**文件数与模态数一致**(face / gesture / audio / camera 各有)
- [ ] `camera.webm` **能被解出音轨与画轨**(R5 的直接验收)
- [ ] `retention.jsonl` 无 `degraded`
- [ ] **原始媒体不在 git 里**(`git status` 干净 / 无未跟踪媒体)

### 7.7 不回归

`pytest -q` 全绿;合并门 `--verify-legacy` **0 / 2835510**(按 M2.5 账本的口径:**它只证明
「没有误伤」**,不是本设计的证据 —— §3.1)。

---

## 8. 已知未满足项与风险

### 8.1 「原始」是有限的:采集端已经在转码、已经在降帧

服务端那一腿留的是 **JPEG q0.8、AssessmentPage 下 5 fps**。它足以支撑
「au4/au9/au23/au26 是否仍饱和」「手部列是否仍 87% 为 0」这类**分布核对**;
**不足以**支撑微表情族(微表情 ~1/25 s 起)。后者只有前端那一腿(`camera.webm`,原生帧率)
将来重抽时才有得谈。**这条要写进 M3 的已知代价,不能因为「留了原始媒体」就默认全都测得了。**

### 8.2 R5 的前提未验:两路 `getUserMedia` 音频流会不会打架

`useCamera` 取摄像头、`useAudioRecorder` 另取麦克风。改成 `audio: true` 后**同时存在两路音频流**。
浏览器通常允许,但**必须实测**(真机、Windows 端浏览器):能否同时开、`camera.webm` 的音轨是否正常、
录音按钮是否仍工作。**打架就退回「另存 `audio.webm`」**(两路不同步,只能靠 `received_at_wall` 粗对齐)。
⚠️ 按既定结论,摄像头相关的验收**必须用 Windows 端浏览器**——WSL 端浏览器的摄像头不可用。

### 8.3 前端改动的提交纪律(§4.9)

三个文件已用 **`git hash-object` vs `git rev-parse HEAD:<file>`** 核过
(`useCamera.ts`、`useAudioRecorder.ts` **与 HEAD 逐字节相同**;`api.ts` 改前再核一次)。
⚠️ 该仓 `git status` **会假阳性**(§5 已记:报 41 个改动而实际只有 14 个有内容差异)。
**提交前必须先点名「哪些文件里装着使用者的未提交工作」并请使用者裁定**,不得 `git add -A`。

### 8.4 磁盘增长

一场 5 fps × 60 s = 300 帧 × ~84 KB ≈ **25 MB**,加原生视频 ~10 MB。127 GB 可用,短期无虞。
但**没有轮转机制**(与跟进项 11 的 `NONE` 桶同属「留存/轮转」问题),本轮不做。

### 8.5 `refresh_manifest` 终于有了生产调用方(顺带)

跟进项 3 记「`refresh_manifest` 仍**零生产调用方**」。M2.6 的会话收尾核对
(素材齐不齐、有无 `degraded`)正是它的自然调用方 —— 但要**先确认它的现有形状**
(`:202-207` 查的是 `expected_file` 是否存在)能不能承接素材核对;接不上就另写,不硬塞。

---

## 9. 与后续里程碑的关系

```
M2.6  原始媒体留存(本设计)           ← 立刻做;解锁录制(长周期、要使用者出镜)
  └ 交付即通知使用者开录 3–5 场
M3.1  仪器标定  跟进项 15(VIDEO 等价性门,新时间基)+ 16 的位移重测
  └ 排面部重构**之前**:它定的是「面部数值的仪器误差」,没定下来就是站在流沙上
M3.2/3/4  面部 / 手势 / 语音 逐列重构   ← 各一份 spec+plan;录制素材到位后跑真数据分布核对
M3.5  收尾  报告层 QUARANTINE 重对齐(R3)+ 新阈值/量程登记
```

**M2.6 不被 M3 的设计阻塞**:留存只需保留 `{像素, 音频, 时间戳}`,而 M3 的 54 列无论怎么定
都从这三样导出。所以 M2.6 可以今天就设计完、执行掉,不用等 54 列论证清楚。

**M2.6 也不验证 M3** —— 它只保证 M3 将来**来得及**被验证。这个区别写在标题里,别混。

---

## 10. 待办(由本设计引出,不属本轮)

1. **跟进项 15/16 的做法要重定**:实测本机**只有 mediapipe 1.0.0**(`mp.solutions` 已删),
   没有 0.10 环境 ⟹ 位移表的旧臂**全机跑不起来**。要重测得先另建 mediapipe 0.10 环境
   (**装包给命令、由使用者执行**),否则只能换口径重登记。
2. **16 的许可边界要先说清**:设计文档 §10.7 明文「取自数据集人群分布的参数不得进上线路径」。
   那 28 帧基线的出处需先查清;**可辩护的读法**是「模型包位移是**仪器常数**、不是人群参数」,
   但必须在 spec 里写明,不能默认。
3. **`QUARANTINE` 名单已陈旧**:里面两条写着「M2 修序列化」(等 M2),而 M2/M2.5 早已做完,
   且 spec §4.1 要**删**这两列。M3.4 要对齐时一并处置。
4. **真实会话仍只有 1–5 帧**(§3.3):录制那一轮才第一次拿到多帧真会话。
