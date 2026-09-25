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
| **阶段 A 元数据层**(R7) | `meta.json` 模板 + 题目时刻记账(见 §5.5)、上报端点(见 §5.6);落点同上 |
| 前端录制 + 题目上报 | `~/JingXin-frontend`:`src/hooks/useCamera.ts`、`src/components/assessment/AssessmentPage.tsx`、`src/services/api.ts` |
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
| **★ 不采自评量表** | 录制需求 §2 前提 3:《科技伦理审查办法(试行)》第 2 条第(一)项,**强制前置**。R7 扩的是**元数据**(题目/设备/人的生理属性/面试官评分),**不含**候选人自评 —— 后者是 M5 标定用的真值 |
| **不把元数据当特征** | `meta.json` 是**协变量与出处记录**,不是 L0 列。M2.6 只负责把它留下来,M3/M4 决定谁进模型 |

---

## 2. 决策记录(2026-09-25,使用者逐条裁定)

**这张表把整个 M3 brainstorming 的九条裁定都记在这里**(末列标了归属),理由:它们出自同一个会话,
而 M3 那本 spec 还没写 —— 先落在一处,总好过散在对话里丢掉。**标 `M3` 的三条将在 M3 的 spec 里
被正式承接**,本设计不因为它们而扩大范围。

| # | 问题 | 裁定 | 归属 |
|---|---|---|---|
| R1 | M3 靠什么自证? | **并行**:先补留存,M3 边做边等录制。理由:留存是**唯一不可逆**的一步,且录制是长周期、要使用者出镜的那条腿 | M3 |
| R2 | 留存存什么保真度? | **两者都要**:服务端存帧(管线所见等价)+ 前端录原生视频(帧率上限不被钉死) | M2.6 |
| R3 | M3 改完列后报告层跟不跟? | **同步对齐**(封停名单重对齐到 M3 处置表)。注:L1 是 M4,M3 的新列在 L1 公式之前本就不出分 | M3 |
| R4 | M3 的内部切分 | 路线 **A**:留存摘成独立的 M2.6 → M3.1 仪器标定(跟进项 15/16)→ M3.2 面部 / M3.3 手势 / M3.4 语音 → M3.5 收尾(报告对齐 + 阈值重登记) | 两个里程碑 |
| R5 | `camera.webm` 带不带声音? | **带**(`useCamera` 的 `getUserMedia` 由 `audio: false` 改 `true`)。前提是实测两路 `getUserMedia` 的音频流不打架(见 §8.2) | M2.6 |
| R6 | 谁算 §10.7 的合法标定源? | **自采数据算**(本设计的 28 帧自采基线 + 3–5 场录制)。注意:数据集线与数据集的衍生物**仍然禁** | M3 |
| R7 | M2.6 只做媒体,还是连阶段 A 元数据一起? | **一起做**。理由:录完再补不回来的不只是像素 —— 该场的面试官结构化评分、每道题的起止时刻、该场的光照/增益都是当时不记就永久没有的 | M2.6 |
| R8 | 跟进项 15 现在验什么? | **换目标**:不再宣称"迁移等价"(旧 `mp.solutions` 臂全机已删、基线又随 M2.5 移动 ⟹ 不可能),改为在**新时间基**上重录两臂、把差值登记为**仪器误差** | M3.1 |
| R9 | 微表情族在 5 fps 下怎么办? | **移出本轮 54 列**。5 fps = 每帧 200 ms,而微表情是 40–200 ms 的 onset–apex–offset 结构 ⟹ 硬算出来就是伪测量。等从 `camera.webm` 原生帧率重抽时再加回;F7 暂不入 L1 | M3 |

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

### 3.8 ★ 那 28 帧基线是**自采**的(故 R6 站得住)

`~/shared/mp_frames/capture_frames.py` 的开头逐字写明:它从**使用者自己的摄像头**抓帧
(`cv2.VideoCapture(CAMERA_INDEX)`),**在 Windows 侧跑**(原文:"Windows 能直接开摄像头,WSL 开不了"),
存 **PNG**(原文:"PNG 而不是 JPG:无损,两个实现读到的是逐字节相同的图"),并附了一份拍摄意图清单
(正脸/微侧/大侧、微笑/皱眉、手挡半张脸、空场景)。

⟹ 这 28 帧**不是** RecruitView / MIT / First Impressions 的切片(数据集用的是 `vid_*.mp4` 命名)。
故 §10.7 的那条线**没有被踩**:位移表来源是自采数据,且它量的是**模型包位移**这个**仪器**量,
不是人群参数。

### 3.9 题目时刻在前端可得,但**没有任何地方上报**

- 前端知道每道题何时推:`src/store/assessmentStore.ts:7`(`currentQuestionIndex`)、
  `src/components/assessment/AssessmentPage.tsx:264-265`(进度 `/10`)、`:279`(`currentQuestion`)。
- 语音服务侧**只有** `POST /tts`(`voice_interaction/api/app.py:168`),**不记**推题时刻。
- ⟹ `response_latency`(L1 的 V2)目前**不可得**,与设计文档 §11.1 的记载一致。
  **而 M2.6 是它唯一来得及的位置** —— 录制一开始,没有 `questions.jsonl` 的场次就永久没有它。

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
               /session/{sid}/question  ← JSON ── 前端推每道题时上报起止时刻(R7,§5.6)

落盘(~shared/jingxin_recordings/{session_id}/,9p → D:\)
  media/
    face/000001.jpg …      原始字节,不转码不缩放不裁剪
    gesture/000001.jpg …
    audio/0001.webm …      到达的原始上传字节
    audio/0001_converted.wav   提取器真正读的 PCM
    camera.webm            前端 MediaRecorder 原生音视频(含音轨,R5)
  retention.jsonl          每件一行,见 §5.2
  questions.jsonl          每道题一行:起止时刻(R7,§5.5)
  meta.json                阶段 A 元数据:人填的那些(R7,§5.5)
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

### 5.5 阶段 A 元数据层(R7)

**为什么必须在这一轮**:媒体留住了,但"这一场是在什么条件下采的"没有 —— 那么 M3 拿到素材
以后,依然分不清一个数值差异是**人的差异**还是**取景/设备的差异**。而下列各项**录完就补不回来**
(它们不是稳定属性,是当时那一刻的状态):

- **该场的面试官结构化评分** —— 审查 Q3(f) 称其为「你现在最该补的真值,成本最低、最贴用途」
- **每道题的起止时刻** —— `response_latency` 只此一途(设计文档 §11.1)
- **该场的光照 / 麦克风增益** —— 审查实测 `energy` 的 ICC = 0.654,是设备增益/距离代理

按**来源**分两种,不混:

| 来源 | 项 | 形态 |
|---|---|---|
| **人填**(会话前后) | 性别/年龄/母语/方言;摄像头型号/分辨率/镜头距离/光照;麦克风增益;题目 ID + 难度;面试官结构化评分;知情同意是否归档 | 一份 `meta.json` 模板,字段照录制需求 §4(**给的是必须覆盖的信息项,不是最终 schema**) |
| **机器记** | 每道题的 `ask_start` / `ask_end`;实际帧率;音频采样率 | `questions.jsonl` + `retention.jsonl`(服务端写) |

**人工项的录入方式**:M2.6 只提供**模板与校验**(字段齐全性),不提供 UI。理由:§3.4 本来就把它
写成一份「录完**当场**逐条打勾、不要事后补」的清单;为它做界面属于把一次性流程产品化,是本末倒置。
**校验要硬**:会话收尾时,`meta.json` 缺必填项 → 报出来(与 §6 的 `degraded` 同一条出口)。

**⚠️ 明确不含**:候选人**自评量表**。那是 M5 标定用的真值,受伦理审查强制前置(§1.2)。

### 5.6 题目时刻上报端点 `POST /session/{session_id}/question`(voice :8001)

- 请求体:`{"qid": "...", "index": 4, "ask_start": 1.7588e9, "ask_end": 1.7588e9}`
  —— 时刻用**墙钟秒**(与 M2.5 的会话内相对时钟**分开**,两者不要混)。
- ⚠️ `response_latency = 首次开口墙钟 − ask_end`,而**ASR 侧的逐字时间戳是段内相对值**
  (已知事实:长音频被 VAD 切多段,段内时间戳从 0 起)。所以"首次开口墙钟"必须
  **由段起始墙钟 + 段内偏移**合成 —— `transcript_store.append_utterance(…, recorded_at=…)`
  已经在记每次回答的墙钟(`session.json`/`transcript.json` 侧),接得上。
  **本条不在 M2.6 实现**(M2.6 只负责把 `ask_end` 留下来),但必须在这里点明,
  免得 M3 真去算时才发现两个基不同。
- 服务端逐行追加 `questions.jsonl`;同 `(session_id, qid)` 重复上报**以后来的为准**(允许多次上报/修正)。
- 前端落点:`AssessmentPage` 推题处(`currentQuestionIndex` / `currentQuestion`)。
  ⚠️ 前端**不判断**哪道题"算数" —— 只如实上报推题与结束的时刻。

---

## 6. 错误处理(写死,不留含糊)

| 时机 | 行为 | 理由 |
|---|---|---|
| **会话开始(首帧)** | `preflight` 不可写 → **大声失败**(抛/500) | 半路才发现等于**已经丢了一半素材**,而这一场是人重跑不回来的 |
| **中途某件写失败** | **不中断分析**,记 `retention_degraded` + 原因;会话收尾报出来 | 中断也换不回已丢的字节,还白耗人的时间;但**绝不允许静默** |
| **留存关闭** | 只有 `JINGXIN_RETAIN_MEDIA=0` 才关;**默认开** | 「忘了开」正是丢素材的那条路径 |
| **落盘位置** | **只写 `~/shared`**(→ `/mnt/d`);不写仓库、不写 WSL ext4.vhdx | §3.6;§4.7 |
| **路径穿越** | `validate_session_id` 拦下 → 400 | 与既有守卫同一口径 |
| **`meta.json` 缺必填项**(R7) | 会话收尾报出来(走 `degraded` 同一条出口),**不阻断分析** | 元数据缺项不会让这一场的特征失效,但**必须让人当场知道** —— §3.4 本来就是"当场打勾"的清单 |
| **题目时刻漏报**(R7) | 收尾时按 `questions.jsonl` 与已上报的回答数对账,缺哪题报哪题 | `response_latency` 缺一道就是缺一道,不能静默少一题 |

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

### 7.6 元数据层(R7)

1. **题目时刻能算通**:造一场 `questions.jsonl`(两题)+ 一份带逐字时间戳的 transcript
   → `response_latency` 算得出,且**用的是 `ask_end` 自己的值**(改为错值 → 结果跟着变)。
2. **重复上报以后来为准**:同一 `(sid, qid)` 报两次不同 `ask_end` → 落盘的只有后一个。
3. **`meta.json` 缺项的校验会响**:删掉 `interviewer_ratings` → 收尾报出来;**且分析结果仍产出**。
4. **漏报对账**:两题只报一题 → 收尾点名缺的那题。

### 7.7 端到端(与录制的第一次真会话合并验)

一场真会话后,逐条打勾(录制需求 §3.4 的原话):

- [ ] `media/face/` 的帧数 == 服务端实际收到的帧数(以 `retention.jsonl` 的行数为准)
- [ ] 素材**文件数与模态数一致**(face / gesture / audio / camera 各有)
- [ ] `camera.webm` **能被解出音轨与画轨**(R5 的直接验收)
- [ ] `retention.jsonl` 无 `degraded`
- [ ] **原始媒体不在 git 里**(`git status` 干净 / 无未跟踪媒体)
- [ ] `questions.jsonl` 的**题数 == 本场实问题数**(R7)
- [ ] `meta.json` **必填项齐全**(R7;录制需求 §3.2 那张表)
- [ ] **未采自评量表**(R7 的边界,录制需求 前提 3)

### 7.8 不回归

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

### 8.6 元数据层的**人工项**可能被跳过(这是它最大的风险)

`meta.json` 里的面试官评分、题目难度、设备参数**只能人填**。技术手段拦不住"录完忘了填"，
只能在会话收尾时**对账并报出来**(§6)。这与 §3.4「录完当场逐条打勾,不要事后补」是同一个意思：
**靠流程,不靠代码**。M2.6 能做的上限就是"让它响"。
**缓解**:3–5 场的规模下，录制前后各跑一次 `--check-session` 之类的对账命令即可，不需要产品化。

---

## 9. 与后续里程碑的关系

```
M2.6  原始媒体留存 + 阶段 A 元数据层(本设计,R7)   ← 立刻做;解锁录制(长周期、要使用者出镜)
  └ 交付即通知使用者开录 3–5 场
M3.1  仪器标定  R8:在**新时间基**上重录两臂、把模式差异登记为**仪器误差**
  └ 不再宣称"与旧 solutions 迁移等价"(旧臂全机已删 ⟹ 不可能);+ 16 的阈值/量程重登记
  └ 排面部重构**之前**:它定的是「面部数值的仪器误差」,没定下来就是站在流沙上
M3.2/3/4  面部 / 手势 / 语音 逐列重构   ← 各一份 spec+plan;录制素材到位后跑真数据分布核对
M3.5  收尾  报告层 QUARANTINE 重对齐(R3)+ 新阈值/量程登记
```

**M2.6 不被 M3 的设计阻塞**:留存只需保留 `{像素, 音频, 时间戳}`,而 M3 的 54 列无论怎么定
都从这三样导出。所以 M2.6 可以今天就设计完、执行掉,不用等 54 列论证清楚。

**M2.6 也不验证 M3** —— 它只保证 M3 将来**来得及**被验证。这个区别写在标题里,别混。

---

## 10. 待办(由本设计引出,不属本轮)

1. **`au15_mouth_down` 的残差回归做不出来**(R6 的直接推论)。R6 裁定"自采数据算合法标定源",
   但 au15 的处置(§4.1)要**用 yaw/pitch/roll 回归预测并输出残差** —— 而可用的自采数据是
   28 张静止帧 + 3–5 场录制(**5 个人**),样本量根本不够拟合一个头姿回归。
   ⟹ 这条列在 M3 只能**降级**(出原始几何量 + 显式标注"含头姿分量"),不能按原处置执行。
   **写出来才算诚实,不能因为"来源合法了"就默认做得了。**
2. **16 的登记口径要写明样本量**(R6)。R6 让 au26 的 p1/p99 映射可以做,但它的"常模"是
   **5 个人**。`evidence_thresholds.json` 的 `basis_kind` 必须如实写成"自采 n≈3–5",
   **不得**写成 population/norm。**5 个人的分布不是常模,写成常模就是自欺。**
3. **`QUARANTINE` 名单已陈旧**:里面两条写着「M2 修序列化」(等 M2),而 M2/M2.5 早已做完,
   且 spec §4.1 要**删**这两列。`M3.5` 对齐时一并处置。
4. **真实会话仍只有 1–5 帧**(§3.3):录制那一轮才第一次拿到多帧真会话。
5. **微表情族移出 54 列(R9)**:`micro_exp_*` 整族不进本轮 L0 契约,F7 暂不入 L1;
   等从 `camera.webm` 原生帧率重抽时再加回。**54 → 约 52 列。**
