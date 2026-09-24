# M1 设计:语音模块接入 FunASR + `session_id` 链路贯通

- 日期:2026-09-24
- 状态:**待使用者审阅**(审阅通过后才写实施计划)
- 上游文件:`docs/下一步.md`(M1 的任务定义)、`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`(§4.1 L0-T 落盘契约、§4.4 隐私分级)、`docs/superpowers/specs/2026-09-22-jingxin-report-layer-evidence-gate-design.md`(报告层的证据门,本设计消费它)

---

## 0. 目标(一句话)

**让报告重新有真话可说**:把语音这条线接通 —— 换掉 vosk、给三份日志装上 `session_id` 让它们能对上号、把转写按契约落盘、并让 `logic_keyword_density` 从假简历常量变成真测量(改名「连接词密度」)。

**成功的定义**:一场真会话跑完,三份日志带**同一个 `session_id`**,报告里「连接词密度」槽**真的过门出分**(不再是证据缺口)。

## 1. 范围

**在范围内**

1. vosk → FunASR(语音模块的 ASR 引擎替换)
2. `session_id` 贯通 voice / face / gesture 三个模块与日志
3. 按总设计文档 §4.1 的**基础字段**落盘转写(仓库外)
4. `logic_keyword_density` → 真测量 + 改名「连接词密度」
5. 三个模块按新契约写日志(首列 + 文件名 + 会话清单)
6. 报告侧同步改名(特征键 / 映射规则 / 白名单)

**不在范围内**(明确划出)

| 不做 | 归属 |
|---|---|
| L0-T 的**派生列**(语速、停顿分布、自我打断、修正/重复、内容相关性…) | M3 的 L0 逐列重构 |
| 时长归一化、特征白名单重做 | M3 |
| **编排层**(统一启动三路采集)、根 `app.py` 与前端改动 | 不起因于 M1;真需要时另立 |
| 报告呈现形态 | ① 已定,本设计不改 |
| 自评量表采集、标定 | M5(卡许可 + 伦理审查) |

---

## 2. 决策记录

每条含依据与"若判断错的代价"。

| # | 决策 | 依据 | 判断错的代价 |
|---|---|---|---|
| D1 | **只做"接管道 + 落盘 + 对上号"**,派生列归 M3 | M1 的目的是让报告有真话可说;派生列的设计要先过时长污染那道关 | 若你其实想一步到位,得重排 M1/M3 边界 |
| D2 | 转写原句**落仓库外** `~/shared/jingxin_recordings/{session_id}/transcript.json`;仓库内只留数值 | §4.4 理由 3:文本是**内容**,保存策略应与声学数值分开;仓库是最不设防处 | 仓库外多一处敏感数据的管理责任 |
| D3 | 留存期**与原始媒体同寿命,按显式动作删除**(项目结束 / 参与者要求 / 一次明确清理),不做自动过期 | 同前提②的推理:特征会重抽,原句也要留着能重算;避免 cron 悄悄删掉将来要用的东西 | 需要人记得清理一次 |
| D4 | `连接词密度 = 连接词数 ÷ 字数 × 100`(**每百字**) | 文档原称「连接词**密度**」是文本比值不是速率;不把**时长**引回文本层指标(时长污染是本项目栽过的那条) | 不反映语速 —— 那是 M3 的时长类列 |
| D5 | 标记表进**数据文件** `connective_markers.json`(带 `version`),不写代码里的裸常量 | 与项目既有规矩一致(阈值/常量必须可登记、可追溯) | 多一个文件;改表要 bump 版本 |
| D6 | `session_id` 由 **voice 的 `/interview/start` 生成并返回**,调用方显式下传三处 | 会话天然起于面试开始;不新增编排层(YAGNI);face/gesture **已有** `/session/{id}/...` 端点正好接上 | 调用方要负责传三处 |
| D7 | 无 `session_id` 的请求:**不拒绝,写 `session_id=NONE` + warning**,报告侧整体排除 `NONE` | 三个模块各有独立示例脚本,硬拒绝会砸掉它们;而"显式 NONE"仍保留可诊断性(不静默) | `NONE` 会话混进数据盘;靠报告侧排除兜住 |
| D8 | **连接词密度在采集时算**(voice 模块有文本),把数字写进语音日志列 | 与现有模式一致(特征都在日志里);**原句因此不进报告路径** —— 报告层永远读不到面试内容 | 改标记表不能重算历史会话(但原句留着,可另跑重算) |
| D9 | ASR 客户端 **vendor 进仓库**(取 WSL 那份**已含 `_merge_finals` 修复**的),pipeline 对外接口不变 | 服务端与共享目录那两份都不受版本控制,且**服务端那份没修**;保持接口则调用方与现有测试不 churn | 仓库多约 400 行外来代码 |

---

## 3. 实测事实(2026-09-24 探测,是设计的地基)

| 项 | 实测值 |
|---|---|
| 服务可达性 | `192.168.72.30:10095` 直连,HTTP **426**(正确响应),建连 **1.4 ms**,ICMP 0% 丢包 |
| systemd | user 服务 `funasr-streaming` = **active**(2026-09-21 23:22 起),监听 `0.0.0.0:10095` |
| 端到端 | 5.5 s 音频 → 文本正确,**1.75 s**(3.1× 实时),VAD **1 段** |
| 逐字时间戳 | **可用,但在 `raw['timestamp']` 里**,`ASRResult` 属性上没有(19 项,形如 `[[210,450],[450,690],…]`) |
| 标点 | `raw['punc_array']` 19 项;`text` 本身已带标点 |
| **ASR 置信度** | **不存在** —— raw 键为 `is_final / mode / punc_array / spk_name / spk_score / text / timestamp / wav_name`,客户端亦未解析置信度(`spk_score` 是**声纹验证**分) |
| 客户端状态 | `~/asr-test/funasr_client.py` 含 `_merge_finals`(长音频多段拼接);jingxin 环境有 websockets 17.0.1 |

---

## 4. 架构

**会话由面试流程发起,`session_id` 显式下传;三个模块保持独立服务,互不调用。**

```
调用方(操作者脚本 / 前端)
   │ ① POST /interview/start
   ▼
voice 服务 ── ② 生成 session_id;建 {RECORDINGS_DIR}/{session_id}/session.json
   │            返回 {"session_id": "...", "question": "..."}
   │ ③ 后续每个请求都带 id:
   ├─ voice   /interview/answer_audio?session_id=…  → ASR → 落转写 → 写日志(含连接词密度)
   ├─ face    /analyze      {…, session_id}          → 写日志
   └─ gesture /analyze      {…, session_id}          → 写日志
```

## 5. 组件

| 文件 | 动作 | 职责(一句话) |
|---|---|---|
| `voice_interaction/asr/funasr_client.py` | vendor | FunASR websocket 客户端(含 `_merge_finals`) |
| `voice_interaction/asr/funasr_engine.py` | 新建 | 薄适配器:`pcm/wav → ASRResult`;**四个已知坑在此收口**(多段拼接 / 增量累加 / `stop()` 不挂死 / 异常不吞) |
| `voice_interaction/asr/connective_density.py` | 新建 | 连接词密度计算(纯函数,输入文本 + 标记表) |
| `voice_interaction/asr/connective_markers.json` | 新建 | 标记表 + `version` + `basis` |
| `voice_interaction/asr/asr_config.json` | 新建 | 采集侧参数(置信度替代判据、最短长度),带 `_provisional` + 依据 |
| `voice_interaction/api/app.py` | 改 | `/interview/start` 发号;`answer_audio` / `/asr` 收 id;转写落盘 |
| `voice_interaction/utils/logger.py` | 改 | `session_id` 首列 + 文件名带 id |
| `face_expression/api/app.py`、`gesture_analysis/api/app.py` | 改 | `/analyze` 接受 `session_id` |
| `face_expression/utils/logger.py`、`gesture_analysis/utils/logger.py` | 改 | 同 logger |
| `report_frontend/feature_engine.py` | 改 | 新列名映射;`_std` / `_n_rows` 成对写入 |
| `report_frontend/research_mapper.py`、`evidence_gate.py`、`evidence_thresholds.json` | 改 | 槽位改名「连接词密度」 |
| `experiments/duration_audit/whitelist_C.json`、`whitelist_Cplus.json` | 改 | 新列名进白名单(否则被静默丢弃) |

## 6. 接口契约

### 6.1 `session_id`

格式 `YYYYMMDD_HHMMSS_<4 位十六进制>`,例:`20260924_153012_9f3c`。时间部分给人看,随机后缀防撞。

### 6.2 端点变更

| 端点 | 变更 |
|---|---|
| `POST /interview/start`(voice) | 响应**新增** `session_id`;副作用:创建 `{RECORDINGS_DIR}/{session_id}/session.json` |
| `POST /interview/answer_audio`(voice) | 接受 `session_id`(query 或表单) |
| `POST /asr`(voice) | 同样接受 `session_id` |
| `POST /analyze`(face / gesture) | body 可含 `session_id` |

缺失时行为见 D7(`NONE` + warning,不拒绝)。

### 6.3 日志

- **首列** `session_id`
- **文件名** `{prefix}_{session_id}.csv`(一个会话一个文件,追加写);`NONE` 时为 `{prefix}_NONE_{YYYYmmdd_HHMMSS}.csv`(保留时间戳以便互相区分)
- 语音日志**一次 `answer_audio` 调用写一行**(VAD 多段在引擎内已拼接成一次结果);新增三列:`connective_density`、`connective_density_std`、`_n_rows`

### 6.4 `transcript.json`(仓库外)

```json
{
  "session_id": "20260924_153012_9f3c",
  "recorded_at": "2026-09-24T15:30:12+08:00",
  "asr": {
    "engine": "funasr", "endpoint": "ws://192.168.72.30:10095",
    "models": {"asr_online": "…vocab8404-online", "asr_offline": "…2pass-offline",
               "vad": "…fsmn_vad…", "punc": "…punc_ct-transformer…"},
    "asr_confidence": null,
    "asr_confidence_source": "unavailable"
  },
  "merged": {"text": "…", "n_chars": 19, "n_segments": 1, "vad_split": false},
  "segments": [
    {"index": 0, "text": "…", "n_chars": 19,
     "timestamps_ms": [[210, 450], [450, 690], "…"],
     "punc_array": [1, 1, "…"],
     "ts_origin": "segment_relative"}
  ]
}
```

`ts_origin` 恒为 `segment_relative`:服务端 VAD 把长音频切成多段,**每段时间戳都从 0 重计**(详见 §9)。

**一个会话一个文件,累积写**:每段回答把结果**追加**进 `segments`(带 `index`),并重算 `merged`(全段文本拼接、`n_chars` 求和、`vad_split` 只要任一段裂过即为 true)。不按回答覆盖。

### 6.5 连接词密度

- 定义:`连接词数 ÷ 字数 × 100`。字数 = 中文字符数(去标点与空白)
- 长度下限:`n_chars < min_chars_for_density` 时**不出值**(记为缺口,不写 0)
  - 初值 **10**,写在 `asr_config.json`,带 `_provisional: true` 与依据:`低于 10 字时,任意一次命中即产生 >10 的密度,比值不稳定`
- 标记表 `connective_markers.json`:
  ```json
  {"version": "1.0.0",
   "basis": "汉语话语连接标记的封闭清单;M1 首版,可辩护即可,改表须 bump version",
   "markers": ["然后", "所以", "但是", "因为", "而且", "如果", "虽然", "不过",
               "因此", "另外", "其实", "首先", "其次", "最后", "总之", "比如", "例如"]}
  ```
- 计算在**采集时**完成(D8),原句不进仓库、不进报告路径

## 7. 数据流(一段回答)

```
answer_audio(session_id, 音频)
  → funasr_engine 识别(多段则拼接)
  → ① {RECORDINGS_DIR}/{session_id}/transcript.json   ← 仓库外,原句 + 逐字时间戳
  → ② connective_density(文本) → 写语音日志行
        session_id, timestamp, …, connective_density, connective_density_std, _n_rows
  → ③ 报告侧:feature_engine 读日志 → 证据门四关 → 过门则呈现「连接词密度 + 本场区间 + 样本量」
```

## 8. 错误处理(写死,不留含糊)

| 情形 | 行为 |
|---|---|
| ASR 服务不可达 / 超时 | 该段**标为失败并记原因**,不写 0、不写常量 → 报告侧呈现为诚实的「未采集到对应数据」 |
| 文本过于短(< `min_chars_for_density`) | 连接词密度**不出值**(缺口),不写 0 |
| 长音频被 VAD 切多段 | **必须拼接**(`_merge_finals`);时间戳按段接续,并**记录已知偏差**(段间静音未计入 → 第 2 段起偏早) |
| 客户端已知缺陷 | `ASRSession.stop()` 固定卡 ≈15 s、`recv()` 吞异常 → 适配器内显式处理(超时可控、异常上抛) |
| 无 `session_id` | 写 `NONE` + warning(D7),不静默、不拒绝 |

## 9. 已知未满足项与风险(必须带进 M1 的实施与报告)

1. **`asr_confidence` 拿不到**(§3 实测)。总设计文档 §4.1 要求落盘它、并据它给派生特征置 NaN 或降权(规矩 2)。本服务版本不返回置信度,故:
   - `transcript.json` 里**显式记 `null` + `source: unavailable`**;
   - 规矩 2 **降级实现**:只在"明显不可用"时置空(文本为空、字数低于下限、ASR 报错),不按置信度分档;
   - **补救选项**(超出 M1,记为待办):查服务端 FunASR runtime 版本是否支持返回逐字置信度,或升级 runtime / 换模型。
2. **VAD 分段阈值需重测**:记忆里「32 s 内一段、40 s 裂两段」是**别的音频**测的,要用真实面试音频复核并记录。
3. **时间戳是段内相对值**:跨段拼接后第 2 段起偏早(段间静音未计入)。需要绝对时间时要按静音自己切段、逐段单独发送。
4. **单段回答会被 G2 拦下**:`connective_density_std` 在只有一个样本时为 0,证据门 G2(非常量)会把它判为「本次会话内无变化」。**这是设计使然,不是 bug** —— 验收会话需 **≥2 段回答**。
5. **标记表的语言学效度**:M1 首版是"可辩护的封闭清单",不是标定产物。它决定的值会进报告,故清单与依据必须随版本记录。

## 10. 测试与验收

**单测**
- `connective_density`:正常 / 空文本 / 全标记 / 低于长度下限 → 不出值 / 标点与空白不计入字数
- 标记表:加载、`version` 存在、清单非空
- logger:写 `session_id` 首列;`NONE` 分支;文件名含 id

**假引擎测 pipeline**(注入假 ASR 客户端,不打真服务)
- 多段结果**拼接**(回归 `_merge_finals`)
- `2pass-online` 增量**累加**而非替换
- `stop()` 在超时内返回(不卡 15 s)
- 服务端异常**不被吞**

**契约测**
- 三条带同一 `session_id` 的合成日志可 join;`NONE` 会话被报告侧排除

**验收门(方案 A,需你配合录一场)**
1. 录一场会话(≥2 段回答)
2. 三份日志出现**同一个 `session_id`**
3. 报告里「连接词密度」槽**过门出分**,呈现为「连接词密度 + 本场区间 + 有效样本量」
4. 顺带确认:`transcript.json` 在 `{session_id}/` 下、原句不出现在仓库内任何文件里

---

## 11. 待办(不属于 M1,但由本设计引出)

- `asr_confidence` 的补救(§9.1)
- 失败段落的重试策略(现在只记失败)
- 编排层:何时需要统一启动三路采集
- 根 `app.py` / 前端接入 `session_id`(现在不需要,因为调用方是操作者脚本)
