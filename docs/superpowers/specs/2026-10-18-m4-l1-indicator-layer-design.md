# M4 设计:L1 指标层(会话级指标 + 元数据结构 + `evidence_gaps`)

- 日期:2026-10-18
- 状态:**M4.0–M4.17 全落;报告端点已由 L1 接管(`?engine=legacy` 可回退);正式语料 13 场跑通(34 条/场,共 38 条)**;剩余 4 条各有硬阻塞(见 `docs/下一步.md` 丙-33)。**逐题窗口(M4.17)本身待裁定** —— 链路完整,但 24 场正式场无台账,粒度方案与"是否离线标边界"见丙-38 与 `2026-10-19-m4-per-question-window-capture-protocol-design.md`(v2)
- 增补对象:`2026-09-21-jingxin-feature-redesign-design.md` §5(L1 会话级指标)
- **范围:** 只读侧新增一个 L1 层。**`report_frontend/` 现有四个模块一行不改**,采集层一行不改。
- 非范围:L2 特质层 / 标定(M5)、学习式多模态融合(见 §8)、26 场自然对话的粒度(戊-15,待裁定)

---

## 0. 摘要

本文件把设计文档 §5 那张 **38** 个指标的清单,变成**可以逐条验收**的四件东西:

| 编号 | 交付 | 为什么它必须排在这里 |
|---|---|---|
| **M4.0** | 统一会话时间轴 | 三个模态现在**时间基准互不相同**(§2)。不先统一,任何跨模态指标都是假的 |
| **M4.1** | 事件窗口(answer windows) | 指标的分母、分子都依赖"一行是什么"。粒度定不死,G3 的阈值就没有意义 |
| **M4.2** | 指标元数据结构 + 四档 `confidence` | 设计文档 §5.6 的核心接口。**先有形状,再有数** |
| **M4.3** | F 族 16 个指标 | 唯一**不需要跨模态对齐**的一族 ⟹ 可以不等 M4.0 就验收 |

**落地情况(2026-10-18):**

| 片 | 文件 | 测试 |
|---|---|---|
| M4.0 | `report_frontend/session_timeline.py` | `tests/test_m4_session_timeline.py`(6 条,3 条红法实测) |
| M4.1 | `report_frontend/event_windows.py` | `tests/test_m4_event_windows.py`(6 条,3 条红法实测) |
| M4.2 | `report_frontend/l1_indicators.py` | `tests/test_m4_l1_indicators.py`(12 条,4 条红法实测) |
| M4.3 | `report_frontend/l1_face.py` | 同上 |
| 总入口 | `report_frontend/l1_session.py` | —— |
| M4.4 | `report_frontend/l1_quality.py` | `tests/test_m4_quality.py`(8 条,2 条红法实测) |
| M4.6 墙钟修复 | `report_frontend/clock_repair.py` | `tests/test_m4_clock_repair.py`(6 条,4 条红法实测) |
| M4.7 G 族 4 指标 + 3 协变量 | `report_frontend/l1_gesture.py` | `tests/test_m4_l1_voice_gesture.py`(9 条,3 条红法实测) |
| M4.8 V 族 12 指标 | `report_frontend/l1_voice.py` | 同上 |
| M4.9 语音序列重建 | `report_frontend/voice_series.py`、`l1_voice_series.py` | `tests/test_m4_voice_series.py`(10 条,4 条红法实测) |
| M4.10 模态实际覆盖率 | `report_frontend/l1_quality.modality_coverage()` | `tests/test_m4_quality.py`(+5 条,2 条红法实测) |
| M4.11 有效素材分级 | `report_frontend/corpus.py` | `tests/test_m4_corpus_tiers.py`(13 条,3 条红法实测) |
| M4.12 正式语料打通 | `session_timeline.session_origin_wall(allow_frame_origin=)`、`audio_timebase()` | `tests/test_m4_session_timeline.py`(+1 条) |
| **M4.5 槽位替换** | `report_frontend/l1_mapper.py` | `tests/test_m4_l1_mapper.py`(18 条,5 条红法实测) |
| **M4.13 τ 解析层** | `report_frontend/l1_thresholds.py` | `tests/test_m4_thresholds.py`(11 条,5 条红法实测) |
| **M4.15 转写补 `n_chars` + 序列补停顿率** | `report_frontend/transcript_chars.py`(新)、`l1_voice.py`、`l1_voice_series.py` | `tests/test_m4_transcript_chars.py`(10 条,8 条红法实测)、`tests/test_m4_voice_series.py`(+3 条,3 条红法实测) |
| **M4.14 端点接管** | `report_frontend/l1_report.py`、`app.py`(`?engine=l1\|legacy`)、前端 `assessment.ts` / `ReportViewer.tsx` | `tests/test_structured_report_endpoint.py`(+4 条) |
| **M4.16 三条裁定落地**(T 族 3 条 + 手势 2 条 + AU 换构念) | `report_frontend/l1_text.py`(新)、`self_repair_markers.json`(新)、`l1_gesture.py`、`l1_face.py`、`l1_thresholds.py`(`self_tau`)、`l1_session.py`、`l1_mapper.py`、前端 `L1IndicatorTable.tsx`(新) | `tests/test_m4_text.py`(17 条,13 条红法实测)、`test_m4_l1_voice_gesture.py`(+6 条,6 条红法实测)、`test_m4_l1_indicators.py`(+3 条,4 条红法实测)、`test_m4_l1_mapper.py`(+2 条,3 条红法实测) |
| **M4.17 逐题窗口的三个缺口** | `report_frontend/event_windows.py`(`count_outside`、段数核对按模式区分)、`l1_session.py`(`frames_outside_note`)、`l1_quality.py`(载荷 `frames_outside`)、`session_timeline.py`(`Timebase.span_sec`) | `tests/test_m4_per_question_windows.py`(6 条,4 条红法实测)、`test_m4_session_timeline.py`(+1 条,1 条红法实测) |

新增测试合计 **164 条**(实测 `pytest --collect-only -k m4` 计数,不是估;含审计后补的 8 条、
拼接场查证补的 7 条、语料审计补的 1 条、M4.17 的 7 条),全部**不依赖 `data/`**。
这四批新守卫**全部**做过红法实测:审计后 8 条(10 次变异,9 次变红;剩下 1 次是
行为等价的无效变异,见丙-36)、拼接场 7 条(5 次变异全红)、语料审计 1 条(2 次变异全红)、
M4.17 的 7 条(**6 次代码变异,6 次全红**:计数恒 0、去掉会话级早退、删"无逐题音频"分支、
删"对不上"守卫、阈值默认改 0、删 `Timebase.__post_init__`)。⚠️ 第 7 条(端到端那条)
的红法是**测试侧改动**(把台账写到别的目录),已在 docstring 写明但**没有实跑**,如实记在这里。
更早各条的红法写在各自的 docstring 里,不在此汇总一个我没逐条数过的总数。

### M4.16 落地结论(2026-10-18)

指标总数 35 → **38**(F 16 / G 7 / V 12 / T 3)。13 场 A 档实测:**34 条/场**(候选H 33 条,
其 `smile_duration_p50` 因全场无笑段而如实不可出)。剩 4 条不可出:`micro_exp_*` ×2
(上游 0 非空)、`pause_ratio`(与 `1/voiced_ratio−1` r=0.9990,故意不产出)、
`response_latency_sec`(缺 `ask_end`,戊-15)。

三件事各自的关键实测:

| 改动 | 关键证据 |
|---|---|
| T 族 3 条 | 原始 TTR 长度依赖(320 字 0.4125 / 3899 字 0.1411)⟹ 改 MATTR;`mattr100_bi` 0.805~0.880 无长度趋势 |
| 手势 2 条 | 手势日志全场 **1.000 Hz**(p10=p90=1000 ms)⟹ 最短段长按秒定义;`burst` 与 `gesture_activity` r=−0.07(不是同一个量) |
| 手势 2 条(审计修正后) | 对抗审计发现 G2 的局部变量把 G1 的结果覆盖了,"运动段"实际量的是姿态信号 ⟹ 9/13 场段数错。修正后 `burst` vs `shift` r 由 0.769 降到 **0.586**、相同值由 9/13 降到 **4/13**;两族的置信改为按**段数**判,13 场里 12 场如实落到「低」、1 场「中」 |
| T 族修正 | `immediate_bigram_repeats` 在周期串上按 2k−3 计数 ⟹ 改成数**接缝**;标记表删掉 `不对`(5 次命中全在「对不对」反问尾)与 `应该是`(9 次全是对冲词),版本 1.0.0 → 1.1.0 |
| AU 换构念 | 联合 AND 三场 **0.000000**;「至少一个」0.13~0.55 ⟹ 有区分度 |

### 一场由 9 段录制拼成:同一场里不同族的观察窗口不同(2026-10-18)

13 场 A 档里 **只有候选B一场**是多段采集。证据链:

1. `retention.camera.jsonl` **9 条**记录,同 `file`、同 `seq=1`、**9 个不同 sha256** ⟹ 整文件重传覆盖;
2. 9 次上传都在某段结束后 26~87 秒;
3. 每份字节数 ÷ 该段时长恒 **~315 KB/s** ⟹ 上传的就是该段完整录像;
4. 盘上 `camera.webm` 实测 **434.7 秒**(视频 13029 包 / 音频 7245 包),只有**最后一段**;
5. 帧时间轴连续(每 10 分钟 ~550 帧),8 处 26~105 秒的重启空档共 468 秒 ⟹ 不是重复帧。

⟹ 访谈真实时长 **6959 秒(116 分钟)**,转写只覆盖**最后 7.2 分钟**。
**F/G 两族量 116 分钟,V/T 两族量 7.2 分钟**,两组各自算得对但不能互相印证。

处置:`capture_segments()` 切段 + `segment_gap_note()` 出缺口说明;
`corpus.has_whole_audio` 从"文件存在"改成"存在 **且** 上传记录 ≤1"。
未解决:9 段是"一场被切 9 段"还是"9 次尝试",数据判不出来,需问现场。

### 正式语料怎么打通的:两处"前提错了",不是数据不够

**症状**:13 场 A 档正式场,面部覆盖 92–100%、手势 94–100%,L1 却**全部 0 条有值**。

**病因一:原点只能来自"第一次提问"。** `session_origin_wall` 取
`min(ask_start)`,而**正式场 24/24 场没有 `questions.jsonl`** ⟹ 原点为空 ⟹
`compute_session_l1` 开头就 `return`。可实测 `c9f0` 的
`retention.face.jsonl` **163/163 行都有 `declared_ts`**,时间轴完全建得起来 ——
挡住它的不是数据,是那个假设。

处置:`session_origin_wall(allow_frame_origin=)` —— **默认 False**,
戊-15「没有台账就不建轴」那条裁定原样保留;`compute_session_l1` 显式开启。
回退原点 = 首个留存帧的 `received_at_wall`,`gap` 里写明"会话级粒度,逐题窗口仍待戊-15"。

**病因二:语音日志 0 行被当成"模态缺失"。** 正式场 24/24 场
`interview_emotion_log_<sid>.csv` **一个数据行都没有**。而 V 族那 8 条序列指标
**根本不读日志**,只要帧序列 —— 却被一刀切成 12 条 `modality_missing`。

处置:日志缺失时**只**把 4 条日志侧指标判死,序列侧照算;并补一条
`audio_timebase()` 让时间轴认识"voice 来自原始音频"(否则质量报告的硬不变量
会正确地拦下"`voiced_ratio` 有值但 voice 模态不存在")。

**打通后实测(13 场 A 档)**:

```
每场 35 条指标,18 条有值(15 高 / 3 低),13 场合计 234 条
高置信:gaze_stability / blink_rate_per_min / blink_duration_mean /
        head_motion_per_sec / au_variability_per_sec /
        gesture_activity_per_sec / posture_stability /
        hand_visible_ratio_l|r / pose_valid_ratio /
        voiced_ratio / pause_duration_p50|p90 / pitch_median / energy_median
低置信(单次测量):pitch_range_semitone / pitch_variability_per_sec /
        energy_dynamics_per_sec
```

### M4.5:14 槽的构念对账(2026-10-18)

判据是"量的还是不是同一个东西",**不是**"能不能算出个数"。**7 换 7 删**:

```
换  gaze_direction_x_std → gaze_stability        au4_frown → brow_furrow_ratio_vs_baseline
    au23_lip_compression → lip_compression_ratio_vs_baseline
    left_wrist_jitter_world → gesture_activity_per_sec
    pitch_variation → pitch_variability_per_sec   pause_duration → pause_duration_p50
    reaction_time → response_latency_sec
删  connective_density  无文本指标            symmetry_score    无对称性指标
    fluency_score       无产出方              left_hand_spread  可见性≠张开度
    shrug_level         稳定性≠耸肩            eye_contact       偏离占比≠接触比例
    n_chars             字/秒≠字数
```

`confidence_level` 三维度全删 ⟹ **维度整体消失**(不是"本场没测到",是那个维度
量的东西当前管线都不测)。权重在幸存槽上按原比例重新归一(本仓删槽的既有做法)。
置信度取**最弱证据**(不是"填了几个槽");不给维度分、不给档位。

**实测 13 场 A 档:4 个维度,`communication_fluency` 低、其余无。**

### M4.14:端点接管 —— 以及顺手抓到的两个 bug

`/api/report/structured` 默认 `engine=l1`;`?engine=legacy` 取旧载荷;拼错的引擎名
返回 **400**(不静默兜底 —— 静默会让"我请求的是旧引擎"和"实际拿到新引擎"分不清)。

**同一场(候选T 2e84)对照**:

```
l1      维度 4(+1 消失)  覆盖 7/14  cap=高   L1 有值 26/35
legacy  维度 5            覆盖 8/14  cap=中
```

legacy 说 `行为表现活跃度` 置信「中」,而那三个槽 L1 判定为当前管线都不测 ——
正是要换掉的那句不成立的陈述。

**bug ①:`NaN` 被数成"有值"。** `l1_indicators.unavailable()` 把"算不出来"记作
`value=math.nan`,而 `math.nan is not None` 是 **True**。`l1_mapper` 第一版只判
`is not None` ⟹ 空数据场的 `n_passed` 从 0 虚报成 **7**、`matched_indicators` 跟着虚高。
抓法:端点测试在**完全隔离**(把 `JINGXIN_RECORDINGS_DIR` / `JINGXIN_SERIES_DIR`
都指向空目录)下断言 `n_passed == 0`,当场红。修法:`_clean()` 把 NaN 归一成 None。

**bug ②:两张置信度排序表同名反序。** `l1_indicators.CONFIDENCE = ("高","中","低","无")`
的顺序是"从可信到不可信",直接 `enumerate` 出来是 **高=0**;而
`research_mapper.CONF_ORDER` 是 **高=3**。两个表同名、都用 `max`、含义正好颠倒 ——
第一版就把"最弱证据"算成了"最强证据",被
`test_dimension_confidence_is_the_weakest_matched_evidence` 抓住。
修法:统一为 `CONF_RANK = {无:0, 低:1, 中:2, 高:3}`,**最弱用 `min`、最强用 `max`**,
并由 `test_confidence_rank_matches_research_mapper` 守住两边不许再分叉。

**分母口径**:`coverage.n_slots` 用**声明的全部旧槽数(14)**,不是幸存维度里的 11 ——
消失的维度那 3 个槽存在过,读者要能看出"14 个里 3 个没有对应量被删"。
新增 `n_dropped` 单独报被删数。

### M4.13:τ 落地 —— 9 条里 5 条自比基线(改名)+ 3 条声明常量 + 1 条仍不可出;修掉 2 个方向 bug

2026-10-18 按"策略随数据性质定"落地(判据 = 场间基线差 vs 场内动态范围):

```
列                    策略        方向   τ 的来源
au12_smile            声明常量     >     0.263 = 全语料 41 场 p75(**未标定**)
au4_frown             自比基线     <     中位数 − 3·MAD
au23_lip_compression  自比基线     <     中位数 − 3·MAD
gaze_direction_x      自比基线     >     中位数 + 3·MAD(先取 |·|)
head_yaw              自比基线     >     中位数 + 3·MAD(先取 |·|)
```

**为什么 `au12_smile` 用常量而其余用自比基线**:它的生产方**已经**除掉了本人基线
(口宽 ÷ 全程 p10,见 `emotion_engine` 登记),场间差异是真行为差异(中位数 0.000–0.612);
其余四列的场间基线差 ≥ 场内动态范围,绝对阈值量的是脸型/机位。

**方向 bug(修掉的两个,都是真错)**:

* `au4_frown` = `dist(lm[107], lm[336]) ÷ 眼距` —— **眉内距 ÷ 眼距,越小越皱**。
  原公式写 `au4 > τ`,量的是"眉展得多开"。`research_mapper` 里早有记录
  「眉内距**越小**眉越皱」,但 `l1_face` 沿用了 `emotion_engine` 里那条
  `au4 > 0.3`(那条自己标注了"门限待 M3.5 重登记")。
* `au23_lip_compression` = 唇红厚 ÷ 面高 —— **越薄越压**。旧公式是
  `1 − 当前唇厚/静息唇厚`(带 `1−`),换口径时 `1−` 被去掉、**方向翻转而阈值没重登记**,
  于是 `au23 > 0.2` 变成"结构性不可达"(三场 max 0.085)。方向同样该是 `<`。

**改名**:构念从"绝对有多皱"变成"比他自己平时更皱多少" ⟹ 4 条指标名加 `_vs_baseline`。
不改名就是偷偷换构念。

**`au_activity_index` 仍然不可出**:可信 AU 里只有 `au12_smile` 能解析 τ,
拿一条算"联合占比"等于把它改名叫 `smile_ratio` ⟹ 如实报 `uncalibrated`。

**效果(13 场 A 档)**:每场有值 **18 → 26 条**(合计 234 → 336),
`话语结构特征` / `情境行为稳定性` 从置信"无"升到"高"。

### τ 的真相:卡的不是"没标定集",是"跨人基线差"

那 9 条 τ 相关 F 指标(微笑/视线偏离/皱眉/唇压/头部偏离/AU 活跃度)全语料无值。
原先记的理由是"τ 只能来自标定集(M5)"。**实测 22277 行 / 41 场后,这个说法要改**:

`au4_frown` / `au23_lip_compression` **不是 FACS 强度,是几何比值**
(眉内距÷眼距、唇红厚÷面高)。`au4_frown` 的**场内**动态范围(p10→p90 = 0.038)
与**场间**基线差(0.290→0.337 = 0.047)**同一量级** ⟹ 统一 τ 量到的主要是
"这个人的脸长什么样"。τ=0.31 时基线 0.337 的人 ~90% 帧判"皱眉",基线 0.290 的人 ~10%。

三条路必须由使用者裁定(不能由代码偷偷选):**(甲)自比基线**(τ = 本场中位数 + k·MAD,
诚实可算但**必须改名**,构念从"绝对皱眉"变"相对本人偏离");**(乙)等 M5**;
**(丙)声明常量**(仅 `au12_smile` 勉强可行 —— 场间中位数 0–0.61 是真差异,
且有 12% 恰为 0 的天然锚点;`head_yaw` 还要先取 `|·|`,符号跨场会翻)。

### 有效素材分级(M4.11)

41 个录制目录 = 实名正式场 24 + 无名测试场 17。按"脸 ≥80% 且臂 ≥70%"分档:
**A 13 / B 4 / C 1 / D 4 / ? 2 / T 17**。取景决定以
`data/logs/.face_backfill_manifest.json` 为唯一来源。

### 模态实际覆盖率:报告层此前只看"日志在不在"(2026-10-18 定案)

全语料实测:面部日志 21861 行里 **25.5% 整行为空**,**13 场是每一行都空**;
手势 12 场肘角 0%、2 场一帧姿态都没检出;研究模式 11 个文件全部 **0 行**。

危险在于:这样的会话**时间轴成立、`timebase.usable` 为真**,于是 16 条 F 指标
全部 `n_valid=0`,而质量报告只说"face 已提供"。读者会把"提供"读成"有数据"。

**处置**:`modality_coverage()` 算每个模态**实际可用行占比**,`provided` 与
`usable_ratio` 两个字段**分开**;`validate()` 新增硬不变量:**名义存在、实际为零**
的模态也不许有值(原来那条只拦"模态缺失",`usable` 为真时抓不到它)。

### 重算姿态能逐帧复现原日志 ⟹ **不重写特征提取**

用 `pose_landmarker_full.task` + 仓内 `PoseDetector` + `pose_angles` 重算
`20260928_212923_f98d` 前 30 帧:重算 30/30 有值、日志 30/30 有值,
**差异中位 0.0100 度、最大 0.59 度**(四舍五入),前 5 对完全一致。

⟹ 盘上的 21176 帧 face jpg + 21176 帧 gesture jpg(全 1280×720)就是产出这批日志的
那条链。**重写不会改变任何读数**,只会花 23 分钟(0.065 s/帧)复现同一结果。

12 场肘角 0% 也**不是管线 bug**:`95a8` 重算 40 帧姿态检出 40/40,但肘/腕可见度
中位仅 **0.104 / 0.114**(低于 `VISIBILITY_FLOOR`)⟹ 角度被正确扣留;`3d8f` 一帧姿态
都没检出。**信息不在画面里** —— 要补只能改采集(机位/取景)。

### V 族:从"只有 4 条"到"12 条全可产"(2026-10-18 定案)

**先前的判断是"粒度错配、要改采集侧",那个判断只对了一半 —— 修法不是改采集,是重算。**

`prosody_extractor` 里 `librosa.pyin` 本来就产出**逐帧**的 `f0 / voiced_flag / voiced_prob`,
逐帧 `rms` 也算了;**只是把序列扔了,只留 mean/std/p10/p90**。聚合量在数学上反推不出
中位数、分位数、首末差、每秒变化量 —— 于是 12 条里有 8 条被判"不可产"。

**素材其实一直在盘上**(实测):

```
recordings/<会话>/media/camera.webm               37/42 场  VP8 1280x720 + Opus 48 kHz
recordings/<会话>/media/audio/NNNN_converted.wav   10 场,102 段(16 kHz 单声道)
recordings/<会话>/retention.audio.jsonl            用 source_endpoint 把提问/回答分开
```

`source_endpoint` 实测是 `/asr`(提问)与 `/interview/answer_audio`(回答)**交替**,
⟹ 回答音频可以**逐题**取到,**不需要跨模态对齐**。

**处置(M4.9)**:新增 `voice_series.py`,从原始音频重算逐帧序列并缓存 `npz`;
`l1_voice_series.py` 据此产出那 8 条。实测成本:30 s 音频 pyin 1.81 s
⟹ 整场 10 分钟约 36 s;7 场 44 段共 **38 s** 跑完,缓存 29 KB/场。

实测产出(`20260927_212143_95a8`,8 题):

```
speech_rate 4.01 字/秒   voiced_ratio 0.577    pause_rate 29.26 次/分
pause_duration_p50 0.83 s  p90 1.98 s          pause_ratio 0.382
response_latency 8.16 s    pitch_median 97.5 Hz
pitch_range_semitone 2.15  pitch_variability_per_sec 24.68
energy_median 0.047        energy_dynamics_per_sec 0.853
```

7 场横向:`voiced_ratio` 0.19~0.88、`pitch_median` 95.8~240.0 Hz、
`pause_duration_p90` 0.25~13.05 s —— 量级合理,且**区分度明显**(旧的四条几乎不动)。

**两条口径必须钉住**(已写进模块 docstring 与用例):

1. **停顿 = 连续非浊音 ≥ `MIN_PAUSE_SEC`(0.2 s)**,含首尾静音。这是**定义**不是标定,
   常量住在 `voice_series` 里;改它会改全部 `pause_*` 读数,所以必须可审计。
2. **`pitch_range_semitone` 严格按规格的"首末差"**(`12·log2(f0首/f0末)`,带符号),
   **不许**偷换成 p10/p90 —— 那是另一个统计量。首末各只一帧、噪声大,这是规格的已知代价;
   要改就改规格并改名,不许在实现里"优化"。

**顺带解掉一条旧账**:`evidence_gate.QUARANTINE["speech_ratio"]` 的 unblock 一直写着
"M3 真 VAD"。真 VAD 现在有了(逐帧 `voiced_flag`),但**不是**把 `speech_ratio` 修好
——自指阈值那条病因改不掉,只能换掉。替代品是 `voiced_ratio`(实测全语料 `speech_ratio`
**100%** 恒 1.0,52/52 文件)。

### 附:先前"只有 4 条"的记录(保留,作为判断演进)

规格 §5.4 的 12 条大多定义在**音频帧**上,而语音日志是"一段回答一行、只存聚合量"。
按原公式精确可产的只有 `speech_rate` / `pause_rate_per_min` / `pause_ratio` /
`response_latency_sec` 四条;其余 8 条报 `not_stored`(新 gap 类型),
**不是** `uncalibrated` —— 前者要"采集侧多存一列",后者要"等 M5 标定",补救动作完全不同。

要补齐只需让 `prosody_extractor` 落盘帧级序列(或至少 p50 与首末值);
本层改不了,也**不许**拿 `pitch_mean` 顶 `pitch_median` 这类同名异义替换。

### 实测产出(2026-10-18,真语料)

48 场同时有语音与面部日志,其中有题库台账的 9 场。逐场跑通的结果:

```
场次                     时间轴  窗口  指标  有值  gap 构成
20260927_212143_95a8     ok      8   128   35   uncalibrated 72 / column_empty 16 / no_samples 5
20260927_195528_afb2     ok      7   112   30   uncalibrated 54 / column_empty 12 / no_samples 16
20260927_191325_253a     ok      1    16    5   uncalibrated 9 / column_empty 2
```

**这轮的真价值不是"算出 35 个值"**,而是那 72 条 `uncalibrated` 与 16 条
`column_empty` 现在**可查询、可统计、不会被悄悄填成 0**。

途中被 `L1Indicator` 的不变量当场抓到两个真 bug(都已修 + 补测试):
1. `head_motion_per_sec` 在整列 NaN 时报 `0.0`(把"没观测"说成"没动");
2. `voice_interview` 与 `voice` 两套键名并存,导致质量报告说"voice 未提供"而其实有。

**三条不变量**(贯穿全文件,违反即返工):

1. **不出复合分,不出跨指标加权。** L1 的 38 个指标**互不相加、互不比较**。每个都是"本场这个读数 + 它的证据链"。
   这条是甲-22 / 2026-10-04「折算层整段删除」的直接延续:那次删的是"分数 → 档位",这次不许从"指标 → 维度分"绕回来。
2. **不引入任何学习式融合。** 理由不是"保守",是**许可**:在 RecruitView 分布上拟合出来的参数(含融合权重)
   用于真实招聘/筛选/心理画像是明确禁止的。见 §8。
3. **算不出来就说算不出来。** 任一前置缺失 ⟹ `value = NaN` + `confidence = "无"` + 写进 `evidence_gaps`。
   **不猜、不补 0、不退回"模态行数"**(后者是本轮之前 `n_valid` 高估 16 倍的成因,见 `tests/test_l1_sample_size.py`)。

---

## 1. 为什么 M4 不是"多模态融合"

设计文档 §9 的里程碑表写的是:

```
| M4 | L1 指标层 + 元数据结构 + evidence_gaps | M3 |
```

M4 是**单模态内部的会话级聚合**。"把几个模态合起来判一个人"是 **L2**,而 L2 被 **M5 的伦理审查**卡着
(`docs/下一步.md` §五:M5 ← 卡许可 + 伦理审查)。

⟹ 本轮 M4 **不产出任何一个"综合判断"**。它产出的是**一组各自独立的读数**,以及每个读数"能信几分"。

---

## 2. 时间:三个模态现在用的是三种基准(实测)

这是本轮**最硬的一个发现**,先把它钉在这里。

### 2.1 现场

| 模态 | 日志里的时间列 | 实际语义 | 出处 |
|---|---|---|---|
| **face** | `timestamp` | **会话相对秒**(`SessionClock` 从 0 起算,严格递增) | `face_expression/pipeline/video_pipeline.py:213`(`timestamp_ms / 1000.0`) |
| **gesture** | `timestamp` / `timestamp_iso` | **绝对墙钟**(logger 内部 `datetime.now()`) | `gesture_analysis/utils/logger.py:353-354` |
| **voice** | `unix_timestamp` / `timestamp` | **绝对墙钟**(音频到达时刻) | `voice_interaction/api/app.py` 韵律日志 |
| **frame 模态的对齐锚** | `retention.<writer>.jsonl` 的 `declared_ts` | **会话相对毫秒**(两个 frame 模态各有各的 0 点) | `face_expression/api/app.py:380`、`gesture_analysis/api/app.py:497` |

### 2.2 实测:**账本里的墙钟会倒退,declared_ts 不会**

这一条是本轮最重要的发现,而且它**推翻了我自己的第一版判断**(第一版以为
"32 s 的差"是相机 0 点与会话原点之间的空窗)。逐帧核完 34 场之后,结论是反的。

判据很简单:`received_at_wall` 取自 `time.time()`(`media_retention._record`),
会被系统时钟调整 / VM 时钟漂移**回拨**;`declared_ts` 取自
`session_clock.SessionClock`,内部是 `time.monotonic()`,**物理上不可能倒退**。

```
场次                       帧数   墙钟倒退次数   declared跨度  wall跨度    差
20260925_230045_b901        50        0           47.0      48.0      -0.9
20260926_153202_2b11       300        6          316.7     299.9     +16.8
20260926_153854_1592       544       17          582.0     549.7     +32.2
20260926_155559_caf0       574       18          606.4     573.7     +32.7
20260928_172642_1320      1491        0         1070.6    2048.1    -977.5
20260927/28/29 其余 28 场      —         0            —         —       0.0
```

三种形态:

* **09-26 那三场**:墙钟每隔约 30 帧回拨 0.7–0.9 s(累计 -14.1 s),而 declared
  逐帧稳定 +1.07 s、**一次都没有倒退** ⟹ declared 才是真实经过时间;
* **09-28 `1320`**:墙钟跑得比单调时钟快一倍(2048 s vs 1070 s),同样不可信;
* **09-29 之后**:两列逐帧相同,没有这个问题。

⟹ **主时基必须是 `declared_ts`。** 墙钟只能用来求"跨模态相位",而且**必须先体检**。

### 2.3 处置:三档状态

| 档 | 判据 | 做法 |
|---|---|---|
| **`OK`** | 墙钟单调,且与 declared 总跨度相差 ≤ 5% | 时间轴 = `declared + 相位` |
| **`WALL_UNRELIABLE`** | 墙钟有倒退,或两列速率差 > 5% | **模态内仍用 declared**(它没坏);跨模态相位带上误差上界,并在 `gap` 里写明 |
| **`UNRECONSTRUCTIBLE`** | 账本缺帧 / 行数与日志对不上 / 锚是 `None` | 整列不换算,`t_sec` 全 NaN |

**关键取舍:`WALL_UNRELIABLE` 不是"不可用"。** 模态内的时间窗(眨眼、注视、头姿)
只依赖 declared,墙钟坏了不影响它们。受影响的只有"把两个模态放同一条轴上"这件事 ——
而那正是 M4.1 要显式登记的东西,不是要偷偷绕过去的东西。

**不许做的事:**
1. 用 `wall − declared/1000` 的某个常数去补 —— 实测那是纪元量级(1790408xxx),
   减完得到的是负的纪元时间,不是"距会话秒";
2. 用 `(logged − offsets) − origin` 这类"数学上成立"的写法 —— 它会把 offsets 的
   逐帧变化**恰好抵消**,于是漂移检测永远报 0(实现时踩过,已写进测试);
3. 因为墙钟坏了就宣布"整场时间轴不可用" —— 那等于把 3 场本可用的素材丢掉。

---

## 3. 事件窗口(M4.1)

### 3.1 旧路:退化解

2026-10-04 之前,窗口粒度是"一整场",于是:

- `speech_ratio` = 语音时长 / **整场时长** ⟹ 实测 **87.9% 恒为 1.0**;
- `pause_duration` 的单位错配就藏在这个"分母是一个会话常量"的假设里。

这两条都不是"参数没调好",是**粒度没定义**。

### 3.2 新路:每回答一个窗口

窗口 **k** = `[ask_end_k, ask_end_{k+1})`,最后一个窗口取到会话结束。

**音频段与题号的对应靠账本,不靠猜语种。** 实测(`20260926_153854_1592`):

```
账本里 source_endpoint == "/interview/answer_audio" 的 raw 段 = 8 段
questions.jsonl 的题目                                   = 8 条
按「最后一个 ask_end ≤ 该段 received_at_wall」推断 ⟹ 8/8 单调且一一对应
```

这条推断用到的两个量(`ask_end`、`received_at_wall`)都**已经落盘**,所以它是**可复核**的,不是猜的。
推断不成立(段数 ≠ 题数,或映射不单调)⟹ 退化为会话级窗口 + `window_granularity = "session"`,并在 gaps 里写明。

### 3.3 这一轮不碰的

- **26 场「老师面试学生」自然对话**:没有题库台账,`reaction_time` 定义上就不存在。
  ✗ 不实现,✗ 不伪造。等戊-15 裁定。

---

## 4. 指标元数据结构(M4.2)

### 4.1 形状

一条指标是一个 dict,键集**固定**(多一个少一个都是缺陷):

```json
{
  "name": "blink_rate_per_min",
  "family": "F3_blink",
  "formula": "count_rising_edges(is_blink) / (valid_span_sec / 60)",
  "required_l0": ["is_blink", "timestamp"],
  "window": "answer",
  "value": 14.0,
  "unit": "次/分钟",
  "n_valid": 187,
  "n_total": 200,
  "confidence": "中",
  "gap": null
}
```

`gap` 非 `null` 时,`value` 必须是 `NaN`,`confidence` 必须是 `"无"`。
这三者的一致性**是机器校验的**,不是靠自觉。

### 4.2 四档 `confidence`,而且四档都**必须可达**

| 档 | 判据 |
|---|---|
| `高` | G1–G4 全过,`n_valid ≥ 2 × 该族阈值`,且 `n_valid / n_total ≥ 0.8` |
| `中` | G1–G4 全过,`n_valid ≥ 该族阈值` |
| `低` | G1–G4 全过,但 `n_valid` 低于上面两条的比例线 |
| `无` | G1–G4 任一不过 ⟹ `value = NaN` |

**为什么要写"四档必须可达"**:2026-10-04 之前 `confidence_from()` 只有"高/中""中/低"两对可选,
`"高"` 与 `"无"` 都不可达,于是**零输入也报"中"**。这一轮的第一个验收就是:构造四个输入,四档各出一个。

G1–G4 的判据**不重写**,直接调 `report_frontend.evidence_gate.gate()` —— 全仓只有一份实现。

---

## 5. F 族 17 个指标(M4.3)

F 族是**唯一不需要跨模态对齐**的一族:全部由 face 日志 + answer 窗口得出。所以它先做,且不 block 在 M4.0 上。

`τ(x)` = 该量的标定阈值,**来自标定集、跨会话固定**(设计文档 §5.1 规矩 1)。
**本轮拿不到标定集 ⟹ 所有 `τ(x)` 的指标本轮一律 `confidence = "无"`,`value = NaN`,
gap = `"uncalibrated"`。** 不许先用视频内分位顶着 —— 那会让指标对每个人都 ≈ 常数,零信息量。

| # | 指标 | 公式 | 本轮状态 |
|---|---|---|---|
| F1-1 | `smile_ratio` | `au12 > τ(au12)` 帧占比 | 卡标定(`uncalibrated`) |
| F1-2 | `smile_rate_per_min` | au12 穿越 τ 的上升沿数 ÷ 有效分钟 | 卡标定 |
| F1-3 | `smile_duration_p50` | 连续超阈段时长中位数 | 卡标定 |
| F2-1 | `gaze_aversion_ratio_vs_baseline` | `|gaze| > 本人中位数+3·MAD` 占比 | **可产**(自比基线) |
| F2-2 | `gaze_aversion_rate_per_min` | 偏离→回正次数 ÷ 分钟 | 卡标定 |
| F2-3 | **`gaze_stability`** | `1 − std(gaze_direction_x) / 视频内极差` ← **归一化,允许用视频内分位** | **可出数** |
| F3-1 | **`blink_rate_per_min`** | 上升沿数(`is_blink` 0→1)÷ 有效分钟 | **可出数** |
| F3-2 | **`blink_duration_mean`** | 连续 `is_blink=1` 段时长均值 | **可出数** |
| F4-1 | **`head_motion_per_sec`** | `|Δyaw,Δpitch,Δroll|` 合成量 ÷ 二次 | **可出数** |
| F4-2 | `head_away_ratio_vs_baseline` | `|yaw| > 本人中位数+3·MAD` 占比 | **可产**(自比基线) |
| F5-1 | `brow_furrow_ratio_vs_baseline` | `au4 < 本人中位数−3·MAD` 占比 | **可产**(自比基线;**方向已修正**) |
| F5-2 | `lip_compression_ratio_vs_baseline` | `au23 < 本人中位数−3·MAD` 占比 | **可产**(自比基线;**方向已修正**) |
| F5-3 | ~~`tension`~~ | 合成别名 | **不出**(它不是独立测量,且是加权合成 ⟹ 违反不变量 1) |
| F6-1 | **`au_activity_index`** | 可信 AU **各自超自己均值**的联合占比 | 卡标定(需 AU 级 τ) |
| F6-2 | **`au_variability_per_sec`** | 可信 AU 每秒变化量(Σ\|Δau\| ÷ 秒) | **可出数** |
| F7-1 | `micro_exp_rate_per_min` | 检出次数 ÷ 分钟 | **不可出**:`micro_exp_*` 四列实测 0/1086 非空 |
| F7-2 | `micro_exp_duration_mean_sec` | — | **不可出**:同上 |

合计 17 条(F5-3 不计,F 实际测量 16 条 —— 与设计文档 §5.7 的注一致)。

**本轮真正能出数的是 5 条**:F2-3 / F3-1 / F3-2 / F4-1 / F6-2。
其余 11 条**如实报"卡标定"或"不可出"**,不出数。这不是 M4 没做完,这是 M4 诚实的形状 ——
阈值只能来自标定集或物理下限,而标定集属 M5。

### 5.1 `blink_rate_per_min` 的一个坑(实测)

face 日志里**已经有一列** `blink_rate_per_min`,但它是**滑动窗口**量
(`video_pipeline.py:288-289`:`sum(1 for t in blink_times if t > now - 60)`),
即"最近 60 秒内的眨眼次数",**不是全窗覆盖率**。直接对它取均值 ⟹ 低估。

L1 的 `blink_rate_per_min` **必须由 `is_blink` 的上升沿重新计数**,不复用那一列。
这条要写进测试:同一份合成数据,"上升沿法"与"对现成列取均值"必须给出不同的结果。

### 5.2 F7 为什么整族不可出

实测(6 份 face 日志、1086 行):

```
micro_exp_au_name / micro_exp_duration_frames / micro_exp_intensity / micro_exp_onset_frame  0/1086 非空
```

⟹ 这不是"样本量不够",是**没采集**。G1 会拦住它,而本轮做的事是**让它被拦住时说出来**,
不是想办法绕过 G1。

---

## 6. 与现有报告层的关系

**本轮不动 `research_mapper.py` 的 14 个槽。** L1 是新的一层,先独立产出、独立验收;
替换那 14 个槽是 **M4.5**,单独一片,单独验收(判据:删掉旧槽后,报告里没有一句不成立的陈述)。

这样切分的理由与 `2026-09-22-...-evidence-gate-design.md` 一致:**新东西先证明自己能站住,再去动已经被信任的东西。**

---

## 7. 每一片的红法(这个仓库的纪律:测试必须能被打红)

| 片 | 红法(改回去 ⟹ 测试必须红) |
|---|---|
| M4.0 | 把 gesture 的墙钟换算换成"首帧常数偏移" ⟹ 漂移场次的断言红 |
| M4.0 | 让缺失锚点的模态静默跳过而不是登记 `timebase_unreconstructible` ⟹ 红 |
| M4.1 | 把窗口改成会话级 ⟹ `n_windows == n_questions` 的断言红 |
| M4.1 | 让段数 ≠ 题数时仍然按序硬配 ⟹ 红 |
| M4.2 | 把 `confidence_from` 的两条分界线删掉 ⟹ 四档可达性测试红(某一档不可达) |
| M4.2 | 允许 `gap != null` 而 `value` 非 NaN ⟹ 一致性测试红 |
| M4.3 | 用现成的 `blink_rate_per_min` 列代替上升沿重数 ⟹ 红 |
| M4.3 | 把 `τ(x)` 指标改用视频内分位 ⟹ "非标定阈值必须为 `uncalibrated`"的红 |

---

## 8. 为什么**不**引入学习式多模态融合(留档,免得下一轮重新论证)

这一节不是"没来得及做",是**做完调研后明确不做**,理由按重要度排序:

1. **许可红线。** 数据集许可禁止"数据集及其上训练的模型"用于真实招聘/筛选/心理画像,
   **在数据集分布上拟合的参数(含 L1 阈值)同样属禁**。任何在 RecruitView 上学的融合权重都不例外。
2. **能力上限已经到头。** 本仓 `experiments/fusion/run_fusion.py` 实跑过:行为单模态 macro r = 0.329,
   early/late 融合 = 0.393;而审查文档记录该数据集标注者间一致性给出的上限约 0.39
   ⟹ **融合已经把能拿的都拿到了**,再加架构没有空间。
3. **那条线的文本特征有循环嫌疑。** `run_fusion.py` 的文本输入含 `gemini_summary`
   (LLM 写的人格分析,直接点名 OCEAN 维度)⟹ `text_rf ≈ 0.354` 被系统性高估。
   在这样一条线上得到的任何融合结论都不可搬。
4. **它们解决的问题与我们要解决的相反。** 近年的"缺失模态鲁棒"路线
   (因果模块 + 模态重建 + 分布漂移迁移;推理时动态选模态)**目标是"模态缺了也照样算分"**,
   而本系统的判据是**"模态缺了就不许算"**。方向相反 ⟹ 引入等于给"没采到"重新装上一张能出数的嘴。

⟹ M4 只借**对齐/归一/显式降级**这三类方法论,**不借任何融合器**。

---

## 9. 验收

一片算做完的标准(每片各自一份,不合并):

1. 新增测试全绿,且**裸 `pytest -q` 不能因为本片变红**;
2. 每一片有至少一条**可执行的红法**,并**实测**过(不是声称);
3. 没有新增写死常数(`τ(x)` 除外,且它们必须是"卡标定"的状态而不是数字);
4. `docs/下一步.md` 的里程碑表同步一行(M4 的状态)。

---

## 10. 开放问题(需要人拍)

| # | 问题 | 我倾向 |
|---|---|---|
| Q1 | 标定集拿不到之前,那 11 条卡标定的指标是"报 `uncalibrated`"还是"根本不列出来"? | **报 `uncalibrated`** —— 不列出来等于让读者以为 F 族只有 5 个指标 |
| Q2 | `gaze_stability` 用视频内极差归一化,而极差对离群点极敏感 | 本轮照设计文档原文实现,**并把"极差 vs IQR"两种实现的差写进测试**,等标定集再定 |
| Q3 | 26 场自然对话(戊-15)本轮完全不碰,同意吗? | 同意 —— 粒度没定义之前碰它只能造假 |
