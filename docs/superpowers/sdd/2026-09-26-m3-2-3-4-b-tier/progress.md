# SDD ledger — plan: docs/superpowers/plans/2026-09-26-m3-2-3-4-b-tier.md

## Setup

- 工作目录:仓库根,分支 `main`(HEAD `adb6990`)。**未新建 worktree** —— 本仓既有惯例是各里程碑直接提交到
  `main`(§0.11 记 `3ced9ca..adb6990` 共 20 个提交),且三个采集服务与真素材路径是仓库外绝对路径,
  worktree 会让「跑真产出方」这条验收腿指向错误的代码副本。
  `Ruling: 在 main 上就地执行(不建 worktree) — 与既有里程碑惯例一致,且验收必须跑仓库外真素材 —
  代价:若中途废弃,回退要按提交逐个还原而非删分支。`
- 任务清单 = `l0_columns.json` 筛 `status=pending` + `maturity=B`。当场数得 **33 条**:
  `gesture 19 / face 10 / text 3 / voice 1`。扣除计划 §1 明写「不在本计划内」的 `text` 3 条 ⟹
  **B1=1(voice)/ B2=10(face)/ B3=19(gesture)**,与计划 §3 逐数相同;另加 **B0=1**(E7)。
- 基线脏:`git status` 43 条未提交(含上一轮 §0.12 的 `docs/下一步.md` / `docs/资源总览.md` /
  `tools/jx_inventory.py` 等)。本计划只提交自己改动的文件,不替上一轮收口。

## Pre-flight scan(共享接口)

计划 §3 的四批并非互不相干:三批都改**采集层产出方**,而 B0 改的是**报告层的匹配口径**,
两者在 `is_quarantined()` 处交汇。逐条列出「先产出 → 后被消费」的接口:

| # | 生产方 | 消费方 | 接口内容 | 结论 |
|---|---|---|---|---|
| I1 | B0 `is_quarantined` 匹配口径 | B1/B2/B3 全部列 | 三批改的列(如 `au4_frown`、`*_jitter`)都以 `<模态>_<基名>_<统计量>` 扁平键进 G4 | **见 R1** |
| I2 | B0 两条 permanent 键 | B3 的 6 个 `*_jitter_world` | `jitter` 子串会顺带封米制 jitter 族(L0 行 `quarantine_ref` 已写明) | 不相撞:选项 (c) 只加豁免、不放宽子串 |
| I3 | B1 `pitch_trend` 单位改半音 | `pitch_direction`(白名单列,非本计划) | `pitch_direction` 的 ±10 **Hz** 门限直接吃 `pitch_trend` 的值 | **见 R2** |
| I4 | B2 `au4_frown` 改 landmark | 封停键 `au4_freq`(报告槽关键字,精确匹配) | 改的是 landmark,不改列名 ⟹ 封停匹配不受影响 | 不相撞 |
| I5 | B3 jitter 归一化 | 封停键 `jitter`(子串) | L0 行写明 unblock 条件是「M3 归一化(除肩宽除时间)」⟹ B3 做完应否解封? | **见 R3** |

无 I1/R1 之外的空接口行。

### R1(预检裁定)

`Ruling: B0 的匹配口径改动只影响 G4 的判定,不改任何列的产出值;B1/B2/B3 的验收一律以
「列值是否按 acceptance 变化」为准,不以 G4 是否放行或拦截为准 — 否则会把「B0 让某个指标
通过/被拦」误当成「B1/B2/B3 改对了」,这正是本仓发生过 5 次的「验证跑错了对象」。
代价:若某列的 acceptance 本身依赖 G4 放行(逐行读后:无此情况),会漏掉一条验收。`

### R2(预检裁定)

`Ruling: B1 必须同时处置 pitch_direction 的 ±10 Hz 门限 — 该门限直接吃 pitch_trend 的值,
pitch_trend 改成半音后、门限若仍按 10 解读,则「10 半音」≈一个八度,pitch_direction 会几乎
恒为「平稳」:一个白名单列被静默改坏。处置方式见 Task 1 的执行裁定。
代价:若实际另有代码把 pitch_direction 换算回 Hz(逐行读后:无),此处置多余。`

### R3(预检裁定)

`Ruling: B3 不改 QUARANTINE 表。L0 行的 unblock 字段写「M3 归一化(除肩宽除时间)」,但计划的
B0 只授权改 E7 点名的两条键(使用者 2026-09-27 裁定「只修 E7 两条」)⟹ jitter 族是否解封
属于 B0 之外的判定,登记在案、留给 M3.3 收口那一轮。
代价:jitter 列即使值已修对,报告里仍按「经 x5 归一化恒饱和」被拦 —— 但这**不是假话的残留**,
而是「解封判据尚未执行」,两者的区别已写进本行,不会被后来人读成「已收口」。`

## Tasks

### 计划缺陷(开工即遇)

`Ruling: 本计划没有 `## Task N` 结构 —— 任务清单在 `l0_columns.json` 里(计划 §1 明写),
故 `scripts/task-brief` / `task-done` 无法按 N 取brief。改用:批次 = B0/B1/B2/B3,
每批的 brief 逐行读 `l0_columns.json` 的 `acceptance`/`normalization`/`basis` 三个字段,
账本手工记。代价:失去脚本自动写入的 `Task N: complete(tests: <cmd> → <result>)` 行,
改由本文件逐批记命令与结果 —— 记录强度相同,只是不自动。`

### B1 — M3.4 语音(1 条:`pitch_trend`)✅

- **RED**:改 `tests/test_voice_first_order_columns.py::test_three_voiced_frames_do_have_a_trend`
  的断言(`100.0` → `12.0`)+ 新增 `test_pitch_trend_is_semitones_not_hertz`
  (八度 = ±12、纯五度 = +7)+ 新增 `test_pitch_direction_keeps_its_hertz_threshold`(R2 钉子)。
  实跑:`3 failed` —— 报错值 `np.float64(100.0) != 12.0`、`20.0 != 0.343`,
  **失败原因即「单位还是 Hz」**(不是拼写错、不是 import 错)。✓ 看着它红了。
- **GREEN**:`prosody_extractor.py` 加 `import math`;`pitch_trend = 12.0·log2(last/first)`;
  首末均值 ≤ 0 时留空;`pitch_direction` 改吃新增的局部量 `hz_diff`(**R2 的处置**)。
  实跑:`tests/test_voice_first_order_columns.py` → **21 passed**。
- **真产出方实测**(契约第 3 条,不是合成数据):`prosody_features_from_pcm` 吃三场正式素材
  `media/audio/*_converted.wav`(真 16k/16bit/单声道,断言的)⟹ 每场 17 段,**17/17 全出值**、
  0 空值。旧口径 vs 新口径:2b11 `[-235.4, 33.3] Hz` → `[-19.84, 5.00] 半音`;
  1592 `[-194.7, 198.2] Hz` → `[-10.62, 12.21]`;caf0 `[-78.0, 894.5] Hz` → `[-12.55, 40.37]`。
- **契约**:`l0_columns.json` `pitch_trend` pending → `implemented`;`schema_errors` 空;
  `tests/test_l0_column_table.py` 11 passed(双向钉子绿 —— 方向 2 确认三个 logger 真产出该列)。
- **全量套件**:`455 passed / 0 failed`(基线 453 + 本支 2 枚新钉子)。
- 提交 `7631d75`。
- ⚠️ **顺手修的一处静默过期**:该行 `source`/`acceptance` 里的**行号**(`prosody_extractor.py:89-93` /
  `:93`)被本次改动弄失效了 —— 行号是纪律 2 点名的那类手写数,已换成符号名
  `ProsodyFeatureExtractor.extract_pitch_features`,不再随行号漂移。

### B3 — M3.3 手势(19 条)⏳ 已派出(独立 implementer,fresh context)

`Ruling: B3 改由**独立 fresh-context implementer** 执行,不在本体内联做 — 理由:B0/B1/B2 已经把
本仓的约定、三条纪律与四类坑全部踩实并写进本账本;B3 是 19 列 + **新加一条重放腿** +
4 个 analyzer 的签名改造(要往 history 里塞时间戳、要拿到 `shoulder_width`),
体量与前两批相当或更大,而本会话的上下文已长 —— 执行计划技能自己写明「inline 的代价是
最后几个任务拿到最少的你」。代价:**它的自述必须我自己复核**(见下),不能转述了事。`

派出的 brief 里逐条写死了:三条纪律 / 19 行的 `definition`+`acceptance` 是权威判据 /
**必须先给 `replay_retained.py` 加手势腿**(计划 §7)/ 三场正式素材实测 / 分 1~2 个提交 /
只 add 自己改的文件(仓库里有上一轮遗留的未提交文件)/ 以及四类坑:
「验收判据可能不可达 → 先量基线再裁定,别硬凑」「第一次红得不对是常事」
「新列要三处都改才算落盘」「`json.dump` 整写会冲掉表的手工排版」。

### B3 复核结果(本体做的,不采信自述)—— 4 项全过,2 项待办

它交了两个提交 `029e783`(接线先行,不动数值)+ `4db35e3`(19 列口径 + 钉子 + 表)。

| 我查的 | 做法 | 结果 |
|---|---|---|
| 全量套件 | **自己跑** | **553 passed / 0 failed** ✓ 与自述一致 |
| 提交是否卷进上一轮遗留文件 | `git status` 逐个看 | ✓ `app.py` / `docs/下一步.md` / `tests/conftest.py` / `tests/test_log_isolation.py` **仍未提交**,一个没被带走 |
| L0 表被动了几处 | `git show 4db35e3 -- l0_columns.json` 数 status 行 | ✓ **恰好 19 个** `pending → implemented`,别的 status 一个没动 |
| 它改过的 3 个既有测试有没有被削弱 | 看 diff | ✓ 只是替身 `update()` 收下新的 `**_kw`;断言一条没删 |
| ★ **19 列的改前→改后** | **不信它的表**:拿**当天活路径写的** `data/logs/gesture_emotion_log_<sid>.csv`(=改前)与我**自己跑的手势重放腿**(=改后)逐列算 | ✓ **与它的表逐数吻合** —— `hand_spread` 0.1807→1.4760 / 0.1772→1.3592、`shrug_level` 0.2482→0.0133 / 0.6569→0.0719、米制 6 列一律 ≈÷30.6。且量级自洽:窗 ≈30 s(客户端约 1 fps)⟹ 反推 `shoulder_width ≈ 0.34`,是合理的归一化肩宽 |
| 改动面是否真的收窄 | NaN-感知的逐格比对,把列分成「19 列 / world 精度 / 派生分」三类 | ✓ **三类之外的意外差异 = 0**(三场都是) |
| 残留行号引用是否还有效 | 逐个打印所指的那一行 | ✓ config.py 5 处(`history_length`×2 / `fist_threshold` / `jitter_multiplier` / `baseline_frames_needed`)、logger.py 3 处,全部仍指向所称的东西 |

⚠️ **一处我一开始读错、必须记下来**(免得后来人重蹈):我第一遍比对报出 16 列"有差异",其中
`left/right_hand_model_label` 286/300 行不同 —— 那是**我自己比法错**:用了 `astype(str)` 比,
于是 `nan != nan` 全被算成差异,而**两边取值分布逐字相同**(`{'Right': 14}`)。
`left/right_arm_angle_world` 的差异则是它**已声明**的 `_WORLD_DECIMALS` 4→6(同值多几位小数)。
改成 NaN-感知比对后:意外差异归零。`timestamp` 列三场全不同,**已查明是既有行为、不是 B3 引入**:
gesture logger 的 timestamp 来自 `datetime.now()`(墙钟),活路径写的是"当时",重放写的是"今天"
⟹ 重放**永远**对不上这一列(它的自述说"逐格相同"时应是指 64 个测量列,不含 `timestamp`/
`timestamp_iso`/`session_id` 这 3 个非测量列)。

**两条待办(我发现的,不是它的错 —— 是我给它的 brief 没写)**:

1. **`*_stability` 三列本应在 M3.3 删掉,本批没删。** `l0_columns.json` 的 `legacy_allowlist` 里
   `left_arm_stability` / `right_arm_stability` / `torso_stability` 的 `planned` 字段写的就是
   **「M3.3 删列」**。而我的 brief 里写死了「本批不新增/删除日志列」⟹ 它照 brief 做了,
   是**我的 brief 与表的 `planned` 字段冲突**。后果:这三列是 jitter 的派生量,值随本批变了
   (实测 291/300、528/544、563/574 行),而它们**不在 QUARANTINE 里**(`is_quarantined` 实测放行,
   同族的 `*_score` 反而都被拦)。它们不是 L0 列、也没有映射槽关键字指向它们,所以**进不了打分**,
   但"报告说真话"的角度上,它们是三列**值变了却既没封停也没删**的残留。
   `Ruling: 留给下一轮 M3.3 收口处理(删列要同时动 logger fieldnames 与派生它的 `_compute_*`,
   与"本批不改日志列契约"的边界冲突,不在本支解) — 代价:这三列在下一轮之前是"改了但没人管"的状态。`
2. **重放腿的保真性我没能直接证实。** 它声称"重放输出与活日志逐格相同 0 格不一致",但那是针对
   `029e783`(未改数值时)说的,而我的重放在 `HEAD`(数值已变)—— 状态不同,比不了。
   **不必需**:验收要的"改前"值可以直接取**活日志本身**(它天然是改前),我用的就是这一路,
   比"重放能不能复现活日志"更直接。⟹ 登记为**未独立证实**(不是"已证伪")。

### 整支复核(第一次 OOM 中断,已重派)

第一次派的复核者**被 WSL OOM 杀掉**(用户跑实验时内存爆了;当时 WSL 内存不足,现已扩容)。
中断后核查:5 个提交全在、43 条未提交条目与开工时**同数同形**、上一轮遗留文件仍未被提交
⟹ **仓库没被中断弄坏**。跨会话不可 resume(ListAgents 里已无该 agent),故**重派**(brief 收紧:
禁跑全量套件 / 禁跑重放腿 / 禁大 DataFrame,只读代码 + 小规模合成实验)。

★ **趁中断自己把最有价值的那条裁定验了**(不依赖复核者):手势 acceptance ①「不再随帧率变」
**不可达 —— 而且是从 L0 表自身就能证明的矛盾,不需要做实验**:
该行 `basis` 的原话是「**窗长 30 帧**是现行代码常量…本行把它**定为** L0 口径」,
而 `acceptance` ① 要求「同一段动作在 1 fps 与 30 fps 下抽,值差 < 20%」。
窗长按**帧数**定 ⟹ 1 fps 下该窗跨 30 s、30 fps 下跨 1 s;而窗口内离散度(std)对窗口时长
按 **√T** 增长 ⟹ 两者相差 √30 ≈ 5.5 倍。**判据与它自己的 basis 直接冲突**。
⟹ 执行者判「不可达」**成立**,不是没做到而找理由。这条不用复核者再查了。

**我(本体)开工时列的四条复核项**:
1. 全量套件自己跑一遍,看尾部的实际数字(基线 471 passed)。
2. **独立**用手势重放腿跑三场正式素材,自己算改前/改后对照 —— 不接受"已修复"这种话。
3. 抽查它声称的 `Ruling:` 是否真的写进了 L0 行 / 提交信息。
4. 检查提交里**没有**卷入上一轮遗留的未提交文件(`tests/conftest.py` / `tests/test_log_isolation.py`
   / `app.py` / `docs/下一步.md`)。

### B2 — M3.2 面部(10 条:4 AU + symmetry + 4 iris + face_scale)✅

- **先复现基线**(改代码之前):`experiments/replay_retained.py` 跑三场正式素材的留存帧
  (300/544/574 帧,喂账本 `declared_ts`),量出 L0 行点名的症状 ——
  `au4_frown` 触地板 87.5%/97.9%/48.4%、`au23` 触地板 73.1%/23.9%/29.2%、
  `symmetry_score` mean 0.983/std 0.012(近常量)、`au9` mean≈0.49、iris 四列 mean≈0.49~0.57。
- **RED**:新建 `tests/test_face_l0_columns.py`(12 条)—— 此前这四个 AU 列**一条测试都没有**。
  每条构造"新旧口径必然分岔"的几何(而不是"值在合理范围"这种新旧都过的判据);
  分岔不出来的用**性质**判据(iris 平移不变、symmetry 缩放不变)。
  **第一次跑红得不对**:12 条全红,但多数是 `ZeroDivisionError` —— 我的假 landmarks 默认点
  全是同一个点,而 `_eye_aspect_ratio` 除以 `dist(lm[33],lm[133])`。
  ⟹ 红的理由是"崩了"不是"口径错了",那这条测试什么都没证明。改默认值为极小稀疏网格后重跑,
  12 条红,**逐条失败理由与预期一致**(如 `实为 0.5 —— 0.50 说明还在读 276/33`)。
- **GREEN**:逐列落地(下标 / 分母 / 相对坐标)。`eye_distance` 在 `calculate` 里**算一次传下去**
  (au4 与 au6 都用它 —— 各自重算在数学上等价,但只改一处就会静默分叉)。12 passed。
- **★ 本支最贵的一个坑(记下来)**:`face_scale` 是新列,必须**三处**都加才算真落盘 ——
  ① `AUFeatures` 字段、② `AnalysisFrameResult.to_dict()` 的**显式**字段表、③ `DataLogger.fieldnames`。
  我加完 ①③ 就跑了重放,**单元测试全绿**,于是差点以为成了;查输出列时才发现
  **重放 CSV 里根本没有 `face_scale`** —— `to_dict` 不是 `vars()` 展开,漏了不报错。
  这正是纪律 1 的形态:**单元测试跑的不是产出方**。补上 ② 后实测 n=271/477/479 帧有值。
- **列序坑**:先把它加在 `bs_*` 块**之前**,`test_face_blendshapes` 当场红
  ("52 列不是追加在锚点之后的一整块")—— 新列一律加在**列序最末**。钉子救了一次。
- **真产出方实测**(改后重放同一批帧):
  `au4_frown` 地板 87.5%/97.9%/48.4% → **0.0%/0.0%/0.0%**;`au23` 地板 → **0.0%/0.0%/0.0%**;
  `symmetry_score` std 0.0119→0.0692(近常量消失);iris 四列 mean 0.49~0.57 → −0.02~−0.21
  (从绝对坐标变成相对坐标)。`face_scale` 落盘 n=271/477/479。
- **契约**:10 行 pending → implemented(`schema_errors` 空;`live_contract.face` 91 → **92**
  —— 钉子 `test_live_contract_matches_the_real_log_contract` 在我改 logger 时当场红,
  是它把我叫去同步的)。全量套件 **471 passed / 0 failed**。提交 `f25430a`。
- ★ **一条验收判据被裁定作废**(au23 的 ①「与 au25 的相关移向 0 附近或正」)—— 计划 §6
  自己写着「复核前的自检:这条验收判据真的可达吗?」,答案:**不可达**。实测改前
  −0.368/−0.721/−0.499、改后 −0.471/−0.403/−0.357,没往 0 走。
  **没有就此收工**,而是查清了原因(这一步是关键 —— "没达到"与"判据错"是两回事):
  ㈠ 定义式的镜像**已彻底消失** —— 新定义根本不读 `lm[14]`,有钉子钉住;
  ㈡ 残余相关来自**解剖耦合**(张口时上唇变薄,au25↑ 与 au23↓ 是同一件事)
  与 **`au25` 自己的尺度污染**(实测 `au25~face_scale` = +0.853/+0.681/+0.780)。
  控制 `face_scale` 后偏相关仍为 −0.259/−0.522/−0.410 ⟹ 剩下的是解剖耦合,不是口径残留。
  `Ruling: au23 的验收①由「相关系数」改用「机制判据」(不读 lm[14] + 张口时逐帧不变) —
  原判据量的是另一个列的缺陷与真实解剖耦合,不是本列的口径 — 代价:若将来 au25 修好尺度
  污染后相关仍未变正,需重新审视本列的分子取点。`
  理由与实测数已写进 L0 行的 `acceptance`(不是只写在这里)。
- **顺手删死代码**:`MouthFeatureExtractor` 里的 `rest_lip_thickness` / `current_lip_thickness` /
  `lower_lip` 因本支改动变成**只写不读**,一并删除(5+1 行)。用户偏好:没用的删掉。

### B0 — E7 封停键对齐(report_frontend,1 条)✅

裁定(使用者 2026-09-27):**选项 (c) 子串 + 条目级豁免**,且**只修 E7 两条**。

- **开工前的自证**(纪律 1:先说清"跑的这条会不会执行到要证的那段代码")。
  我**没有**采信简报 §12.4 的引文,而是把真产出方跑了一遍:三场正式素材过
  `LogDataLoader` → `PsychologicalFeatureEngine` → 摊平。结果**与简报有两处出入**,
  都不影响结论、但影响"该照谁写":
  ① 简报/计划 §4 引的键是**双下划线**(`gesture__head_tilt`),**真键是单下划线**
     (`gesture_head_tilt_mean`)—— 只有 `__n_rows` 那三个键是双下划线。
  ② 简报只说"两条"永久封停打不中;**实测 8 条 `permanent=True` 里有 7 条打不中**
     (只有 `overall_score` 命中)。E7 点名的两条只是其中"有活产出方"的那两条。
- **RED**:先在 `evidence_gate.py` 里**只改名、不加豁免**(天真改法),实跑
  `test_head_tilt_does_not_swallow_head_tilt_angle` → **红**,
  报错原文:``gesture_head_tilt_angle_mean 是 maturity=A 的保留列,不该被封停 ——
  被 `head_tilt` 的子串匹配误伤了``。⟹ **豁免是承重的**,不是防御性代码。
  同一轮里"真键命中"那条已经是绿的 ⟹ 两条测试一起证明了「改名解决一半、豁免解决另一半」。
- **GREEN**:`Quarantine` 加 `not_substrings` 字段;`is_quarantined` 在**同一套**语义内
  排除(候选去掉被豁免者再取最长;候选空 ⟹ 放行,不退化成"随便挑一个")。
  实跑 `tests/test_evidence_gate.py` → **27 passed**。
- **改动面实测**(不是"看着对"):三场真键**并集 619 个**,新旧规则逐键比对 ⟹
  判定**只变 12 个**(6 个 `gesture_head_tilt_*` + 6 个 `gesture_is_calibrated_*`),
  `gesture_head_tilt_angle_*` 六个键**全部照旧放行**。零附带损伤。
- **契约**:全量套件 **459 passed / 0 failed**。提交 `db2f5ae`。
- ⚠️ **本支自己踩出来的一个坑(已修,记下来)**:`l0_columns.json` 里有 **78 处**
  `evidence_gate.py:NN` **行号**引用 —— 改前**我逐条核过,全部是准的**;而 B0 只往
  `evidence_gate.py` 加了 7 行,这 78 处**当场集体失效**,且**静默**(读表的人会照着
  错行号读错的代码)。这正是纪律 2 的教科书形态,而**是我造成的**。
  处置:78 处全部换成键名引用(`QUARANTINE['jitter']` / `is_quarantined`),并加钉子
  `test_quarantine_refs_point_at_real_keys` 守两条(不许行号 / 引用必须真实存在),
  **两条红法都实跑验证过**(注入 `QUARANTINE['jitter_world']` → 红;改回 `:63` → 红;还原 → 绿)。
- `Ruling: 简报 §12.4 的两处引文(键名拼写、永久封停条数)以**实测**为准,简报不改 ——
  它是当轮的过程记录;改的是 `l0_columns.json` 那条被引的散文(它是活的任务清单,
  描述的是**当前**状态)。代价:若有人只读简报不读实测,会得到"只有两条失效"的印象。`
- `Ruling: E7 归属冲突(l0_columns.json 写"归 M3.3"、§11.4 写"归 M3.5")就地消解为
  "已由 B0 收口" —— 不再挂账。代价:若后续发现 E7 还有残余,要重新指定归属。`
- **未做(按裁定)**:另外 11 条"从不命中"的死条目(含 7 条 permanent)与同族的
  `energy_variation` / `pause_frequency` 两处键/列错配 —— 均已在表里登记,留给 M3.3 收口。


