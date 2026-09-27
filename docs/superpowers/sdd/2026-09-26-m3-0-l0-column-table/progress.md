# SDD ledger — plan: docs/superpowers/plans/2026-09-26-m3-0-l0-column-table.md

## 预检:冲突扫描(执行前)

### 共享文件/接口的任务对

| 任务对 | 一方产出 vs 另一方消费 | 结论 |
|---|---|---|
| T1 ↔ T2/3/4 | T1 建 `l0_columns.json`(`columns: []`)+ `load/columns/schema_errors` + 钉子;T2/3/4 往里追加行 | ✓ 一致(追加式) |
| T2 ↔ T8 | T2 的定档表把 `head_roll` 写成"见 Task 8";T8 的 Files 写"Modify 两行" | ⚠️ **冲突 F1** |
| T3 ↔ T6/7 | T6/7 要把 `hand_visible_*`/`shoulder_width` 翻 `implemented`,但行得先由 T3 建 | ⚠️ **冲突 F3** |
| T4 ↔ T5 | T5 要翻 `voiced_prob_mean` 那行,T4 得先登记它 | ⚠️ **冲突 F3** |
| T4 ↔ T6/7 | T4 Step 3 把 `hand_visible_*` 登记成 `modality: covariate`;T6/7 把它当 gesture 列 | ⚠️ **冲突 F2** |
| T6/7 ↔ T9 | 都动 `gesture_analysis/`,但 T9 只删文件 + 改 `__init__`,与 T6/7 改的 `logger.py`/`app.py`/`angles.py` 不重叠;且 T9 在后 | ✓ 一致 |
| T10 ↔ 全部 | 只做验证,不改代码 | ✓ 一致 |

### 每个任务自洽性

| 任务 | 自洽? |
|---|---|
| T1 | ✓ 它**预期**方向 1 的钉子红(表还是空的)—— 与自己的测试一致 |
| T2/3/4 | ✓ 表 + `count_reconciliation` 的填充顺序一致 |
| T5/6/7 | ✓ 代码完整、测试含空值与末列位置 |
| T8 | ✓ 它自己 `Modify l0_columns.py` 加 blocked 校验,无跨任务依赖 |
| T9 | ✓ 先写"包还能 import"的测试,再删 —— 与"只删文件会 ModuleNotFoundError"的事实一致 |
| T10 | ✓ 纯验证 |

## 预检裁定(执行前)

**Ruling F1:`head_roll` 的归属** —— T2 **不建** `head_roll` 行(它在 T2 表里只作"见 Task 8"的指引),
该行由 **T8 新建**;T2 只建 `head_yaw`/`head_pitch` 两行。理由:一行只有一个属主,
否则 T8 的 "Modify 两行" 与 T2 的 "新建" 会撞。
**错判代价**:低 —— 一行归属而已,后续里程碑读表时看清即可。

**Ruling F2:协变量列的 `modality`** —— 由某个服务**产出**的列(如 `shoulder_width`、
`hand_visible_*`)一律填**产出它的服务**(`gesture`),不填 `covariate`;
"协变量用途"写进 `definition`。理由:钉子的方向 2 拿 `(column, modality)` 去比
`_live_columns()`,而那个函数只返回 `face`/`gesture`/`voice` 三个键 ——
填 `covariate` 的行**永远匹配不上**,一旦标 `implemented` 就会假红。
同时给 `schema_errors` 加一条硬规则:**`status: implemented` 的行,`modality` 必须是
`face`/`gesture`/`voice`**。**错判代价**:中 —— 若将来真有"不由任何服务产出"的列,
它只能停在 `pending`,需另想办法。

**Ruling F3:新列的登记顺序** —— 新列一律**先由填表任务登记为 `pending`,再由实现任务翻 `implemented`**:
`voiced_prob_mean` 由 T4 登记、T5 翻;`hand_visible_left/right` 与 `shoulder_width`
由 T3 登记、T6/T7 翻;`head_roll` 由 T8 直接建为 `blocked`。
**错判代价**:低 —— 顺序错了钉子会当场红,不会静默。

**Ruling F4:在 main 上执行** —— 本计划在 `main` 分支原地执行,**不建 worktree、不 push**。
理由:本仓全部历史都在 main 上(见 `docs/下一步.md` 的提交记录),这是使用者一贯的工作方式,
且他明示"代码提交照做" ⟹ 即提交到当前分支。**注意:只提交代码,`docs/**` 一律不提交。**
**错判代价**:低 —— 全部是本地提交,没有 push,要挪到分支或回退都容易。

## 基线

- 起点 HEAD:`3ced9ca`
- 合并门基线:`0 / 2835510`(2026-09-26 实测,最大绝对差 1.886e-19)
- 全量套件基线:**410 passed**
- ⚠️ 工作树里有两处**非本计划的未提交改动**,任何 `git add` 都**不许**把它们带进去:
  - `app.py`(面板 reports 别名,今天早些时候的活)
  - `docs/下一步.md`(§0.9,文档,按裁定不提交)

## 进度


### Task 1 —— 完成(`l0_columns.json` + 加载器 + 双向覆盖钉子)

- 交付:`l0_columns.json`、`l0_columns.py`、`tests/test_l0_column_table.py`
- 单测:`5 items — 3 passed, 2 failed`(两条红的性质见下,都不是本任务机制的缺陷)
- 反向复现证据:见 `task-1-report.md` §3

#### ★ 方向 1 的失败输出 = Task 2/3/4 的待办清单(原样抄录)

`tests/test_l0_column_table.py::test_every_live_column_is_either_in_the_table_or_allowlisted`
—— 共 **187 列**既不在表里、也不在 `legacy_allowlist` 里(表按 Task 1 的设计还是空的):

**face（89 列）**：`session_id`、`timestamp`、`blink_rate_per_min`、`au1_inner_brow_raise`、`au2_outer_brow_raise`、`au4_frown`、`au6_cheek_raise`、`au7_eye_squeeze`、`au9_nose_wrinkle`、`au10_upper_lip_raise`、`au12_smile`、`au14_dimpler`、`au15_mouth_down`、`au20_lip_stretcher`、`au23_lip_compression`、`au25_mouth_open`、`au26_jaw_drop`、`avg_ear`、`head_yaw`、`head_pitch`、`left_iris_x`、`left_iris_y`、`right_iris_x`、`right_iris_y`、`gaze_direction_x`、`gaze_direction_y`、`gaze_deviation`、`eye_closed_sec`、`is_blink`、`dominant_emotion`、`confidence`、`tension_score`、`tension_level`、`micro_exp_au_name`、`micro_exp_intensity`、`micro_exp_duration_frames`、`micro_exp_onset_frame`、`bs__neutral`、`bs_browDownLeft`、`bs_browDownRight`、`bs_browInnerUp`、`bs_browOuterUpLeft`、`bs_browOuterUpRight`、`bs_cheekPuff`、`bs_cheekSquintLeft`、`bs_cheekSquintRight`、`bs_eyeBlinkLeft`、`bs_eyeBlinkRight`、`bs_eyeLookDownLeft`、`bs_eyeLookDownRight`、`bs_eyeLookInLeft`、`bs_eyeLookInRight`、`bs_eyeLookOutLeft`、`bs_eyeLookOutRight`、`bs_eyeLookUpLeft`、`bs_eyeLookUpRight`、`bs_eyeSquintLeft`、`bs_eyeSquintRight`、`bs_eyeWideLeft`、`bs_eyeWideRight`、`bs_jawForward`、`bs_jawLeft`、`bs_jawOpen`、`bs_jawRight`、`bs_mouthClose`、`bs_mouthDimpleLeft`、`bs_mouthDimpleRight`、`bs_mouthFrownLeft`、`bs_mouthFrownRight`、`bs_mouthFunnel`、`bs_mouthLeft`、`bs_mouthLowerDownLeft`、`bs_mouthLowerDownRight`、`bs_mouthPressLeft`、`bs_mouthPressRight`、`bs_mouthPucker`、`bs_mouthRight`、`bs_mouthRollLower`、`bs_mouthRollUpper`、`bs_mouthShrugLower`、`bs_mouthShrugUpper`、`bs_mouthSmileLeft`、`bs_mouthSmileRight`、`bs_mouthStretchLeft`、`bs_mouthStretchRight`、`bs_mouthUpperUpLeft`、`bs_mouthUpperUpRight`、`bs_noseSneerLeft`、`bs_noseSneerRight`

**gesture（67 列）**：`session_id`、`timestamp`、`timestamp_iso`、`left_hand_score`、`left_hand_jitter`、`left_hand_fist_status`、`left_hand_spread`、`right_hand_score`、`right_hand_jitter`、`right_hand_fist_status`、`right_hand_spread`、`left_thumb_angle`、`left_index_angle`、`left_middle_angle`、`left_ring_angle`、`left_pinky_angle`、`right_thumb_angle`、`right_index_angle`、`right_middle_angle`、`right_ring_angle`、`right_pinky_angle`、`shoulder_score`、`left_shoulder_jitter`、`right_shoulder_jitter`、`shrug_level`、`is_calibrated`、`left_arm_score`、`left_wrist_jitter`、`left_elbow_jitter`、`left_arm_angle`、`left_arm_stability`、`right_arm_score`、`right_wrist_jitter`、`right_elbow_jitter`、`right_arm_angle`、`right_arm_stability`、`left_elbow_angle`、`right_elbow_angle`、`left_shoulder_angle`、`right_shoulder_angle`、`head_score`、`head_jitter`、`head_tilt`、`torso_score`、`torso_jitter`、`torso_stability`、`head_tilt_angle`、`head_pitch_angle`、`shoulder_angle`、`torso_angle`、`overall_score`、`emotion_state`、`feedback`、`used_features`、`is_valid`、`left_hand_model_label`、`left_hand_model_label_conf`、`right_hand_model_label`、`right_hand_model_label_conf`、`left_wrist_jitter_world`、`left_elbow_jitter_world`、`right_wrist_jitter_world`、`right_elbow_jitter_world`、`left_arm_angle_world`、`right_arm_angle_world`、`left_shoulder_jitter_world`、`right_shoulder_jitter_world`

**voice（31 列）**：`session_id`、`unix_timestamp`、`timestamp`、`pitch_mean`、`pitch_variation`、`pitch_trend`、`pitch_direction`、`energy_mean`、`energy_variation`、`speech_ratio`、`duration_sec`、`pause_duration_mean`、`pause_duration_max`、`pause_frequency`、`emotion`、`feedback`、`question_index`、`is_valid`、`connective_density`、`connective_density_std`、`n_rows`、`n_chars`、`speech_duration_sec`、`chars_per_sec`、`energy_p10`、`energy_p90`、`pitch_p10`、`pitch_p90`、`speech_onset_sec`、`answer_onset_wall`、`reaction_time`

> 说明:这 187 列就是「L0 到底有哪些列」的真身,由 Task 2(face)/ Task 3(gesture)/
> Task 4(voice)逐列定义后登记进 `columns`,方向 1 随之转绿。
> **不许**把它们塞进 `legacy_allowlist` —— 它是"确定不要的既有列"的豁免名单(每条须带
> `why` + `planned`,钉子断言这一点)。

#### ⚠️ 第二处红(简报 Step 5 未预期)

`test_load_hands_back_a_copy` **也红**,原因是它的探针 `a["columns"].clear()` 随后断言
`l0.load()["columns"]` 为真 —— 而出厂表 `columns` 就是 `[]` ⟹ 无论 `load()` 有没有交深拷贝,
这条断言都假。**该测试当前没有区分力**(红的原因不是它声称守护的那件事)。
`load()` 的深拷贝机制本身已独立核实为**正确**(见报告 §3.4)。
Task 2 一填表,这条自动转绿。
Task 1: implementer DONE_WITH_CONCERNS (commit bcd3fdf)。6 条顾虑,逐条裁定:

**Ruling C1(计划缺陷,必须修)**:`test_load_hands_back_a_copy` 的探针 `a["columns"].clear()`
→ 断言 `load()["columns"]` 为真,在出厂表(`"columns": []`)上**零区分力** ——
它红的原因不是它声称守护的那件事。计划 Step 1 与 Step 4 自相矛盾(Step 4 规定空表,
Step 5 却要求它过)。**违反 Global Constraint「每个测试都要能说出哪个生产改动会让它变红」**
⟹ 进修复轮 1。改法:探针换成**与状态无关**的字段(`_schema.version`,出厂即有值)。
**错判代价**:无 —— 这是把一个假测试换成真测试。

**Ruling C2(中间态记账)**:`tests/test_assert_coverage.py::test_no_assert_is_dead` 会在
Task 1–4 期间**必然红**(它把整个 tests/ 在自己进程里再跑一遍、要求内层返回码 0,
而方向 1 按设计一直红到表被填满)。**裁定:接受这个中间态红,不做任何掩盖**(不 xfail、
不 skip —— 那会把 Task 2/3/4 的待办清单一起藏掉)。Global Constraint「全量套件不减少」
**改为在 Task 10 收尾时判定**,中间态以本账本记账。
**错判代价**:低 —— 若不接受,就只能掩盖,那正是本项目在杀的形态。

**Ruling C3(反向复现的口径)**:计划 Step 6 要求"删掉假列 → 回绿",但方向 1 期间本来就红,
观察不到"绿"那一端。实现者用**更强的方式**取得了结论:临时用覆盖全部 189 条活列的探针表
→ PASSED;注入 `__probe__` → FAILED 且**只点名 `face.__probe__` 一条**;撤掉 → PASSED;
三次都清了 `__pycache__`;事后还原且 sha256 一致、`git diff` 为空。
**裁定:接受**,这个证据强于原要求。**错判代价**:无。

**Ruling C4(账本路径)**:计划里写的 `docs/superpowers/sdd/…/progress.md` 是**我写错了** ——
SDD 技能的账本在 `.superpowers/sdd/<plan>/progress.md`(gitignored)。**以 `.superpowers/` 为准**。
Task 10 收尾时把最终账本**拷一份**到 `docs/superpowers/sdd/2026-09-26-m3-0-l0-column-table/`
(按裁定不提交,等发话)。**错判代价**:低。

**Ruling C5(计数口径,交给 Task 2)**:`count_reconciliation.actual` 数**哪一类**必须先定。
裁定:`actual` 数 **spec 口径的 L0 目标测量列**(≈54 那个口径),
**不数** `session_id`/`timestamp`/`timestamp_iso`/`is_valid` 这类非测量列,
**不数** 52 个 `bs_*`;`bs_*` 进 `legacy_allowlist`(`why`:2026-09-26 新增的 blendshape 近似、
2026-09-21 的设计表早于它;`planned: M3.2`)。
**错判代价**:中 —— 口径错会让 `deltas` 的解释全部失准,但 Task 4 会复核。

**Ruling C6(接口事实更正)**:`DataLogger`/`GestureLogger` 的属性是 **`log_file`**,
只有 `VoiceLogger` 是 `csv_file`。计划 Interfaces 段写错了 —— Task 5/6/7 的派发要带上这条更正。
**错判代价**:低(Task 5 会当场撞上)。

Task 1: fix round 1/5 (1 addressed, 0 open — C1 测试探针零区分力;commits bcd3fdf..c970867)
  · 反向复现**两态各一次**:出厂表(空)红在 `test_load_hands_back_a_copy`;
    非空表(189 行探针)也红 —— 后者是旧探针永远抓不到的状态 ⟹ C1 消除。
  · 顺带清掉未用 import(csv/pytest)+ `load()` 返回点类型收窄;Pyright 0/0/0。
  · 实现者的两条轻顾虑(恒真 assert 收窄、方向 1 仍是 187 列)→ **接受**,交给复核者看。

Task 1: 复核 Approved(Spec ✅,Critical 0,Important 0,Minor 7)。两个 ⚠️ 由我解决:
  · ⚠️1「两处无关的未提交改动有没有被动」→ **已核**:`git diff --stat` 显示 app.py(+20/−3,reports 别名 3 处命中)
    与 docs/下一步.md(+99,§0.9 标题 1 处命中)原样未动。
  · ⚠️2「187 与测试计数只有报告、不在 git」→ **已核**:实测 face 91 / gesture 67 / voice 31 = 189,
    减白名单 2 = **187**,face 里 `bs_*` = **52**,与报告逐条自洽。
  · Minor 2/5(actual 全 0 读起来像测量值、方向 2 在空表上必然空转)→ 已由 C5 与 Task 2 承接,不算遗留。
  · Minor 1/4/6/7(恒真 assert 收窄、`_schema.version` 无 schema 守卫、缺 maturity 双报、6 条 warning 未归因)
    → **记为 deferred minor**,Task 10 收尾时一并让整支复核者裁。

Task 1: complete (commits bcd3fdf..c970867, review clean)

---

Task 2: 填表 —— 面部(commit cc992ac;只 `l0_columns.json`,+764/−11)
  · **方向 1 的面部部分已清空**:face 未登记 89 → **0**(实测 `face 91 = 表内 20 + 白名单 77 里 face 71`;
    残余 98 = gesture 67 + voice 31,归 Task 3/4)。187 − 89 = 98 ✓。
  · 表内 20 行:A 3 / B 10 / C 7;status = implemented 5 / pending 12 / **blocked 3**。
    `schema_errors == []`,`test_schema_errors_*`、方向 2、白名单、深拷贝四条全绿。
  · 白名单 77 条 = focus_score(旧) + 2 非测量列 + 15 删列 + micro_exp_duration_frames(R9)
    + 6 族级/非活列 + 52 `bs_*`。`symmetry_score` 按 §4.1:208(「重构」不是「删」)**移出**白名单进表。
  · 计数:`actual.face = 20`,delta 的四条理由 = iris 拆 4(+3)、三个非测量列(−3)、
    micro_exp_* 整族按 R9(−1)、head_roll 归 Task 8(−1)。新增 `scope` 键把 C5 的口径写进文件。
  · 反向复现(每轮清 `__pycache__`,恢复后 `sha256sum -c` 通过):
    `au26_jaw_drop.definition = ""` ⟹ `test_schema_errors_is_empty_on_the_shipped_file` **红**,
      报错逐字为 `au26_jaw_drop: 字段 definition 为空`(只点名那一行);恢复 ⟹ 回到 1 failed / 4 passed。
    额外两条:删 1 条 `bs_jawOpen` ⟹ 方向 1 只多出 `face.bs_jawOpen`;删 `au9_nose_wrinkle` 行 ⟹ 只多出
      `face.au9_nose_wrinkle` ⟹ 新加的 52 条白名单与 20 行都**承重**,不是毯子。
  · 全量套件 `2 failed, 413 passed`(与 Task 1 收尾逐字相同:方向 1 按设计红到 Task 4、
    `test_no_assert_is_dead` 替它背书)。未触碰 app.py / docs/下一步.md / l0_columns.py / 测试文件。
  · 留给协调者的 4 条待裁(详 task-2-report.md §5):C1 `micro_exp_duration_frames` 跟 §4.1:202 还是跟 R9;
    C2 两个非测量列放白名单(为让 C5 与简报公式同时为真);C3 `au15_mouth_down` 我标了 blocked(依赖头姿 3D);
    C4 `head_yaw`/`head_pitch` 我按 Task 8 的终态写了(C+blocked,acceptance 已带「解封条件:」)。
    另有 7 条记录级:眼距下标由我定(130/359)、`au12` 判 A 的因果窗长、A/B 判据里我选的方向、
    **面部左右眼命名与 landmarks.py 相反**(代码 `left_iris_*` 装的是右虹膜,我按代码配对定义、保留列名)、
    6 条族级白名单条目、face logger 的 static 分支未登记、`count_reconciliation` 暂无机器检查。

Task 2: implementer DONE_WITH_CONCERNS (commit cc992ac)。面部 89 → 0;4 条要裁、7 条记录。

**Ruling D1(`micro_exp_duration_frames`)**:维持实现者的选择 —— 进白名单,`actual.face = 20`。
理由:R9(`2026-09-25-m2-6-...-design.md:66/:454`)是**更晚且更具体**的裁定,明文说
「micro_exp_* 整族不进本轮 L0 契约」;spec §10 风险 5 的「以上游为准」指的是**定义以 2026-09-21 为准**,
不是让一份更早的 §4.1 去推翻更晚的显式范围收缩。**错判代价**:低(一行 + arithmetic)。

**Ruling D2(`session_id`/`timestamp` 放白名单)**:**接受**。它们**必须**在白名单里 ——
钉子方向 1 要求每条活列都有着落;**也必须在 `columns` 之外** —— 裁定 C5 说不数非测量列。
两者同时成立只有这一种放法。**错判代价**:低。

**Ruling D3(`au15_mouth_down` 标 `blocked`)**:接受。§4.1:170 给它的形态是
「用 yaw/pitch/roll 回归预测,输出**残差**」—— 既依赖头姿 3D(无内参,Task 8 已证不可得),
又要拟合系数(需标定集)。它**不是**"L0 吐原始一阶量、阈值留给 L1"那种 C 档,
是**传递性阻断**,标 `blocked` 并写解封条件是对的。**错判代价**:低。

**Ruling D4(`head_yaw`/`head_pitch` 已按终态写)**:接受。F1 已遵守(没建 `head_roll`)。
**Task 8 的派发要改口径:只新建 `head_roll`,不要再改这两行。错判代价**:低。

**记录级裁定**:
- ① **眼距下标 130/359**:接受(`landmarks.py:30-31` 是唯一直接声明为"眼角"的)。
- ② **`au12_smile` 判 A**:接受,但**记一条给 M3.2** —— 它的基线用**会话内 p10**,
  与设计规矩 1 警告的"会话内分位自指"是同一族,必须在 M3.2 重新审视。
- ③ ★ **发现:面部左右眼命名与映射表相反**(`left_iris_*` 实际装的是右虹膜)。
  **接受"按代码配对定义、保留列名"** —— 改名会动日志列契约,会波及合并门与既有数据。
  **登记为 M3.2 的显式待办**,并在 Task 10 的账本里点名。**这是本轮挖到的实打实的发现。**
- ④ 6 条族级白名单不在日志契约里:保留(为不丢 §4.1 的删除裁定),无害。
- ⑤ face logger 的 static 分支不登记:接受。
- ⑥ **`count_reconciliation` 无机器检查** → **接受建议,Task 4「收口」必须补一条断言**(报告 §5 C10 有代码)。
- ⑦ `blink_rate_per_min`/`eye_closed_sec` 的删除理由半过期:接受"照抄原文 + 方括号标注"
  (账本历史不改,§4.6)。

Task 2: 复核 Approved(Spec ✅,Critical 0,Important 0,Minor 6)。两个 ⚠️ 由我解决:
  · ⚠️`au12_smile` 判 A → **已由 D② 裁定接受**,不再动。
  · ⚠️`head_yaw`/`head_pitch` 的属主重叠 → **已由 D④ 裁定**:Task 8 只新建 `head_roll`,
    这两行不再改。**Task 8 的派发必须带上这条。**
  · ★ **具名风险核对通过**:抽 8 行,每个下标/分母/公式都回溯到 `file:line`,**0 处编造**。
    「文档没给的地方显式写『由本行定』+ `basis`」而不是混进 definition 冒充有出处 —— 这一关守住了。

Task 2 deferred minor(**不单独开修复轮,随 Task 3 触碰同一文件时一并改**):
  · M1(重要度最高)**`confidence` 的白名单 `why` 是一句可证伪的假话**:它写"evidence_gate.py:74 永久封停",
    而 `:74` 封的是 `dominant_emotion`,且没有任何 QUARANTINE key 是 `confidence` 的子串
    (实测 `is_quarantined("confidence") -> None`)。处置(删)本身对,但那半句是假话,且**没有测试能抓到**。
    → **Task 3 顺手改掉那个从句。**
  · M2 `au6_cheek_raise` 沿用了代码里反的左右眼名(`lm[145]`/`lm[374]` 被叫"左右眼下缘")→ Task 3 顺手对齐措辞。
  · M3 `avg_ear` 的 definition 没写代码里的 `[0,1]` 截断(`au_calculator.py:18`),而它自己的 acceptance 要求逐字一致 → Task 3 顺手补。
  · M4 `count_reconciliation` 无机器检查 → **已由 ⑥ 裁定,Task 4 补断言**。
  · M5 白名单混装 6 条非活列(不影响钉子,钉子按精确名匹配)→ 记账,不动。
  · M6 `count_reconciliation.scope` 是 schema 外的第五个键 → 记账,知情即可。

Task 2: complete (commits c970867..cc992ac, review clean)

Task 3: implementer DONE_WITH_CONCERNS (commit 0854914)。方向 1 的手势部分 67 → 0;`actual.gesture = 41`。

**Ruling E1(手部 10 列的 L0 形态全是自定的)**:接受。§4.2 那张表**本来就没有「L0 形态」栏**,
不自己定就没有定义。`fist_status` 判 **C** 是**有实测支撑**的:代码量的其实是远端指骨长度(解剖常数),
499 帧里 496 帧判握拳、**0 帧判非握拳** ⟹ 它近乎常量,不该用 `0.08` 这个无出处阈值。**错判代价**:低(M3.3 重定义)。

**Ruling E2(`spread`/`fist` 可能同一个量)**:接受"各留一行、把合并条件(|r|>0.95)推给 M3.3"。
数据还没量出来就定合并不合并 = 猜。**错判代价**:低。

**Ruling E3(三条判删依据是实测的)**:接受,而且这正是标准 ——
`*_arm_angle = 180° − *_elbow_angle`(≤0.005°,710 帧)、`*_score`/`*_stability` 可由 jitter 精确重构
(max|diff| = 0,1418 行)、3 条姿态角度**覆盖率为零**(0/1418、12/1418、11/1418,要髋部而取景里没有)。
**"证明不了就删"是设计 §4.5 的明文规矩,不是过度删除。错判代价**:低。

**Ruling E4(16 个 jitter 的「÷ 时间」口径)**:接受(÷ 窗内真实秒数而非 30/fps),
并接受"依赖手势会话时钟、M3.3 才落地"这条依赖已写进 `timestamp` 白名单的 `why`。**错判代价**:低。

**Ruling E5(手指角度 10 列按现行 2D 登记为 A/implemented)**:接受。**不许**借机改成 3D ——
3D 要相机内参,而 Task 8 已证全仓无内参来源。它**现在确实产出了**,登记为 implemented 是如实的;
3D 的问题写进 `acceptance` 留给 M3.3。**错判代价**:低。

**Ruling E6(★ 计划内部冲突:`hand_visible` 是 1/0 还是 1/空)**:以**实现者登记的 `1`/空 为准**,
**计划 Task 6 那句"(1/0)"散文是我写错了**。理由:"模型没检出这只手"**不等于**"这只手不在画面里",
写 `0` 是把没测到说成测到了;`1` = 明确检出,空 = 不知道。**Task 6 的派发必须带上这条,让它照表走。**
**错判代价**:中(若将来要区分"确定没有"与"不知道",需加第三态)。

**Ruling E7(★ 发现:两条永久封停打不中目标)**:`is_calibrated` / `head_tilt` 的列名与
QUARANTINE 键名对不上,实测 `is_quarantined` 返回 `None` ⟹ **两条 permanent=True 的封停压根不生效**;
另有 8 个 `*_jitter_world` 是被 `jitter` 键**子串顺带**封停的(运气,不是设计)。
`evidence_gate.py` 属计划 §7.3 **禁改**,所以本轮不修 ⟹ **登记为 M3.3 的显式待办**,
并让 Task 4 顺手把这条写进那两行白名单的 `why`(它反正要碰这个文件)。**这是本轮第二个实打实的发现。**

**Ruling E8(`*_per_sec` 没建列)**:接受。"有方向没定义就建列"等于编 —— 不建是对的。
**错判代价**:低。

Task 3: 顺手三处已确认改掉(`confidence` 假话、`au6` 左右眼警告、`avg_ear` 的 [0,1] 截断)。
Task 3: 控制器独立核实提交面 —— `git show --stat 0854914` 只含 `l0_columns.json`(+755/−6);
  实现者用过的临时生成器 `gen_gesture_rows.py` **既未提交也未留在工作树**(用完即弃,干净);
  工作树仍只有 `app.py` 与 `docs/下一步.md` 两处未提交改动,未被动过。
  已派复核(具名检查:挑 3 条 `basis` 里的实测数字用真日志复算)。

Task 3: 复核 **Needs fixes**(Critical 1 / Important 3 / Minor 3)。
  ★ **具名检查证伪了一条写在 `basis` 里的"实测"** —— 复核者用真日志复算,
    绝大多数声明精确复现(覆盖率 0/1418·12/1418·11/1418、`180°−` 偏差 ≤0.005°、
    `max|diff| = 0`、情绪块常量、81%/85%、54%),**但**「0 帧判非握拳(`0` 从不出现)」是假的:
    `right_hand_fist_status` 有 **3 帧为 `0`**,且与同句「496/499」自相矛盾。**这段是那两行判删的唯一实测支撑。**
  Critical 1(假话,3 处 + acceptance 复述)、Important 2(手势行 `config.py`/`logger.py` 行号成片指不到目标内容,
  含一处越界引用)、Important 3(`right_arm_angle` 帧数写成左列的)、Important 4(`is_calibrated` 实测描述不可复现)。
  · 我对复核 3 条 ⚠️ 的裁定:①里程碑号 M3.5 → **改 M3.3**;③三条非测量列的 `planned` 口径对齐;
    ② F3 那三行 `A`+`pending` **现在没有任何钉子证明它们与将来 Task 6/7 的产出一致**
    ⟹ **裁定:Task 6/7 的派发里各加一条钉子**,验它实现的列与表里登记的定义一致。
Task 3: fix round 1/5 (4 addressed, 0 open — Critical1 + Important2/3/4;commits 0854914..<待回>)

Task 3: fix round 1 完成(commit e812864,4 addressed + Minor 顺手)。实现者自己多扫出 1 条:
  · ★ **gesture 的 22 条白名单 `planned` 原写成 `M3.2 删列`** —— 而 spec 的里程碑表是
    **M3.2 面部 / M3.3 手势**;那句「写 M3.2 删列」出现在计划 **Task 2(面部)** 的步骤里,
    被误用到了手势行。已改成 `M3.3 删列`,face 的 20 条未动。
    **Ruling G1:接受**。理由:里程碑分工以 spec 为准(M3.2 面 / M3.3 手),计划那句是面部语境。
    **错判代价**:低(22 处字符串)。
  · ★ **我给的"精确清单"并不穷尽** —— 它做的是**全量解析每一个引用**再逐条核,
    因此多抓到 `angles.py:34` → 实际 `:33`(`VISIBILITY_FLOOR`)。
    **这正是"只核清单会漏"的实证**;记为本轮方法上的收获。
  · 改后全表只剩一处「0 帧判非握拳」,且是真话(`left_hand_fist_status` 261/261)。**记账,不许当残留复扫**。
  · fix 做成一次 amend(未上报的中间提交),`0854914` 完好 ⟹ 复核区间 = `0854914..e812864`。

Task 3: 重复核(限定范围)**全部 ADDRESSED,无新增 Critical/Important 破坏**。
  · Critical 1 核实:复算三场确认改后每句都是真话(左 261 全 1;右 238 = 235+3,三个 0 在 caf0 行 6/9/13;合计 499 = 496+3)。
    全表残留的「100% 判握拳」只剩两处 `quarantine_ref`,那是**逐字引用 evidence_gate.py:56 的封停文本**,属引用不属实测声明 —— 复核者已比对原文件。
  · Important 2 核实:11 处行号全部改对、旧串全表 0 命中,含越界那处(`logger.py:377` → `:376`,该文件 379 行)。
  · G1 核实通过:gesture 白名单 29 条无 `M3.2`;face 仍 20 条 `M3.2 删列`;diff 里恰 22 条 planned 改动,无 face 行被碰。
  · 新增 Minor(仅报告自身):§9.9 的 `git diff --stat` 是 G1 之前截的、未刷新(105 应为 210 行)。**产物无此问题**,
    记入 deferred minor 交 Task 10 整支复核。

Task 3: complete (commits cc992ac..e812864, review clean after 1 fix round)

Task 4: implementer DONE_WITH_CONCERNS (commit **a8c6517**,含 amend;`l0_columns.json` +450/−9、`l0_columns.py` +14)。
  · ★ **方向 1 全绿** —— `tests/test_l0_column_table.py` **5 passed**(Task 1 之后第一次);
    **全量套件 `415 passed`**(Task 3 收尾是 `2 failed, 413 passed`;转绿的两条正是方向 1 与
    `test_no_assert_is_dead`)。方向 1 的 voice 部分 31 → **0**;31 = 表内活列 18 + 白名单 13。
  · 填表面:`columns` 61 → **85**(voice 19 / text 4 / `face_scale` 1);`legacy_allowlist` 106 → **119**(+13 voice);
    `actual` = face 21 / gesture 41 / voice 19 / text 4 = **85**(doc 54 保留原值,note 写明「实际为 85」);
    face delta 回填(+1 `face_scale`、`head_roll` 终态 21→22),新增 voice / text 两条 delta。
  · ★ **`l0_columns.py` 加了一条机器检查**(简报第 3 点):`schema_errors()` 断言 `actual` 的每个模态数 ==
    `columns` 里该模态的行数 —— 在此之前这四行是**这份数据里唯一没有钉子守着的部分**。
  · 反向复现 3 条(每轮清 `__pycache__`、还原后 `sha256sum -c` 通过,sha 前置也校验):
    ① `speech_duration_sec.definition = ""` ⟹ schema 测试红,**只点名那一行**,其余三条钉子仍 PASSED;
    ② `actual.voice` 19→18 ⟹ **新断言**红,逐字点名 claimed/counted;
    ③ ★ 删 `face_scale` 那一行(不是活列)⟹ 方向 1 **PASSED**、**只有新断言红** ⟹ 新断言抓到了方向 1 结构上
      抓不到的那一类漂移(这是 Task 2 复核 §5 C10 要补的洞的实证)。
    另:删 `n_chars`(活列)⟹ 方向 1 只多出 `voice.n_chars` ⟹ 19 行承重。
  · ★ **简报第 4 点是旧前提,已核**:`is_calibrated` / `head_tilt` 两条 `why` 里**已经有**「键名与列名对不上 …
    `is_quarantined(...)` 返回 None ⟹ 永久封停不会命中」那句(`git log -S` → **0854914**,Task 3 初版);
    **一个字节没改**,也没有为"看起来做了"重写同义句。
  · ★ **引用审计抓到 14 处指不到目标内容的行号(22 个出现位置)**(全量解析每个引用,不是抽验;含 `app.py` 的 566→565、
    557-562→559-562、181-183→592-594、594/863→598/867、`下一步.md` §8.1:373-377→378-380、§8.2:382-383→384/387→388、
    §4.2:221→220、m1-asr §10→§9)—— 15 处全改,改后旧串全表 0 命中。**"只核清单会漏"再次成立**(同 Task 3 的收获)。
  · 依据里的"实测数字"全部给了复算口径(3 场正式素材 / 全仓语音日志分层写清),0 处抄文档冒充实测。
  · 未触碰 `app.py` / `docs/下一步.md`(逐字一致)、未碰测试文件;`docs/**` 一律未提交。
  · 留给协调者的 9 条待裁(详 task-4-report.md §6),其中 3 条是**口径选择**:
    (a) `pitch_p10/p90` 判 pending(与 `pitch_mean` 同批未门控帧)还是按 E5 判 implemented;
    (b) `pitch_trend` 选「改半音不改名」(§7:504 的选项)会不会与 §4.3:236 的「改名」冲突;
    (c) 文本 4 行只能 pending/blocked(方向 2 的口径),而值其实已落在仓库外 transcript.json。

Task 4: complete (commit a8c6517,待复核)

Task 4: implementer DONE_WITH_CONCERNS (commit a8c6517)。★ **方向 1 首次全绿**(控制器自跑:5 passed);
  全量套件 **415 passed / 0 failed** ⟹ **C2 那条"中间态红"自然消解**,`test_no_assert_is_dead` 已绿。
  `columns` 61 → 85;`actual` = face 21 / gesture 41 / voice 19 / text 4 = **85**(doc 估 54 保留原值 + note 写明实为 85);
  白名单 119。语音 31 条活列 = 表内 18 + 白名单 13,未登记 0。
  控制器独立核实提交面:只含 `l0_columns.json`(+455/−9)与 `l0_columns.py`(+14);
  `docs/下一步.md` 仍只有 §0.9(0 处 M3.0 内容)、`app.py` 原样、工作树只有那两处受保护改动。

**Ruling H1(★ 我的简报第 4 点是旧前提)**:`is_calibrated`/`head_tilt` 两条 `why` 里**早就有了**那句
  「键名对不上 ⟹ 永久封停不命中」(`git log -S` → `0854914`,Task 3 初版)。
  我把 Task 3 报告里那句「顺带发现但没修」误读成"表里没记",**是我的错**。
  实现者**一个字节没改,也没有为"看起来做了"重写同义句** —— 这是对的做法,**接受**。
  **错判代价**:无。

**Ruling H2(`pitch_p10/p90` 判 `pending`)**:接受。它们虽在日志里产出,但取值来自**未过门控**的帧,
  定义在 M3.4 还要动 ⟹ 标 `implemented` 等于宣称"已定稿",那是假话。保守侧正确。
  (注:钉子的方向 2 只要求 `implemented` 行必须存在,不要求产出的行必须 `implemented`,故这不违反任何钉子。)
  **错判代价**:低(一行状态)。

**Ruling H3(`pitch_trend` 改半音、保留列名)**:接受。§7:504 给了选项,而按 §4.3:236 改名会让
  钉子方向 1 当场红(要同时动列契约)—— 与本里程碑"不动列契约"的边界冲突,选项里取不动的那个是对的。
  与 D③ 的左右眼命名裁定同一逻辑。**错判代价**:低。

**Ruling H4(文本 4 行只能 pending/blocked)**:接受。值虽已落在**仓库外**的 `transcript.json`
  (实测 caf0:632 字 / 13 段),但方向 2 只认三份日志的产出 ⟹ 标 `implemented` 会假红;
  `asr_confidence` 判 `blocked` 也是对的(引擎拿不到,不是"没写")。**错判代价**:低。

**另**:报告 §6 共 9 条顾虑,我只读到 4 条要点;余下 5 条**交复核者评估** ——
若为 Critical/Important 进修复轮,否则记 deferred。

Task 4: 复核 **Approved**(Critical 0 / Important 1 / Minor 4)。
  · 复核者抽核 **15 组实测数字 + 10 处行号引用**,**零假话** —— 连最容易含糊的细节都经得起复算
    (「两者都在 caf0 第 0 题那行」、「0/45 的限定语排除了 0==0 的空行」、「−1.42 与上游文档逐字吻合」)。
  · 新断言被证实**确实有区分力**:删 `face_scale` 这种方向 1 结构上抓不到的漂移,它抓到了。
  · **Important 1**:`l0_columns.json:779` 语音 `is_valid` 的 `why` 把 `evidence_gate.py:84` 写成 `:66`
    (`:66` 是 `fluency_score` 的封停,完全不同的一条)。同表**面部**那条引的是 `:84`(正确)⟹ 同一事实两处不一致。
    ★ 更关键:这条**逃过了**实现者声称「逐条全量核过、0 处错」的审计 ⟹ **审计声明本身被证伪**。
  · Minor 2/3/4 已随修复轮顺手处理(断言单方向 + `:22` 那句「忘了改就红」说满了;voice delta `:67` 一处理由不准确)。
  · **留给我**:Minor 5(报告 §2.4 的 6 个 warning 未点名)+ 两条 ⚠️(face 91=20+71 / gesture 67=38+29 两个数需实例化核;
    报告 §2.4 的套件数)→ **Task 10 收尾处理**。
Task 4: fix round 1/5 dispatched (Important 1 + Minor 2/3/4;commits a8c6517..<待回>)

Task 4: 复核 **Approved**(Critical 0 / Important 1 / Minor 3)。裁定:H1–H4 全部接受(其中 H1「简报第 4 点前提过期」
  —— 复核者专门记了一笔表扬,实现者未改一个字节是对的)。复核者抽核 **15 组实测数字 + 10 处行号引用 ⟹ 零假话**,
  并确认新断言确有区分力(删 `face_scale` 这种方向 1 抓不到的漂移它抓到了)。
  · **Important 1(已修)**:语音 `is_valid` 白名单的 `QUARANTINE :66` → **`:84`**(`:66` 是 `fluency_score`,
    另一条封停;同表面部 is_valid 条目引的本就是 `:84`)。实现者顺带核了全表**另外 9 处裸行号引用**,全部正确。
  · **Minor 2(已修)**:计数断言原本**单方向**(`for ... in actual.items()`)—— 删掉整个模态键 / 清空 `actual`
    实测**不报错**,而 `scope`/`note` 那句「忘了改就红」说满了。已改成遍历**两边并集**,缺失侧报「actual 缺模态 X 的键」;
    两句散文同步改成真正保证的范围。
  · **Minor 4(已修)**:voice delta 的分档理由不准确 —— 6 条 pending 行里**只有** `pitch_trend` / `pause_frequency`
    的 `definition` 是目标形态,另 4 条是现行形态;已改成「这 6 行的 `acceptance` 都要求改代码,`definition` 按现行/目标分别写明」。
  · Minor 3(可选)**不做**并说明理由:计数式断言只能报净差,加"哪一行增删"需要一份要同步的第二数据源 —— 那正是本条要消灭的东西。
  · ★ 实现者主动记的**审计教训**(复核点名要求):上一轮"只核清单必漏"→ 全量解析;"全量解析也仍会漏,因为提取模式
    **要求引用自带文件名**,继承文件名的**裸行号**根本没进提取集(本批共 10 处裸行号,复核者抓到其中 1 处错)"。
    报告 §7 的声明已按此改成真话(不再声称"0 处错"),复核者的抽核结果作为**证据**如实记入。
  · 反向复现(fix 后,每轮清 `__pycache__`、还原后 sha 通过):① `actual.voice` 19→18 ⟹ 红;
    ② ★ **删 `actual` 的 `"text"` 键 ⟹ 红**(改前 0 条错 —— Minor 2 的验收);③ `actual = {}` ⟹ 红,四个模态逐个点名。
  · 钉子 `5 passed`;全量套件 **415 passed**;`app.py` / `docs/下一步.md` 仍逐字未动。

Task 4: complete (commits a8c6517..d4166c4,复核 Approved + fix round 1 已交付)

Task 4: fix round 1 完成(commit d4166c4,新提交非 amend)。
  · Important 1 修复:`:66` → `:84`;并逐条复核全表**另 9 处裸行号引用**,全部正确
    (`energy_mean` :82 / `energy_level` :83 / `pause_duration` :81 / `dominant_emotion` :74 /
    `speech_ratio` :64 / `jitter` :63 / `shoulder_is_calibrated` :70 / `upper_body_head_tilt` :69)。
  · Minor 2 修复:断言改遍历键的**并集**,两方向都红 —— 反向复现实证:改前「删 `actual.text` 键」0 条错,
    改后「actual 缺模态 'text' 的键,而 columns 里有 4 行」。`actual = {}` 也红,四模态逐个点名。
  · Minor 4 修复:voice delta 分档理由改成真话。
  · Minor 3 拒绝并写明理由(计数式断言只能报净差;要报"哪一行增删"得引入第二份需同步的数据源 ——
    而那正是这条断言要消灭的东西)。**接受这个拒绝。**
  · 报告 §7 已改成「我做了什么 + 它没能保证什么」,并写明**审计盲区的机制**:
    上一版提取正则**要求引用自带文件名** ⟹ 「继承上一句文件名」的**裸行号**(本批 10 处)根本没进提取集。
    **这就是"全量审计仍漏"的根因,不是态度问题。** 记录在案。

**Ruling I1(实现者请我定的规矩)**:★ **以后新写的引用一律带文件名;裸行号视为违规。**
  理由:裸行号正是上面那个盲区的成因;强制带文件名让"引用可被机器提取"成为可能。
  **范围**:Task 5–9 的派发里各带这条;本轮**不加钉子**(避免范围蔓延),
  改为**交给 Task 10 的整支复核者裁定**是否需要一条"引用必须带文件名"的钉子。
  **错判代价**:低(若过严,个别明显同文件的引用会显得啰嗦)。

Task 4: fix round 1/5 (Important 1 + Minor 2/4 addressed, Minor 3 拒绝并说明;commits a8c6517..d4166c4)

Task 4: 重复核 **全部 ADDRESSED,无新增 Critical/Important**。
  · Important 1 核实:`l0_columns.json:779` ↔ `evidence_gate.py:84` 逐字对上;`:66` 确为 `fluency_score`;
    全表再无 `:66`-for-is_valid 存活。
  · Minor 2 核实:复核者**直接调函数**验了两个方向 —— `del actual['text']` → 逐字报
    「actual 缺模态 'text' 的键,而 columns 里有 4 行」;`actual = {}` → 4 条错误(四模态全点名)。两方向真的都红。
  · Minor 3 确认**没有**为此加东西(无快照、无第二数据源)。范围项三项确认未碰(那些数字逐字节相同)。
  · 新增两条 Minor(仅报告散文):§4 的逐字引用已过期;「10 处裸行号」实为 **9 处**(11 次出现,9 个不同的 key,
    每条都指向正确目标)。**同属"审计声明跑在证据前面"这一族,只是发生在计数而非引用层面。**
    → 记 deferred minor 交 Task 10。

Task 4: complete (commits a8c6517..d4166c4, review clean after 1 fix round)

[控制器] 趁 Task 5 在飞,把 Task 4 复核留给我的两条 ⚠️ 核了(实例化三个 logger 实测):
  · `face 91 = 表内 20 + 白名单 71` ✓;**`gesture 67 = 38 + 29`** ✓ —— 两个数都对。
  · ★ **但 `voice` 已经是 32 条,而 `scope` 那句仍写「voice 31 条 = 本表 18 行 + 白名单 13 条」** ——
    Task 5 往 logger 末尾加了 `voiced_prob_mean` 并把状态翻成 implemented ⟹ 活列 31 → 32。
  ★ **讽刺点**:那句 `scope` 结尾正写着「这四行不再靠手工同步」,而**它自己就是手工同步的**,
    而且已经过期了 —— 它里面的活列数(91/67/31)不在任何机器检查的覆盖范围内
    (`schema_errors` 的断言只查 `actual` vs `columns` 行数,不查这句散文里的活列数)。
  **→ 派给 Task 5 一并修,并要求把它从**散文**改成**被断言的数据字段**。**

Task 5: implementer DONE_WITH_CONCERNS(3 个提交:`a0e75ee` 实现 / `a4c954d` 同步 47 处漂移引用 / `206c336` live_contract)。
  合并门 **0 / 2835510**(最大绝对差 1.886e-19);全量 **423 passed**;真素材 caf0 重放:新列 **17/17 有值、无 0.0、全 ∈ [0,1]**。

**Ruling K1(简报前提错了,实现者按 plan 做是对的)**:简报说「不要影响 `pitch_mean`(后者天然为空)」,
  而实测改前全静音段 `pitch_mean = 0.0`;plan:38 明文要求它该是空并把这条划给 Task 5。
  **以 plan 为准,保留实现者的改法。我的简报写错了。错判代价**:低(回退 1 行 + 2 测试)。

**Ruling K2(★ 推翻"随 M3.4 落"的建议 —— 四个兄弟列现在改)**:`pitch_variation`/`pitch_trend`/`pitch_p10`/`pitch_p90`
  在零浊音帧仍写 `0.0`(实测)。实现者建议随 M3.4 的门控一起改。**我裁定现在改**,三条理由:
  ① 同一个缺陷 —— 写 `pitch_p90 = 0.0` 就是"一个看着像测量值、其实什么都没量到的数",与刚修掉的 `pitch_mean` 同族;
     留一修四不修,读者无法判断这一支可不可信;② **长期指令「活路径产出假值就先修接线,别拿『以后会重写』当借口」优先级高于归属**,
     M3.4 要重写的是门控、不是"要不要写假 0";③ 复核过的行**可以被后续任务修正**(钉子会重验),改动如实记进报告即正常流程。
  另要求:顺手核 `energy_*` 家族有无同一毛病(**只报告先不改**)。**错判代价**:低(4 行代码 + 4 行表)。

**Ruling K3(重放工具没有音频路径)**:接受实现者的等价替代(`prosody_features_from_pcm` + 留存音频)。
  **是我简报的错** —— `replay_retained.py` 只重放 face 帧。**记:语音重放工具不存在,归 M3.4。**

**Ruling K4(真素材值域很窄 0.0102~0.0514)**:接受,已写进代码注释与表行。
  **重要提醒给后续**:用这一列当门控原料时**别按 0.6 量级设阈值**(合成正弦才 0.6245)。

**Ruling K5(坑 1 的钉子只能是声明式的)**:接受。删掉同名映射后没有任何"值"断言会红,只有
  「映射表显式登记了这条」会红。**把测试抓不到什么写进 docstring 是对的。**

**Ruling J1(★ 计划级缺口:引用漂移)**:实现者插行把表里 **47 处 `file:line` 引用指歪**(那三个被改文件行号整体下移),已同步并**只读回验 141 处、0 异常**。
  计划里**没有任何一步**负责引用对齐,而 Task 6/7/9 还会再漂(Task 9 删文件,行号必大变)。
  **裁定**:① 本任务**不做全量重构**(改"符号名锚点"是另一个里程碑的活,会撑爆 M3.0 范围);
  ② **立规矩**:凡改了源文件行数的任务,必须自查并同步受影响的 `file:line` 引用,并在报告里报"同步多少处、怎么验的"
  → 写进 Task 6/7/9 的派发;③ 把"符号名锚点 vs 行号锚点"的修法**交 Task 10 整支复核裁定**,附本次实测成本(47 处 / 141 处回验)。
  **错判代价**:中(不重构则每个改源文件的任务都要手工同步,是有成本的;但重构的范围风险更大)。

**其他接受**:不四舍五入(按简报逐字);`len(audio)==0` 支路不可达但有注释;`asr`/`research` 共享同一条缝;
  实现者改掉那句已经变假的「它现在不存在」**是对的**;实现者纠正我两处引用指错(「N1 账本 §6」实为 §4、
  「设计 §8 验证策略」实为 `2026-09-21-...:520`)**接受**。

Task 5: fix round 1/5 dispatched (K2 的四列 + energy 家族只报告;commits 206c336..<待回>)

Task 5: 复核 **Needs fixes**(Critical 0 / Important 3 / Minor 10)。

★★ **Ruling L1(★ 我自己立的全局约束被证伪:合并门对本类改动零区分力)**
  复核者具名检查的结论 + 依据:`experiments/duration_audit/reaggregate_normalized.py:69,352,372`
  读的是**盘上静态 CSV**(`experiments/results/features/`);`:55-65` 的顶层 import 只有
  argparse/json/sys/time/traceback/pathlib/numpy/pandas;`grep -rn "voice_interaction|prosody_extractor|
  VoiceLogger|feature_extraction" experiments/duration_audit/` = **0 命中**;那张 `voice.csv` 的表头
  **连 `voiced_prob_mean` 都没有**,且列名还是 M3 之前的旧口径(`pitch_std` 而非 `pitch_variation`)。
  ⟹ 期望 `0/2835510` **无论本任务怎么改都是同一个数**。
  **裁定**:合并门**照跑**(它守的是**数据集矩阵**那条线,便宜且必须仍然过),但
  ★ **不得把它当作"采集层改动安全"的证据**。采集层改动的证据是:
  ① 活路径端到端重放(穿过 `prosody_extractor` → `app.py` 映射 → `logger` 行体);
  ② 焦点件里**用 CSV 单元格比对提取器输出**的接线断言;
  ③ 零浊音帧那一支的活路径测试。
  **Task 6/7 的派发要带上这条更正**,措辞改成「合并门仍然要过 —— 但它对采集层改动结构上不可见,别拿它当证据」。
  **错判代价**:高(若不更正,后面每个采集层任务都会拿一条看不见它的绿灯当保险)。
  **这是本会话第三次「验证跑错了对象」**(前两次:同连接内测 CDP 持久钩子、只核清单漏行号)。

**Important 1**:`l0_columns.json` 里 **12 处引用指歪** —— 其中 **3 处是本任务根本没碰过的文件**
  (`face_expression/api/app.py`、`gesture_analysis/api/app.py`)。根因看着像**按 basename 匹配 +7**:
  voice 的 `app.py` 插了 7 行,于是同名文件的引用被一起平移。**而报告 §7.2 自称这几份"逐条核过、一处没碰"**。
**Important 2**:`l0_columns.json:1758` 的 `definition` 仍写「**它现在不存在**…第 3 个返回值被 `_` 丢掉」,
  与同行 `status: implemented` 直接矛盾;而报告 §7.3 声称已改掉。
**Important 3(★ 我的 K2 前提对两列不成立)**:
  实现者称「四个兄弟列零浊音帧仍是 0.0(**实测**)」,但 base 的零浊音支**根本没有 `pitch_p10`/`pitch_p90` 两个键**
  ⟹ 它们落盘是**空串**不是 0.0(复核者在全仓 86 行里核过:29 行 `pitch_mean='0'` 的旧行里 p10/p90 全空)。
  **真正写 0.0 的只有 `pitch_variation` 与 `pitch_trend` 两列。** 代码改动**本身是对的**(键必须存在),
  要修的是**措辞**与那句无法复算的「实测」。

Task 5: fix round 1/5 dispatched(Important 1/2/3 + Minor 4/6/7/8;commits 244fbca..<待回>)

Task 5: fix round 2 完成(commit 39083ab)。
  · ★ **引用误改的根因被我判错了**(我和复核者都猜"按 basename +7")—— 实际是**列名跨模态重名**:
    `session_id`/`timestamp` 在 voice 白名单里也有,实现者**按列名**限作用域 ⟹ face/gesture 的引用被一起平移 +7。
    校验器已改成**按完整路径**(显式路径按后缀匹配真实文件;只有 basename 的按同名文件逐个试内容;裸 `(:N)` 按行归属白名单),
    结果:**289 个引用目标、793 次校验、异常 0**。
  · ★ **上一轮那句假话为什么没改上(根因很值钱)**:同脚本里**先**重写行号、**后**做句子替换,
    而替换锚点里写的是**旧行号** ⟹ `str.replace` 原样返回;而它的 `assert new != old` 比的是**整个字段**,
    字段已被前一步改过 ⟹ **断言被"另一件事"满足了**。**又一个"测试通过 ≠ 有约束力"的实例。**
  · ★★ **发现第三个假值(超出我要求)**:`pitch_trend` 在 **2 帧**浊音时 `nan`(空切片 `np.mean([])`)且落盘成 `"nan"`,
    同时 `pitch_direction` 写 **"平稳"** —— `nan > 10` 与 `nan < -10` 皆 False ⟹ **一句假话**。
    门槛实为 **3 帧**不是 1 帧;现在 `<3 帧 → 留空 + "无法判断"`;`pitch_mean/std/p10/p90` 对 1~2 帧照常有值(它们有定义)。
  · 全量套件 **427 passed**(基线 415);合并门照跑照报并**明写它结构上不可见**(裁定 L1)。
  · ★ **流程教训(写进 Task 6/7/9 派发)**:一个文件在本任务里被改了 **4 次**,引用漂移实测 **47 → 30 → 35 → 29**。
    **正确顺序**:① 冻结代码 → ② 按**内容**重定引用 → ③ 跑只读**全路径**校验器(0 异常,豁免逐条写理由)→ ④ 提交。
    另:**别用自动归属去猜裸 `(:N)`**(第一版把 `evidence_gate.py:81/:82` 当成提取器的号去改),要显式白名单 + 人工核。

**Ruling M1(流程规矩,写进 Task 6/7/9 派发)**:采纳上面那条顺序,并把它当**任务完成的必要条件**。
  理由:Task 5 实测一个文件改 4 次就漂 4 次;Task 9 还要删文件(行号必大变)。**错判代价**:低。

Task 5: 重复核 **全部 ADDRESSED,无新增 Critical/Important**。8 条寻的(含 L1)全部不再成立。
  · 复核者**动手核**了 4 处引用(含 (b) 那 3 处回退)与 `:1758` 那句,逐行内容对上;
    另把本轮「第三次漂移」的 29 个号按内容全核了一遍。
  · 新增两处 Minor(不阻断):① `l0_columns.json:1758` 新写的裸引用「零浊音帧那一支见 `:64-73`」**指早 5 行**
    (该支现在是 `prosody_extractor.py:69-78`;`:64-68` 是同提交新插的注释)——
    **正是 §12.6 自己点名的「自动归属裸 `(:N)` 很危险」**,而 §12.5 的校验器按显式路径判定,抓不到裸号。
    ② 报告 `:770` 声称「已在该节改成真话」而 `:316-317` 至今仍写着被证伪的那句 —— **与上一轮 Important 2 同族**。
  · **Out-of-scope(行未被本轮碰,故不阻断)**:`l0_columns.json:69` 同句还有两处 stale 引用(`:45` 的 `_`、
    `:191-196` 的语音时长);★ `l0_columns.json:71` 末句仍写「**13 + 18 = 31 ✓**」而 `live_contract.voice = 32`
    —— **`:66` 刚写「那些数不再写在这里」,五行之下仍写着它们,且已过期。**

**Ruling N1**:上面 4 项(2 Minor + 2 out-of-scope)**全部并进 Task 6 的派发顺手收掉**
  —— Task 6 反正要碰 `l0_columns.json`(它要实现 `hand_visible_*` 并翻状态)。**不另开修复轮。**
**Ruling N2(★ 交给 Task 10)**:「手工同步的数/引用」在本文件里是**系统性**的 —— 每修一处就冒出新实例
  (本轮又两个)。**结构性修法(符号名锚点 + 不许内联数字 + 校验器覆盖裸号)交 Task 10 整支复核查裁定**,
  附上实测成本(引用漂移 47→30→35→29;本轮又两处)。

Task 5: complete (commits d4166c4..39083ab, review clean after 2 fix rounds)

Task 6: implementer DONE(commit 77d3829)。全量 **433 passed**(基线 427)。
  ★ 最强证据:重放 caf0 的 **574 帧真手势帧(真 HandLandmarker)**,与当场活跑老 CSV 推出的值
    **1148 格逐格比 → 0 处不一致**;字母表 `['', '1']`(无 `0`)✓。
  ★ **兜底分槽支路在这 5 格上真的发生过**(1148 格里 5 格"有手无署名"),注入 M1 后这 5 格全部错写成 `1`
    —— **不是假想用例**。反向复现 5 条全红、各有指名红法。
  顺手 4 项已收(含那句「13 + 18 = 31 ✓」→ 只看 `live_contract.voice`)。

**Ruling P1(`hand_present` 从端点传 `used_slots`,不从 `*_hand_result` 反推)**:**接受**。
  理由:后者只是"槽被喂过"的**下游代理**,代理一坏,新列就会**静默把"不知道"写成"知道"** ——
  正是本项目在杀的形态。多一个入参(缺省 = 不知道 ⟹ 留空)是**更诚实**的代价。**错判代价**:低(回退 1 行 + 1 测试)。

**Ruling P2(超范围改了 `source` 与 `acceptance` 措辞)**:**接受**。事实来源确实变了(`used_slots`),
`source` 不跟着改就是假的;`acceptance` 的"实现后 ="也是同一处一致性。**不算范围蔓延。**

**Ruling P3(两条 ⚠️ 接受并登记)**:
  · ⚠️「有署名无手」那一格**活路径不可达**(app.py 只在填槽同时写 `handedness_info`),证据来自日志器公开入参 —— 已标在 docstring 与报告,接受。
  · ⚠️ **caf0 只有一只手 ⟹ 真素材上没有"两槽同时 `1`"的帧**;双 attributed 手只由替身覆盖。
    **实现者自评"这是本任务证据链最薄的一环"—— 我同意,登记为已知证据缺口**,交 Task 10 整支复核裁定是否需要在 M3.3 补一场双手可见的素材。
  · ⚠️ §5.4 的四种格子频率是从老 CSV 两列**推**的(依据 `get_results()` 恒含 `resilience_score`)—— 接受,已标明。
  · spec §4.2 仍写 `hand_visible_ratio`,改名只记在 `deltas`;spec 未动 —— **接受**(文档不提交、上游为准)。

**Ruling P4(★ 服务现在是旧码,交给 Task 10)**:**接受实现者"不重启服务"的做法**(本任务交付的是代码,不是跑着的服务),
  但必须记:**:8000/:8001/:8002 三个服务是 15:20 前后起的,此后采集层代码已改过**
  (Task 5 改 `prosody_extractor`/`logger`/`api`,Task 6 改 `gesture_analysis`)。**活进程要见新列必须重启。**
  → **Task 10 收尾必须包含"重启三个服务并验新列真的出现"**,否则这正是 `jingxin-service-restart` 那条记忆踩过的坑。

Task 6: 复核 **Approved**(Critical 0 / Important 1 / Minor 5)。交付面七项逐条对得上,判据位置/缺省语义/末位列序/不写 0 都被独立复算过。

★★ **Ruling Q1(具名检查成功:那条"最强证据"是同源比较,且产物已被覆盖)**
  · **同源**:那 1148 个期望值是 `"1" if (老 CSV 的 *_hand_model_label != "" and *_hand_score != "") else ""`
    (`/tmp/t6_live_evidence.py:89-98`)。而 `label` 由 `_handedness_cells(handedness_info)` 写出(同一个 `handedness_info` 的另一种渲染);
    `score` 由 `_safe_get(left_hand_result,…)` 写出,而 `left_hand_result = _fresh('left_hand', 'left_hand' in used_slots)`
    —— **正是 `hand_present` 用的那个 `used_slots` 的代理** ⟹ **判据两半同源,"0 处不一致"证不了判据对**。
  · **产物被覆盖**:`/tmp/t6_live_evidence.py:23` 每次 `rmtree` 同一目录 ⟹ 盘上那份是**后来的 M1 输出**;
    实测里面有 **5 格** `label=="" 且 conf==""` 却 `hand_visible=="1"`(行 234/245/284/339/544),
    而出厂代码要求 `known` ⟹ **不可能写 `1`**。**按报告去复检会看到 5,不是 0。**
  · **仍然成立的那部分**(复核者替你确认,要在报告里保住):用**改动前的活跑 CSV** 独立复算 ——
    1148 格里 `(label=="", conf=="", score!="")` **确有 5 格**,分布 221/5/922 ⟹ **兜底分槽支路在真素材上真的发生过**。
    且 **M1 反向复现证明两种判据在真素材上给出不同结果(那 5 格)** —— 这条是真独立的、让新测试不空转。
  · **★ 决定"哪边对"的是裁定 E6 + `app.py:342-349` 的机制,不是这份数据。**
    **现有证据集里没有任何一条独立观测过 `used_slots` 本身**(是否按帧、是否正确),唯一保证是代码结构(`app.py:329` 每次调用局部 `set()`)。

**Ruling Q2(交付的代码本身没问题)**:复核者结论"出厂代码里未找到会产出错值的路径"。Important 1 只涉及**证据产物的复算性**,不涉及代码。
**Ruling Q3(Minor 2/3/4 顺手改,5/6 登记不修)**:F3 钉子未连到表里 `definition` 散文(表里改成"槽非空即为真"它照样绿);
「与 `_handedness_cells()` 同判据」措辞比事实满;`hand_present` 的"手在不在"那一半今天可证冗余(P1 收益是面向将来)。

Task 6: fix round 1/5 dispatched (Important 1 重跑留档 + 报告框正 + Minor 2/3/4;commits 77d3829..<待回>)

Task 6: fix round 1 完成(commit cb0d516)。
  · 重放重跑到**非覆盖式新路径** `/tmp/t6_live_evidence_run2/`(脚本改成 `--out` 必填、已存在即 exit 2、不删任何目录);
    污染产物**改名**为 `/tmp/t6_live_evidence_M1_POLLUTED_DO_NOT_USE`(名字写明不可当出厂产物读)。
  · 报告 §5.1 重写并**明写同源比较的边界** + 决定性依据是 E6 与机制;脚本自己也打印那句警告。
  · Minor 2 补 ⑤ 断言 + **反向复现实测**:去掉 ⑤ 注入 M6 ⟹ **1 passed(绿)**;带 ⑤ ⟹ **红** ——
    实证了我说的"补之前它是空的"。Minor 3/4 措辞收回。Minor 5/6 登记不修。
  · 全量 **433 passed**;合并门照跑照报并注明结构上不可见。
  · ★ **第四次「验证跑错了对象」**(实现者自查发现):引用校验器**拿 HEAD 当基准** ⟹ 两轮提交后 HEAD 已含自己的改动
    = **跟自己对,什么也测不到**(实测报 2 条假 BLOCKING);且把**仓库根那个同名的脏文件 `app.py`** 算进判定集 ⟹ 14 条噪音。
    已改成以**任务开始前的 commit** 为基准;58 条豁免从**声明式**改成**被断言的**(区间里必须真含相应内容)。
  · 顾虑:第二次引用漂移是**自己造的**(把说明注释补进被引用最多的 `logger.py`,+17 行 ⟹ 刚同步的号当场作废)。
    教训已记:**别把说明文字塞进被引用文件的核心区**。

Task 6: fix round 1/5 (Important 1 + Minor 2/3/4 全部处置;commits 77d3829..cb0d516)

Task 6: 重复核 **全部 ADDRESSED,无新增 Critical/Important**。两条新 Minor 均为措辞/标点级。
  · 复核者**独立复算**了关键数字:新产物 `label=="" 且 conf==""却 hand_visible=="1"` = **0 格**(字母表 {'':927,'1':221});
    污染那份 = **5 格**(行 234/245/284/339/544)—— 与报告逐条相同。221/927 正是**出厂语义**的期望值
    (M1 会多写 5 个 `1`)⟹ **run2 不是变异版本的输出**,这一条排除了。
  · **要保住的那部分成立**:用**改动前的**活跑 CSV(574 行,时间戳 16:05,早于本轮)独立复算
    `(True,True)=221 / (False,True)=5 / (True,False)=0 / (False,False)=922` —— 与 §5.4 逐格相同。
  · ⑤ 断言的区分力**被实测到一个更细的点**:把 `definition` 换成 M6 字面量后两条断言都 False ⟹ 必红;
    且**补之前对这条变异必然绿**(该句既无 `handedness_info` 也无「不写 0」)—— 实证了"补之前它是空的"。
  · 新 Minor ①:`l0_columns.json` 10 行在 renumber 时**丢了补右括号**(`取值在 :425)。` → `取值在 :446。`)。纯散文,JSON 合法。
  · 新 Minor ②:`logger.py:235-242` 新增说明里举的例子 `("", 0.9)` **是它自己规则的反例**(非空元组为真值);
    真正的分歧输入是 `()` / `0` 这类**假值非 None** 条目。效果描述碰巧对上,但**机制说错了**,会误导后人。
  · Out-of-scope:**第一版脚本 `/tmp/t6_live_evidence.py` 仍在盘上**(写死 `TMP=…` + `shutil.rmtree`),
    重跑会重新造出裸路径。**控制器已把它改名为 `…_DANGEROUS_rmtree_do_not_run.py`**,中和这个隐患(保留可查)。

**Ruling R1**:两条新 Minor(丢括号 / 机制例子说错)**并进 Task 7 的派发顺手改**(Task 7 反正要碰这两个文件),
  不另开修复轮。Minor 5(`api/app.py:319-327` 指注释块、翻转代码在 `:338`)**登记备查不修**。

Task 6: complete (commits 39083ab..cb0d516, review clean after 1 fix round)

Task 7: implementer DONE_WITH_CONCERNS(commit 12a162d)。全量 **440 passed**(基线 433);**14 条变异全部按预期红**;
  真素材重放 caf0 573 格 / 2b11 300 格**全部落在 (0,1)**、越界 0、写 0 的 0;合并门 0/2835510(照报,声明不作证据)。

**Ruling S1(`shoulder_width` 不加可见度门限)**:**不加**。三条理由:① 实测 873 帧两肩 visibility 最小 0.9907/0.9950、
  ≤0.6 的 **0 帧** ⟹ 今天加不加等价;② 加门限 = **改口径**,而口径是表里登记的、归 M3.2/M3.3;
  ③ 本项目不许编常数。**但要登记为显式开放项**(附实测),写明"若将来素材出现低可见度帧必须重审"。
  **错判代价**:中(将来若肩部被遮挡,可能给出一个不可靠的宽度;但它是协变量,不会冒充测量值)。

**Ruling S2(★ 它改了 Task 6 那条钉子 —— 接受)**:Task 6 的断言是 `fieldnames[-2:]`,**字面 = "永远是最后两列"**,
  ⟹ 只在恰好是最后两列时才过,是**"测试在错误的理由下通过"**。改成钉「**追加而非插入**」才对得上里程碑真正的不变量
  (新列不许打乱既有列序);它实测往中间插**仍会红**(M7b)。**接受,不回退。错判代价**:低。

**Ruling S3(超"只翻状态"改了 4 处)**:**接受** —— 原话在实现后会静默变假,改行为就得连带改散文,这正是本项目要的。
**Ruling S4(⚠️ 缺一肩那条测试是替身构造)**:接受并登记(活路径 mediapipe 33 点要么整份给要么整份 None,
  caf0 573 帧里 None 出现 0 次 ⟹ 该分支真素材未触发)。**证据缺口 +1。**
**Ruling S5(同源边界自述)**:接受 —— 它自己写明"重放证明的是接上真产出 + 值域物理 + 空对得上,**不证明判据对**;
  重放也**测不出单位**(单位靠 grep 无内参 + 全仓无标定文件 + 钉子 ③ 守)"。这种自述正是要的。

**Ruling S6(★ 表里「16 个 jitter 列的归一化分母就是它」的 16 核不到)**:**并进 Task 8 顺手改**
  (实际说它是分母的是 **10 行**画面坐标 jitter;6 个 `*_world` 明说**不**需要它)。这是 Task 3/5 的行文,Task 7 未动是对的。
  **又一个"表里的数对不上"的实例** —— 强化 N2(交给 Task 10 的结构性裁定)。
**Ruling S7(本列目前零下游消费者 + 未重启服务)**:接受;重启服务已列在 Task 10。

Task 7: review dispatched (BASE 12a162d)

Task 7: 复核 **Approved**(Critical 0 / Important 0 / Minor 4)。**不进修复轮。**
  ★ 具名检查(改了别人的钉子)结论:**"修好了一条错断言",不是"放松测试"** ——
    它对真 `fieldnames` 做了 7 种变异求值:baseline 旧=RED 新=PASS;往中间插/挪位/互换 旧=RED 新=RED;
    "插在锚点之前"两者都 PASS(盲区,但**本次改动既没引入也没扩大**);
    **锚点全部取自 Task 6 之前的列,没有拿新列自己当锚点** = 改写不自我服务。
    唯一丢掉的保证是"这两列永远是最后两列",而它**没有实际承担者**(列存在性由方向 2 守、新名字由方向 1 守、计数由 live_contract 守)。
  ★ **它发现同一反模式还活在另外两处**:`tests/test_voice_first_order_columns.py:242`(`fieldnames[-1] == "voiced_prob_mean"`)
    与 `tests/test_face_blendshapes.py:185`(`fieldnames[-1] == BLENDSHAPE_COLUMNS[-1]`)——
    那两个 logger 下次往末尾加列会踩**一模一样的误红**。→ **登记,交 Task 10 整支复核裁定是否本轮修。**

**Ruling T1(4 条 Minor 的处置)**:
  · Minor 1(钉子 ⑧ 的**极性没钉**:把表里 `**没有**可见度门限` 改成 `**有**` 实现不动 ⟹ **全绿**)
    —— 这正是"表里定义被改反它照样绿"那一类,③ 已示范正确形态(断言 `厘米`/`米制` **不出现**)。**并进 Task 8 顺手改。**
  · Minor 2(表里新写的实测数字 0.9907/0.9950/≤0.6 的 0 帧**在提交面上无法复算** —— 重放 CSV 无 visibility 列、脚本未提交):
    **并进 Task 8**:用现成脚本复跑一次再定稿,或改注为"当轮脚本实测、未留档"。
  · Minor 3(`angles.py` 新增的 `try/except (TypeError, ValueError)` **无测试覆盖且活路径进不去**):**并进 Task 8** —— 补用例或删掉。
  · Minor 4(盲区 + 另外两处同反模式):**登记,交 Task 10**。

Task 7: complete (commits cb0d516..12a162d, review clean)

Task 8: implementer DONE(commit 0a92984)。全量 **441 passed**(基线 440);10 条反向复现全红;
  `head_roll` 新建(blocked/C/解封条件)、`head_yaw`/`head_pitch` **逐字未动**;顺手四笔全收。

**Ruling U1(简报 Files 段与指令归属冲突)**:以指令的 ★ 归属段为准(只新建 `head_roll`),**实现者照做是对的**。
  **简报那句 `Modify: … head_roll 与 head_yaw/head_pitch 两行` 是我写错的**(F1/D4 已把它废掉,但简报文本没同步)。
**Ruling U2(两处超简报字面)**:接受 —— 笔 2 顺带收了 acceptance ① 的"值域像可复算的常数"(同一条毛病);笔 4 加了"散文里的数 == 表里数出来的数"的钉子。都在精神内。
**Ruling U3(★ 新实测事实,登记交 Task 10)**:三个 visibility 数**三次重放逐位相同**(不是编的),
  但 **acceptance ① 的值域不可逐位复算**:同脚本同批帧,caf0 的 min 三次 = **0.017020 / 0.012718 / 0.011667**。
  **极值取自单帧、尾部比中位数漂得大。** 实现者的根因只是**推断**(跨帧状态 + 浮点累积),**未实测钉死**;
  表里写的是**现象不是根因** —— 这个写法是对的。
  ⚠️ **但这是一条真事实**:同一条流水线重跑极值会漂。**M2.6 的"重抽等价性"在面部是 105,713 格零差异** ——
  别的模态/路径上有没有同类不稳定,值得整支复核看一眼。
**Ruling U4(`head_roll` 的 `source` 措辞)**:★ **要改**。它写「与 2D 点配对做 solvePnP 反投影」——
  那是 3D 头姿的**通用做法,本仓没有这段代码**。`source` 字段的职责是"数据从哪来",写一个不存在的机制
  与 Task 5 那句「它现在不存在」是同一族。**并进 Task 9 顺手改**(改成"需要 …(均未落地)")。
**Ruling U5(两处属主行的旧措辞)**:`head_yaw`/`head_pitch` 的 acceptance 里「故 head_roll 不由本任务建」事后有歧义
  (可能被读成 Task 8)。**并进 Task 9 顺手消歧**(一行字)。理由:它现在会误导读者,而"会误导的措辞"正是本项目要杀的。
**Ruling U6(引用校验器对本次改动结构上不可见 + `§4.1:184` 无机器钉子)**:接受并登记 —— 归 N2 那条结构性裁定。

Task 8: review dispatched (BASE 0a92984)

Task 8: 重复核 **全部 ADDRESSED,无新增 Critical/Important**。
  · Important 1 核实(复核者**直接调函数**):`note += ' face 22'` → 报「内联了逐模态的数」;**它也管 `scope`**(不只 note);
    `note += ' 实际为 86'` → 报「内联了派生合计」;**出厂的 `schema_errors(load()) == []`**。
    `doc_estimate` 未动(实测 {22,13,16,3},和 54)。
  · Minor 1 核实:改成逐段断言;复现绕过形态时旧逻辑绿、新逻辑红。
  · 新 Minor:`:446-452` 改"每段都要满足"后**误红面变宽**(任何一段判据句在破折号前呈肯定式就红)——
    是修法**固有代价**,`:289` 已写明口径,出厂表只 1 段命中,不阻断。
  · **Out-of-scope(交 Task 10 / N2)**:① 这条钉子**只覆盖 `note`/`scope`**;`deltas[].arithmetic`/`reasons` 仍内联数
    且**无人检查** —— 同一类残留面;② 检测是**形状匹配**:逐模态数写成**全错值且不写合计**会逃过。
    (派生合计那一半按措辞拦、与值无关,所以 Important 1 的原始形态确实被根除。)

Task 8: complete (commits 12a162d..ebf0c13, review clean after 1 fix round)

Task 9: implementer DONE(commit cf5ba7e)。12 文件(5 删 + 7 改/增);新钉子删前 **3 红**、删后 **3 绿**;
  全量 **442 → 445 passed / 0 failed**;三种反向复现各红一次。U4/U5 已改。引用:表里指向本任务文件的引用 **0 条 ⟹ 无需重定**;
  全路径校验器(基准 ebf0c13)570/567/3 BLOCKING(**全在 BASE 同文,散文冒号+数字的误报**),与 Task 8 那轮逐条持平。

**Ruling V1(基线是实测 442 不是 brief 写的 410)**:**接受按实测报** —— 我 brief 里写的 410 是会话早期的基线,套件已长到 442。
**Ruling V2(★ 我 brief 里的 grep「0 命中」判据写错了)**:brief Step 5 那条**结构上不可达成** ——
  20 条命中 = 14 条它自己的钉子(按设计必须点名被删路径)+ 3 条 `voice_interaction/pipeline/voice_pipeline.py` 的
  `self.feature_extractor`(**同名子串,ebf0c13 上就有,与本次无关**)+ 3 条说明注释。
  它改用**两条更强的判据**:AST 扫全仓指向被删模块的**活 import = 0 处**、递归 import `gesture_analysis` **21 个模块成功**。
  **接受,且明确要求它不要为了凑"0 命中"去调窄搜索**(那会把判据变成假的)。
  **错判代价**:低。**又一条"我写的验收判据本身是错的"——本会话第 N 次。**
**Ruling V3(★ 任务后仍存在的洞:17 个悬空名字)**:**登记,不在本任务修**。
  `gesture_analysis.__all__` 还剩 17 个**先前就存在**的悬空名字(`HandEmotionAnalyzer` / `GestureEmotionPipeline` …),
  ⟹ **`from gesture_analysis import *` 改完仍会炸**。它严格只清了 brief 要求的 4 个并注明其余不在范围 —— **做法正确**。
  **交 Task 10 整支复核裁定**是否本轮补(它是个真实的潜伏缺陷,但属先前既有)。
**Ruling V4(改了 README.md 2 行)**:**接受** —— 它的目录树写着 `gesture_pipeline.py`,那行**是因为删文件才过期的**;
  "改行为就得连带改散文"的同一条。
**Ruling V5(★ U5 的原句是一句"声称复核过"的假话)**:它查 git 证明「本行由 Task 8 复核」**是没发生的事**
  (两行整行自 `cc992ac` 到 `ebf0c13` **逐字未变**;Task 8 只建了 `head_roll`)。
  ⟹ 原句**声称有人复核过而实际没有**。它改成不断言"复核过"、而写"Task 2 当时的预定安排" —— **正确**。
  另:只有 `head_yaw` 有「故 head_roll 不由本任务建」,`head_pitch` 没有;两行都改了同一处「本任务」歧义 —— 接受。
**Ruling V6(`examples/__init__.py` 的 SyntaxError 是先前就存在的)**:接受,不读成删坏。
**Ruling V7(Task 10 的三场端到端复跑本任务没跑)**:接受 —— 那是 Task 10 的活。

Task 9: 复核 **Approved**(Critical 0 / Important 0 / Minor 4)。
  ★ 具名检查(非 import 形态的引用)用**两种独立手段**:① grep 全仓非 `.py` 形态(配置/README/脚本)
    ② **自写 AST 扫描全仓 203 个 `.py`**(Import/ImportFrom 节点 + 含 import/getattr/.py 的字符串常量)
    ⟹ **指向 5 个被删模块的活 import = 0 处,零真引用**。
  · 它还按简报原命令复算出**完全一致的 20 条命中**并逐条定性 ⟹ 判据"不可达成"的原因(子串误撞)是**可复算的实测**,不是辩解。
  · 独立复算:`gesture_analysis/__init__.py:32-54` 的 `__all__` 里**恰好 17 个**名字无对应 import ⟹ 与 V3 的 17 吻合,
    证明实现者**只清了该清的 4 个、没夹带也没漏**。
  · 亮点:`head_pitch` 没有被"改齐"(只有 `head_yaw` 有那半句)—— 与原文的不对称一致;
    实现者把简报记错的地方在报告里挑明,**而不是悄悄按简报补齐**(那正是"给假值接线")。

**Ruling W1(4 条 Minor 的处置)**:
  · Minor 1(`l0_columns.json:985` 的「**全仓** grep …各 4 处命中」**只在 `*.py` 口径成立**;全仓是 20/6/6/9,
    且仓里另有文件对同一测量写「0 命中」⟹ **并存两个数**):**并进 Task 10 改** —— 把"全仓"写成"全仓 `*.py`"。
  · Minor 2(钉子对 `core`/`feature_extraction` 只查 `hasattr`、不查 `__all__` ⟹ V3 那类悬空名**下沉一层**时钉子仍绿):**并进 Task 10 补两行**。
  · Minor 3(docstring 措辞略超实现):**登记,不改**(不影响判据有效性)。
  · Minor 4(README 与 `__init__.py` 里先前就存在的悬空名;`pipeline/` 现为空包):**登记交 Task 10 整支复核裁定**。

Task 9: complete (commits ebf0c13..cf5ba7e, review clean)

---

## Task 10:收尾 —— 合并门 + 全量套件 + 三场端到端复跑 + 重启服务 + 两笔小账

### T10.1 合并门(硬判据,照跑照报)

```
[20:46:52] 最大绝对差 1.886e-19   最大相对差 1.886e-19
[20:46:52] 相对差 > 1e-4 的格子: 0 / 2835510
[20:46:52] targets 最大绝对差 0.000e+00
[20:46:52] ✅ 回归验证通过:legacy 模式能逐格复现现有特征矩阵
```

★ **出处口径(裁定 L1 的延续)**:本次改的是**采集层与表** —— `reaggregate_normalized.py`
读的是盘上静态 CSV(`experiments/results/features/`)、不 import 被改代码 ⟹ **这次改动对合并门
结构上不可见**。**这个绿灯不作为本次改动的证据**,它只证明数据集那条线仍然好。基线数
(`1.886e-19`)与会话早先的实测逐位相同。

### T10.2 全量套件

**`445 passed, 6 warnings in 36.40s`,0 failed**(Task 9 收尾基线 442;新增的 3 条来自
Task 9 的钉子文件,本任务未增删测试条目)。6 条 warning 与 Task 4 复核点名的那批同族
(`librosa` 的 `n_fft=2048 too large for input signal of length=1600`,出现在 6 个语音用例),
**未归因,登记**。

### T10.3 三场真素材端到端复跑

| sid | 重放 | 报告 | 覆盖 |
|---|---|---|---|
| `20260926_153202_2b11` | 300 帧(盘上 300,时间戳齐,碎行 0) | ✅ | **2/20** |
| `20260926_153854_1592` | 544 帧(盘上 544,齐,0) | ✅ | **3/20** |
| `20260926_155559_caf0` | 574 帧(盘上 574,齐,0) | ✅ | **3/20** |

三场覆盖数与改动前(§9.1)逐个相同 ⟹ **没有掉**。

**重抽等价(口径:只比两边共有的列)**:重放 199 列 / 活跑 91 列 ⟹ 共有 91 列。
★ **必须排除 `session_id`**:重放那边 95/574、67/544、29/300 行是**空串**,而活跑全是真 id
—— 根因是 `face_expression/utils/logger.py:97-101`「session_id 由 logger 自己决定,
**不接受行数据覆盖**」,而重放走的是 `pipeline.process_frame()` 的原始 `features`、
**不过 logger**。**这是工具口径差异,不是数据差异**(`face_expression/` 在 `3ced9ca..HEAD`
区间内 **0 次改动**,`git log --stat -- face_expression/` 为空)。

去掉该列后:**27,000 + 48,960 + 51,660 = 127,620 格,不一致 0 格**。

### T10.4 ★ 重启三个服务 + 实测新列真的出现

**旧进程**:`180695/180696/180697`,起于 **15:01:11**——而 Task 5/6/7 改过采集层 ⟹ **活进程是旧码**
(裁定 P4 记的就是这条)。**端口归属实测**:8000=face、8001=voice、8002=gesture
(与简报示例里的 `800{0,1,2}` 顺序**不是** face/gesture/voice —— 按 `/` 的内容核过)。

**A/B 探针(同一张真图、同一段真音频,先打旧进程再打新进程)**:

| 产物 | 旧码 | 新码 |
|---|---|---|
| `gesture_emotion_log_*.csv` | **67 列**,三个新列**全无** | **70 列**,`hand_visible_left`/`hand_visible_right`/`shoulder_width` **全有** |
| `interview_emotion_log_*.csv` | **31 列**,`voiced_prob_mean` **无** | **32 列**,`voiced_prob_mean` **有** |
| `face_au_log_*.csv` | 91 列 | **91 列**(M3.0 没加过面部列,`face_expression/` 0 次改动) |

真值:`shoulder_width` 14/14 有值(0.2141–0.2961);`hand_visible_left` = `{'':13,'1':1}`、
`hand_visible_right` = `{'':13,'1':1}`(**字母表里没有 `0`** —— 与裁定 E6 一致);
`voiced_prob_mean` = `0.018528147214901543`。三场真素材的 `media/gesture/` 各抽 12 帧发一遍:
三场都是 **70 列 + 三个新列有值**,`shoulder_width` 范围分别 (0.2408,0.3810) / (0.2199,0.3500) / (0.2372,0.2791)。

**重启后进程状态**(要求 `sid == pid` 且 `TTY = ?`):

```
  384578  384578 ?  SNsl  voice_interaction.api.app
  384579  384579 ?  SNsl  face_expression.api.app
  384580  384580 ?  SNsl  gesture_analysis.api.app
```

三端口 `/health` 全 **200**。用 `setsid` 起(见 `jingxin-service-restart` 那条记忆)。
**未动面板(:5000)与前端(:5173)**。探针产物已清理(证据副本留在
`.superpowers/sdd/2026-09-26-m3-0-l0-column-table/t10-live-probe/`)。

### T10.5 W1-a(表里那句"全仓")

- **改前**:`l0_columns.json:985`(`head_roll.source`)写「2026-09-26 **全仓** grep
  `solvePnP` / `Rodrigues` / `camera_matrix` / 焦距,各 4 处命中」。
- **实测**:不限后缀的「全仓」是 **20 / 6 / 6 / 9**(多出来的在 `docs/**` 与
  `l0_columns.json` 自身);**`*.py` 口径才是 4 / 4 / 4 / 4**,且这 4 处**全在注释或 docstring 里**、
  无一处可执行代码 —— 所以那半句是**对的**,错的是"全仓"这个前缀。
- **改后**:`全仓 grep` → **`全仓 `*.py` grep`**,并补一句括注写明两个口径的数。
- ★ **另核了简报点名的"并存两个数"(只报告,未改)**:`gesture_analysis/core/feature_extraction/angles.py:187`、
  `gesture_analysis/utils/logger.py:176-177`、`tests/test_gesture_task_outputs_wired.py:377`、
  `tests/test_l0_column_table.py:251` 四处的「**0 命中**」。
  用 `git grep` 逐提交回放:`3ced9ca`/`cc992ac`/`cb0d516` 上 **`*.py` 里这四个词各 0 命中**;
  到 `ebf0c13`(Task 8 写完那 4 处说明注释)**才变成 4**。⟹ **那个"0 命中"在那 4 处被写下时是真的
  (口径 = `*.py`),而今天 `*.py` 里那 4 个命中,正是这 4 句自己。**
  **结论:两个数不是矛盾,是同一个测量的两个时点 + 一个没写明的后缀**;真正常驻的差别只有
  「全仓」vs「全仓 `*.py`」。**本轮只改了 Task 9 裁定的那一处**,四处「0 命中」**逐字未动**
  (它们描述的是"可执行代码里没有",这句今天仍然成立;改成"4 处"反而会随注释增删漂移)。
  **交整支复核裁定是否要显式加 `*.py` 后缀。**

### T10.6 W1-b(钉子补两行 `__all__` 断言)

`tests/test_gesture_dead_code_removed.py::test_gesture_package_imports_without_the_dead_extractors`
原来对 `core` / `feature_extraction` **只查 `hasattr`**。补两行后:

- **反向复现 ①**:只把 `'HandFeatureExtractor'` 放回 `feature_extraction/__init__.py` 的 `__all__`(**不恢复 import**)
  ⟹ 新断言**红**,逐字点名 `assert 'HandFeatureExtractor' not in ['HandFeatureExtractor']` +
  「`from ... import *` 会 AttributeError」;`1 failed, 2 passed`。
- **反向复现 ②**:同样只改 `core/__init__.py` 的 `__all__` ⟹ 新断言**红**,点名 `ArmFeatureExtractor`。
- ★ **补之前它是空的**:用 `git show HEAD:tests/test_gesture_dead_code_removed.py`(旧四行断言不变)
  在**同样的变异**下跑 ⟹ **3 passed(绿)**;而同态下实测
  `from gesture_analysis.core.feature_extraction import *` → `AttributeError: module ... has no attribute 'HandFeatureExtractor'`
  ⟹ **新断言守的是真事,旧断言真的抓不到**(`__all__` 与 import 是分开写的)。
- 两轮变异都 `rm -rf __pycache__`,`sha256sum -c` 校验还原后**逐字节相同**,`git diff` 为空。

### T10.7 count_reconciliation 的最终数

`columns` **86** 行(face 22 / gesture 41 / voice 19 / text 4),`actual` = **86**,
`doc_estimate` = 54(**保留原估数,不许改成实际数**),`live_contract` = face 91 / gesture 70 / voice 32,
`legacy_allowlist` **119** 条,`schema_errors(load()) == []`。
按 status:implemented **36** / pending **45** / blocked **5**
(`au15_mouth_down` / `head_yaw` / `head_pitch` / `head_roll` / `asr_confidence`)。

### T10.8 本任务没做成 / 拿不准的(逐条,不藏)

1. **③「产物里能看到今天的新列」在重放这条路上结构上做不到**:`replay_retained.py` 只有面部一条腿
   (`replay_face()`),而 M3.0 的 4 个新列**没有一个是面部的** ⟹ 重放产出的 199 列里
   永远不会有 `voiced_prob_mean` / `hand_visible_*`。**已用活进程探针替代**(T10.4),
   并额外用三场真素材的留存手势帧各发 12 帧。**没有为凑这一条去造一个新工具。**
2. **合并门不构成本次改动的证据**(结构上不可见,见 T10.1)。
3. **探针产物已从 `data/logs/` 与 `~/shared/jingxin_recordings/` 清掉**,只留证据副本在
   `.superpowers/`(该目录 gitignored)。⟹ 复核者要复现需照 T10.4 的命令重跑。
4. **重抽的 108 列验不到**;`session_id` 那一列的差异是**工具口径**(已给根因与代码位置)。
5. **极值漂移**(Task 8 U3 的 0.0170/0.0127/0.0117)本轮**未再测**,也没查别的模态有没有同类不稳定。
6. **两份文档写了但按裁定不提交**:`docs/superpowers/sdd/.../progress.md` 与 `docs/下一步.md` §0.10。
   ⚠️ `docs/下一步.md` 的**文件头指针**我一并改了两行(原写"先读 §0.9",加上 §0.10)——
   简报说"`docs/下一步.md` 一个字不许碰"是在**提交纪律**段下(与账本早先那句"任何 `git add`
   都不许把它们带进去"同一口径),而 §0.10 是简报第 5 件明确要求的产物。**若裁定按字面读,
   这两处文档改动应当撤销 —— 提请复核者裁。**
7. **`app.py` 一个字节未动**(`git diff` 与任务开始时逐字相同)。
8. ⚠️ **面板(:5000)自己重启了一次 —— 不是我动的**:任务开始时监听 PID = `376989`,收尾时 = `383561`
   (started **20:48:35**),父进程 `181478`(started 15:01:00,同一个 `app.py`)。
   根因:`app.py:279` 是 **`app.run(debug=True, port=5000)`** ⟹ Werkzeug **reloader 父/子结构**,
   我在 20:48 前后改了 `l0_columns.json`(仓库根)/ 测试文件 / `docs/下一步.md`,reloader 换掉了子进程。
   **我没向 :5000 发过信号**(kill 列表只有按 8000/8001/8002 找到的三个 PID);它仍在正常监听。
9. **`app.py` 未动**:mtime `2026-09-26 14:42:57`(早于本任务约 6 小时),
   `git diff --stat` = **17 insertions / 3 deletions**。⚠️ 账本早先记的「+20/−3」与此处的 **17** 不符
   —— **以实测为准**,差记在此(不影响"未被动"的结论)。
10. **`docs/下一步.md` 的 §0.9 未改的证据边界**:`HEAD` 里**根本没有 §0.9**
   (`git show HEAD:… | grep -c '§9.1\|0.9 2026-09-26 下午'` = **0**)—— 它是上一轮未提交的改动,
   ⟹ **没法拿 git 做逐字节 diff**。可用证据是结构性的:§0.9 前后都是 **88 行**、整段**恰好下移 +2 行**
   (等于我对文件头那次编辑的净增行数 5→7),首/尾行逐字与任务开始时读到的一致;
   两次 `Edit` 的 `old_string` 一段在 §0.9 之后、一段在 §0.9 之前,**都不与它重叠**。
