# M1.5 最终审查 · 修复波次 1 报告

- 出处:`.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/final-review-report.md`(I1/I2/I3/I4 + Minor M3/M9)
- 分支 `feat/m1-asr-session-id`;本波次代码 + 测试 + 记账提交 = **`4609d8b`**(文档收尾提交紧随其后)
- 解释器:`~/miniconda3/envs/jingxin/bin/python`
- 改动面:6 个代码/测试文件 + 3 个文档 + `logging_config.py` 入 git;**未派任何子智能体**
- **未动** `face_expression/pipeline/video_pipeline.py` 的几何层(逐字未改)

| 条 | 裁决 | 状态 |
|---|---|---|
| I1 `close()` 同步阻塞 | 挪到后台线程,不改 async 结构 | ✅ 修 + 钉住 + 实测前后对比 |
| I2 `docs/下一步.md` 与账本 | 按报告清单改 §0/§2/§3/§5/§6 | ✅ 改 + 账本归档进仓库 |
| I3(a) gesture `verify_models` 无测 | 补两条 | ✅ 补 + 反向复现 |
| I3(b) 无"同会话复用"钉子 | 补一条 | ✅ 补 + 反向复现 |
| I4 `logging_config.py` 未跟踪 | `git add` | ✅ 已入 git(未被 `.gitignore` 挡) |
| Minor:计划修正版未提交 | 提交 | ✅ |
| Minor:金标比对判据口径(M9) | 改写判据 | ✅ 写成"B 臂 = IMAGE 内层替身 / 或另跑 VIDEO 基线",并明写不许把 B 臂结论当 A 臂 |

---

## I1:`close()` 挪到后台线程

### 改动

**各加一个模块级 helper(两份刻意不共享,按要求不合并):**

- `face_expression/pipeline/detector.py`(+18 行)
- `gesture_analysis/core/detectors.py`(+18 行)

```python
_CLOSER = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="detector-close")

def close_detached(detector) -> None:
    """把 close() 挪到后台线程。
    为什么:close() 实测恒 5.0s(构造只要 0.1–0.3s),而它是在**请求路径上同步**调的
    —— TTL 回收 2 个探测器 = +10s、无 id 的 /reset = +20s,期间整个事件循环被冻住
    (并发 /health 实测 19.73s,基线 0.0019s)。挪到线程后请求立刻返回,native 句柄
    仍会被释放(晚几秒)。**别"顺手"把它改回同步** —— 那会把这个停顿带回来。
    """
    _CLOSER.submit(detector.close)
```

**调用点全部改走 helper(6 处,只改调用、不改语义):**

| 文件 | 位置 |
|---|---|
| `gesture_analysis/api/app.py` | TTL 回收段(2 个:`get_or_create_analyzers` 与 `get_or_create_detectors`)、`/reset` 两条路径(带 id、无 id) |
| `face_expression/api/app.py` | TTL 回收段、`/session/{session_id}/reset` |

两个封装自己的 `.close()` 方法**保留**(它才是实际干活的那个)。

**一处必须说明的接线细节:** gesture 的 `api/app.py` 只按名字 import `close_detached`,
**没有**把 `HandDetector`/`PoseDetector` 提到模块级 —— 那两个名字必须留在
`get_or_create_detectors` 的函数体内按调用时解析,否则
`tests/test_analyze_session_fallback.py` 对 `detectors.HandDetector` 的 monkeypatch 会失效
(模块级 import 会把补丁之前的值绑死)。`gesture_analysis.core.detectors` 自身 import 期不碰
mediapipe,所以这条 import 不破坏 D3 的"import 期不构造探测器"。

### 实测前后对比

同一台机器、同一批模型、同一帧文件(`~/shared/mp_frames/frames/frame_0009.png`)、
同一脚本 `/tmp/m15fix/measure.py`,唯一变量 = 调用点是否走 `close_detached`
("之前"用 `git show 8cbe149:gesture_analysis/api/app.py` 的原文件替换后重启)。

| 量 | 之前(同步 close) | 之后(close_detached) |
|---|---|---|
| `/health` 基线(5 次中位数) | 1.0547 ms | 1.2698 ms |
| `POST /reset`(无 id,**2 个活跃会话**) | **20.04 s** | **0.01 s** |
| 期间**并发** `/health` 最大耗时 | **20.03 s** | **0.00 s** |
| 期间并发 `/health` 样本数 / 中位数 | 42 / 4.0 ms | 43 / 3.2 ms |

> 审阅报告的对应数字是 `/reset` 20.04s、并发 `/health` 19.73s —— 本波次在"之前"那一侧
> **逐位复现**(20.04 / 20.03),所以"之后"的 0.01s / 0.00s 是同一口径下的干净对比。
> 两个会话建起来各 HTTP 200(0.17s / 0.18s)。

原始输出:

```
===== BEFORE (同步 close) =====
  建会话 m15fix_a: HTTP=200 (0.17s)
  建会话 m15fix_b: HTTP=200 (0.19s)
  基线 /health: 1.0547 ms (5 次中位数, max 1.6910 ms)
  POST /reset(无 id,2 会话): HTTP=200 耗时 **20.04s**
  期间并发 /health: n=42 max **20.03s** median 4.0 ms

===== AFTER (close_detached) =====
  建会话 m15fix_a: HTTP=200 (0.17s)
  建会话 m15fix_b: HTTP=200 (0.18s)
  基线 /health: 1.2698 ms (5 次中位数, max 1.4306 ms)
  POST /reset(无 id,2 会话): HTTP=200 耗时 **0.01s**
  期间并发 /health: n=43 max **0.00s** median 3.2 ms
```

### 钉子

`tests/test_detector_contract.py::test_close_detached_does_not_block_the_caller`
(**parametrize 两条:face / gesture** —— 两个 helper 都要压)。两半缺一不可:

1. `close_detached(d)` 在 **0.2s 内返回**(假探测器 `close()` 里 `time.sleep(1.0)`);
2. 稍后断言底层 `close()` **确实被调到**(轮询≤5s)—— 否则只是把"阻塞"换成"泄漏"。

### 反向复现(证明钉子有牙齿)

把两个 helper 都改回 `detector.close()`:

```
E  AssertionError: close_detached 在调用者线程上等了 1.00s —— 请求路径会被冻住
   (真 close() 是 5.0s,TTL 回收/`/reset` 会把它放大到 10–20s)
E  assert 1.0004841769987252 < 0.2
E  AssertionError: close_detached 在调用者线程上等了 1.00s —— ...(gesture 那条同形)
FAILED tests/test_detector_contract.py::test_close_detached_does_not_block_the_caller[face]
FAILED tests/test_detector_contract.py::test_close_detached_does_not_block_the_caller[gesture]
```

恢复后 18 passed。

### 顺带:一条既有接线测必须跟着改(否则假红)

`tests/test_gesture_detector_wiring.py::test_expired_sessions_close_their_detectors`
原来在 `get_or_create_detectors("t_new")` 之后**原地**读 `hands.closed`。接口从"同步关"
变成"后台关"之后,原地断言量到的是"还没跑"而不是"没跑" —— 这是**假红**,不是回归。
已改成 `_wait_until(lambda: hands.closed and pose.closed)`(超时 5s,超时仍 False 才红),
**没放宽语义**:close 一次不调就照样红。改前实测 5/5 次红,改后 3/3 次绿。

---

## I2:`docs/下一步.md` + 账本归档

### 文档改动(按报告 I2 清单)

| 节 | 改了什么 |
|---|---|
| 头部 + §0 | 现状改为 **M1.5 已完成待合并**、两个服务**已复活**;测试数 177 → **198**;HEAD 指向本波次代码提交 `4609d8b`;下一步 = **跑 T7**(不再是"spec 待写") |
| §2 | 标题改为"**曾经**挡在 T7 前面的硬伤…(**已修:M1.5**)",**保留诊断记录**(它是立项依据);表格第三列换成**修完之后的实跑状态**(face/gesture 都 ✅ 200 + 落盘带 sid);新增一段点名两件"已登记但未验"的事;T7 段补上三个服务的起动命令 |
| §3 | 第 1 条(迁 tasks)与第 7 条(`logging_config.py` 入 git)标为**已完成**;**新增第 15 条** = spec §10.4 的 VIDEO 模式等价性门(A 臂 1425/9082 vs B 臂 9560/9560 的口径写清);**新增第 16 条** = spec §12.1 的 M3 阈值/量程重登记(指名 `evidence_thresholds.json` 的 `au` 族 + `scale_factors`) |
| §5 | 关键文件索引补上 M1.5 的 spec / plan / 账本(并注明账本已从被 gitignore 的 `.superpowers/sdd/` 归档进仓库);CV 服务那一行从"待迁 tasks API"改成两个封装 + `close_detached` 的实际位置 |
| §6 | 路线图里 M1.5 标 ✅ 已完成,箭头移到 **M2**;M3 那行补上"含跟进项 16 与 spec §3.4 的位移表" |

### 账本归档(M4)

`.superpowers/sdd/` 的 `.gitignore` 是 `*`(整个目录被忽略),所以原账本进不了仓库。
已复制到**被跟踪**的 `docs/superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/`
(已确认该路径未被任何 `.gitignore` 命中,与 M1 账本落位一致):

```
progress.md  task-2-brief.md  task-2-report.md  task-3-brief.md  task-3-report.md
final-review-report.md
```

Task 1 由控制器在主会话亲自做,它没有独立报告 —— 其裁决与反向复现记录在 `progress.md` 里
(任务书里说的"三份 task 报告"落地为 task-2 / task-3 的 brief+report 两对,加上最终审查报告;
比字面多带了两份 brief 与最终审查报告,因为它们是同一份证据链的上下文,不带上就会断链)。

---

## I3:两处补钉子

### (a) gesture 的 `verify_models`

`tests/test_detector_contract.py` 新增两条(此前只 import 了 face 的那一个):

- `test_gesture_verify_models_fails_loudly_when_a_file_is_missing`:hand 缺 / pose 缺
  **各一半**,都断言抛 `RuntimeError` 且消息里含**缺的那个路径**;
- `test_gesture_verify_models_accepts_present_files`:两个文件都在 → 不抛。

用 `factory=` 注入假工厂,不真加载模型。

**反向复现**:把 `if not path.exists()` 改成 `if False` →

```
E  Failed: DID NOT RAISE RuntimeError
1 failed, 1 passed, 11 deselected
```

### (b) 同一会话复用同一个探测器

`tests/test_gesture_detector_wiring.py::test_same_session_reuses_the_same_detectors`:
对同一个 `session_id` 连调两次 `get_or_create_detectors`,断言 `is`(整表 + hands + pose)。
用现成的 `fake_detectors` fixture。

**反向复现**:把 `if session_id not in detectors:` 改成 `if True:` →

```
E  AssertionError: 同一会话第二次调用换了探测器 —— 帧计数归零、ts 回退,且泄漏句柄
FAILED tests/test_gesture_detector_wiring.py::test_same_session_reuses_the_same_detectors
```

> 这条正是报告说的那个缝:改完之后**别的 193 条照样全绿**,只有这一条会红。
> 断言点选的是 `is`(对象同一性)而不是"第二次调用的 hands 能 detect" —— 后者在重建下
> 也成立,钉不住。

---

## I4:`logging_config.py` 入 git

- `git check-ignore -v logging_config.py` → **exit 1**(未被任何 `.gitignore` 命中,不需要加例外);
- `git add logging_config.py` → 63 行入库,与 `face_expression/api/app.py:11`、
  `gesture_analysis/api/app.py:11`、voice 的 import 对上。

---

## 顺带修(Minor)

- **计划的修正版已提交**(`docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md`,
  +114/−13 在工作树里躺了很久):含"最终审查判据按**可达性**判"的修正、开工前预检掉的
  三条缺陷、`fake_detectors` fixture 那一整段。
- **金标比对判据口径(M9)**:在计划的最终审查清单里就地改写成"**要么**用 IMAGE 模式内层替身
  (B 臂,实测 face 9560/9560、gesture 1119/1119),**要么**另跑一套 VIDEO 模式新基线",
  并加了 A 臂/B 臂对照表与一句 **"不许把 B 臂的结论写成 A 臂的"**;顺带把该清单里的
  "Review Focus **五条**"改成**六条**(它自己的 Review Focus 节列了 6 条)。

---

## 实跑验收

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q
198 passed in 14.51s          # 连跑 3 次均 198 passed(基线 193,本波次 +5)
```

新增 5 条:2 (close_detached parametrize) + 2 (gesture verify_models) + 1 (同会话复用)。

合并门(改动前后各跑一次,均通过):

```
$ cd ~/jingxin/experiments/duration_audit
$ $PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe_fixwave
[legacy] 2011 视频 × 1410 维 (face=1260 gesture=140 voice=10), 死特征 169
$ $PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe_fixwave
最大绝对差 1.886e-19   最大相对差 1.886e-19
相对差 > 1e-4 的格子: 0 / 2835510
targets 最大绝对差 0.000e+00
✅ 回归验证通过
```

---

## 顾虑

1. **`max_workers=1` 的排队**:两个 helper 各只有一个后台线程(刻意的:close 是 native
   释放,串行更安全)。所以连续回收 N 个会话时,后台的 close 会**排队**,总时长仍是
   ~5s × N —— 但**请求路径**不再等它们。文件描述符的峰值窗口因此变长(最坏 = 排队总时长),
   这是"把停顿从请求路径挪走"必然的代价;若将来真出现句柄压力,再考虑加线程数。
2. **进程退出时的清理**:`ThreadPoolExecutor` 自己注册的 `atexit` 会 join 后台线程,所以
   排在队列里的 close 在正常退出时仍会跑完;但 `kill -9` 下 native 句柄由 OS 回收。
3. **spec §10.4 的 VIDEO 门仍未验**(已按裁决登记进 `docs/下一步.md` §3 第 15 条)——
   本波次**没有**动它,也没有把 B 臂的结论借用给 A 臂。
4. `docs/下一步.md` 里引用的哈希 `4609d8b` 是**本波次的代码提交**;文档收尾提交紧随其后,
   所以严格说 HEAD 比它多一个(文档里已写明这一句)。
