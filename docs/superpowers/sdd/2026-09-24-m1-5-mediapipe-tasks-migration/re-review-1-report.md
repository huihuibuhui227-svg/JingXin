# M1.5 修复波次 · 范围受限复审报告(re-review-1)

- 复审对象:`8cbe149..e462d16`(2 个提交:`4609d8b` 代码 + 文档记账、`e462d16` 文档收尾)
- diff:`.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/review-8cbe149..e462d16.diff`(2814 行 / 16 文件)
- 复审范围:**只看这个 diff**;不重审整条分支
- 复审者:范围受限复审者(只报告不改动;未派任何子智能体)
- 环境:`~/miniconda3/envs/jingxin/bin/python`(mediapipe 1.0.0;三个 `.task` 齐全;`~/shared/mp_frames/frames/` 有真帧)
- 前置报告:`.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/{final-review-report.md,fix-wave-1-report.md}`

**一句话判定:4 条 Important + 相关 Minor 全部 ADDRESSED,逐条有实证;修复 diff 引入 1 条 Important 级(低概率、一行可修)的新破坏 = 后台 `close()` 的异常被完全静默吞掉;另有 3 条 Minor。修复波次可以收。**

---

## 0. 判定总览

| 条 | 内容 | 判定 | 关键证据 |
|---|---|---|---|
| I1 | `close()` 同步阻塞 → 加模块级 `close_detached()`,6 处调用点改走它 | **ADDRESSED** | 6/6 调用点已改;两模块各一份 helper;`.close()` 保留;端到端 `/reset` 20.04s → **0.0025s** |
| I2 | `docs/下一步.md` §0/§2/§3/§5/§6 + 账本归档进被跟踪目录 | **ADDRESSED** | 五节全改;§3 新增第 15/16 条;账本 7 个文件 `git ls-files` 可见 |
| I3(a) | gesture `verify_models` 无测试 | **ADDRESSED** | 新增 2 条;变异测试确认会红 |
| I3(b) | 无「同会话复用同一探测器」钉子 | **ADDRESSED** | 新增 1 条;变异后**只有它(的实质失败)红** |
| I4 | `logging_config.py` 未跟踪 | **ADDRESSED** | `git ls-files` 命中;`git check-ignore` exit 1 |
| M9 | 计划里「金标比对」判据口径(A 臂/B 臂) | **ADDRESSED** | 计划里就地改写 + 明写「不许把 B 臂的结论写成 A 臂的」 |
| — | 计划修正版未提交(M3 那条 Minor) | **ADDRESSED** | 计划 +114/−13 已在 `4609d8b` 里 |

---

## 1. I1 —— `close()` 挪出请求路径:**ADDRESSED**

### 1.1 6 处调用点是否真的都改了(逐处点名)

`grep -rn "\.close()" face_expression/api/app.py gesture_analysis/api/app.py` → **零命中**;两个文件的调用点全部改走 helper:

| # | 归属 | 位置 | 现在的写法 |
|---|---|---|---|
| 1 | face · TTL 回收 | `face_expression/api/app.py:130` | `close_detached(pipeline)` |
| 2 | face · `/session/{sid}/reset` | `face_expression/api/app.py:366` | `close_detached(pipeline)` |
| 3 | gesture · TTL(`get_or_create_analyzers`) | `gesture_analysis/api/app.py:139` | `close_detached(d)` |
| 4 | gesture · TTL(`get_or_create_detectors`) | `gesture_analysis/api/app.py:179` | `close_detached(d)` |
| 5 | gesture · `/reset?session_id=` | `gesture_analysis/api/app.py:479` | `close_detached(d)` |
| 6 | gesture · `/reset`(无 id) | `gesture_analysis/api/app.py:488` | `close_detached(d)` |

6/6,无遗漏。**残留的 `.close()` 全部是应该留的**(复核过全包,不含 `examples/`、`utils/`):

- 封装自己的释放体:`face_expression/pipeline/detector.py:94`、`gesture_analysis/core/detectors.py:116`(`self._landmarker.close()`);
- 转发层:`face_expression/pipeline/video_pipeline.py:190`(`self.detector.close()`,由 #1/#2 经 helper 调);
- `gesture_analysis/core/detectors.py:178-186` 是 `verify_models()` **启动自检**里的一次性 close(启动路径,不是请求路径;pre-existing,+10s 启动开销不在 I1 的 6 处之列)。

### 1.2 helper 是否两个模块各一份;`.close()` 是否保留

- `face_expression/pipeline/detector.py:23-36`:`_CLOSER` + `close_detached()`;`gesture_analysis/core/detectors.py:25-38`:同样一份。
- **实测确认两份不是同一个对象**:`fd._CLOSER is gd._CLOSER` → `False`(刻意不共享,与 `_default_factory` 同处理)。
- `.close()` 方法两侧都在:`detector.py:91-95`、`detectors.py:113-117`,且 `if self._landmarker is not None:` 幂等守卫未动。
- async 结构**未改**:两个服务的端点仍全是 `async def`(`gesture_analysis/api/app.py:223,464`;`face_expression/api/app.py:184,356`),调用点仍是同步调用一个「不阻塞的」helper —— 符合「不改 async 结构」的要求。

### 1.3 独立验证(自己造的假件 + 端到端真服务)

**(a) 假件:helper 立刻返回、底层稍后确实被调到**(`/tmp/rereview/verify_detached.py`,`close()` 里 `sleep(3.0)`):

```
OK face:    helper 返回 6.36 ms;底层 close() 稍后跑完,线程=detector-close_0
OK gesture: helper 返回 1.20 ms;底层 close() 稍后跑完,线程=detector-close_0
OK 幂等: 同一对象提交两次,底层实际释放次数 = 1 (期望 1)
```

**(b) 端到端:真 uvicorn + 真模型 + 真帧**(`/tmp/rereview/e2e_real_server.py`,口径与最终审查报告一致 = 2 个活跃会话 + `POST /reset` 无 id + 期间并发 `/health`):

```
服务已就绪 (:8002)
基线 /health 中位数 = 2.800 ms (n=5)
  建会话 rr2_a: HTTP=200 (0.16s)
  建会话 rr2_b: HTTP=200 (0.18s)
POST /reset(无 id,2 会话): HTTP=200 耗时 0.0025s  body={'status': 'success', 'message': '所有会话已重置'}
  期间并发 /health: n=53 max=2.68 ms median=0.95 ms
判定: OK 事件循环没有被冻住
```

对照被审报告的数字(`/reset` **20.04s**、并发 `/health` **20.03/19.73s**):**修复成立,量级差 4 个数量级**。

**(c) 钉子有牙齿**:把两个 helper 都改回 `detector.close()` →

```
FAILED tests/test_detector_contract.py::test_close_detached_does_not_block_the_caller[face]
FAILED tests/test_detector_contract.py::test_close_detached_does_not_block_the_caller[gesture]
E  assert 1.0002896369987866 < 0.2
```

两侧都压住了(`tests/test_detector_contract.py:205-247`,parametrize face/gesture)。

---

## 2. I2 —— 文档 + 账本:**ADDRESSED**

### 2.1 五节逐节核对(`docs/下一步.md`)

| 节 | 现状 | 证据 |
|---|---|---|
| 头部 + §0 | 已改:M1.5 记为「已完成,尚未合并」,测试数 177 → **198**,下一步 = **跑 T7**,两个服务**已复活** | `docs/下一步.md:3`、`:12-20` |
| §2 | 标题改为「**曾经**挡在 T7 前面的硬伤 …(**已修:M1.5**)」;**诊断记录保留**;表格第三列换成修完之后的实跑状态;补了三个服务的启动命令 | `:40`、`:42`、`:44`、`:65-70` |
| §3 | 第 1 条、第 7 条标 ✅ 已完成;**新增第 15 条与第 16 条** | `:98`、`:104`、`:112`、`:113` |
| §5 | 补 M1.5 spec / plan / 账本三行(并注明账本已从被 gitignore 的目录归档进来);CV 服务那行改成两个封装 + `close_detached` 的实际位置 | `:149` 起 |
| §6 | M1.5 标 ✅ 已完成、箭头移到 **M2**;M3 行补「含跟进项 16 与 spec §3.4 的位移表」 | `:156` 起 |

### 2.2 §3 里那两条会丢的 deferral(**确实新增了**)

- **第 15 条 = spec §10.4 的 VIDEO 模式等价性门**(`docs/下一步.md:112`):写明「仓库封装固定在 VIDEO、基线是 IMAGE → 原样比不可能逐点一致」,给出 A 臂 1425/9082、gesture 152/1056,**并明写 100% 的是 B 臂(face 9560/9560、gesture 1119/1119)「那是适配层写对了的证据,不是 A 臂的结论」**。
- **第 16 条 = spec §12.1 的 M3 阈值/量程重登记**(`:113`):点名 `report_frontend/evidence_thresholds.json` 的 **`au` 族 + `scale_factors`**,并带上 §3.4 的位移数字与「系统性偏小」的方向。

两条都在,且都没有把 B 臂的结论借用给 A 臂。

### 2.3 账本是否真的进了**被跟踪**的目录(`git ls-files` 为准)

```
$ git ls-files docs/superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/
docs/.../final-review-report.md
docs/.../fix-wave-1-report.md
docs/.../progress.md
docs/.../task-2-brief.md
docs/.../task-2-report.md
docs/.../task-3-brief.md
docs/.../task-3-report.md
```

7 个文件**都被跟踪**(不是「文件在磁盘上」而已)。内容与源目录**逐字节相同**(逐个 `cmp` 通过:`progress.md` / `final-review-report.md` / `fix-wave-1-report.md`)。

> 注:修复报告里写的是 6 个文件(没算它自己那份 `fix-wave-1-report.md`)。实际 7 个,多出来的正是修复波次报告本身 —— 数量对不上是措辞问题,不构成缺口。

---

## 3. I3 —— 两处补钉子:**ADDRESSED**

### (a) gesture `verify_models`:`tests/test_detector_contract.py:255-280`

两条都加了:

- `test_gesture_verify_models_fails_loudly_when_a_file_is_missing`:**hand 缺 / pose 缺各一半**,两边都断言抛 `RuntimeError` 且 `str(missing) in str(e)`(消息在 `gesture_analysis/core/detectors.py:172-174`,含缺失路径);
- `test_gesture_verify_models_accepts_present_files`:两文件都在 → 不抛。

用 `factory=` 注入假工厂(`_factory` 的 `build(*args, **opts)` 与 gesture 的 `factory(hp, num_hands=2, …)` 签名兼容),不真加载模型。

**反向复现(我做的)**:把 `gesture_analysis/core/detectors.py:171` 的 `if not path.exists():` 改成 `if False:` →

```
E  Failed: DID NOT RAISE RuntimeError
FAILED tests/test_detector_contract.py::test_gesture_verify_models_fails_loudly_when_a_file_is_missing
```

有牙齿。

### (b) 同会话复用:`tests/test_gesture_detector_wiring.py:112-140`

`test_same_session_reuses_the_same_detectors`:同一 `session_id` 连调两次,断言整表 `is`(整表 + `hands` + `pose` 三处),断点选**对象同一性**,不是「第二次还能 detect」(后者在重建下也成立,钉不住)。

**反向复现(我做的)**:把 `gesture_analysis/api/app.py:181` 的 `if session_id not in detectors:` 改成 `if True:` →

```
FAILED tests/test_gesture_detector_wiring.py::test_same_session_reuses_the_same_detectors
        同一会话第二次调用换了探测器 —— 帧计数归零、ts 回退,且泄漏句柄
2 failed, 196 passed
```

关于「应当只有它红」:失败共 2 条,第二条是 `tests/test_assert_coverage.py::test_no_assert_is_dead` —— 它把整个套件放子进程跑一遍并断言绿,所以是**级联**(报告里也自述了这一点),不是第二个独立探测器。所以「别的都全绿、只有这一条会红」**在实质上成立**。

---

## 4. I4 —— `logging_config.py`:**ADDRESSED**

```
$ git ls-files logging_config.py
logging_config.py
$ git check-ignore -v logging_config.py
(exit=1 —— 未被任何 .gitignore 命中)
```

63 行入库(`logging_config.py:1-63` 的 `new file` hunk 在 diff 里),与 `face_expression/api/app.py:11`、`gesture_analysis/api/app.py:11`、voice 三处 `from logging_config import setup_logging` 对上。

---

## 5. M9 —— 计划里的金标比对判据口径:**ADDRESSED**

`docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md` 的最终审查清单里就地改写了判据(diff 250-264 行):

- 给出 **A 臂 / B 臂对照表**(A = 原样用封装(VIDEO)→ face 1425/9082、gesture 152/1056;B = IMAGE 内层替身 → face 9560/9560、gesture 1119/1119,逐点 100%);
- 明写**判据必须写成二者之一**(① 用 IMAGE 模式内层替身比,或 ② 另跑一套 VIDEO 新基线);
- **明写「不许把 B 臂的结论写成 A 臂的」**,并把 A 臂的差归因到模式差异、指向 spec §10.4 那道已知未验的门(与 `docs/下一步.md` §3 第 15 条互指)。

**没有把 B 臂的结论误写成 A 臂的。**

顺带两处也对上了:Review Focus「五条」→ **六条**(与它自己列出的 6 条一致);`mp.solutions` 的判据从「照字面 `grep -v examples/`」改成**按可达性判**(正是账本记的那条修正)。

---

## 6. 修复 diff 本身有没有引入新破坏

先回答指定要看的两个点,再给新发现。

### 6.1 指定关注点:逐条结论

**(i)「把 close 挪到后台线程之后,有没有哪条路径会用到已经进入关闭流程的探测器?」→ 没有找到。**

三条理由,都实测/复核过:

1. **每个 close 站点都是「先摘表、后提交」**:`gesture_analysis/api/app.py:476-488`(`detectors.pop` / `detectors.clear()` 在 `close_detached` 之前或同一段同步代码内)、`:136-139`、`:176-179`、`face_expression/api/app.py:129-130`、`:365-366`。摘表之后该对象不再被任何请求取到。
2. **两个服务的端点全是 `async def`,跑在同一个事件循环上;而「取到探测器 → 用它」之间没有 `await`**,所以别的请求无法在这中间插进来 `/reset`:gesture `analyze_image` 是 `:261` 取、`:263`/`:274` 用(中间无 await);face `analyze_frame` 是 `:227` 取、`:230` 用(中间无 await)。取之前有 `await`(`:240`/`:246`、`:246` 等),但那时还没拿到对象。
3. **helper 在重复提交下是幂等的**:`max_workers=1` 把 close 串行化,加上 `close()` 的 `if self._landmarker is not None` 守卫,同一对象提交两次只释放一次 —— 实测 `底层实际释放次数 = 1`。这正好堵住「TTL 与 `/reset` 撞车 → double free」这个 native 资源上最容易出事的地方。

**(ii)「两个 `ThreadPoolExecutor(max_workers=1)`(模块级、非 daemon)的对象生命周期有没有问题?」→ 有两个真问题,但都不是 Critical。**

- 对象本身没问题:构造是**惰性的**(import 后 `threading.active_count() == 1`,没有线程被提前拉起),两份刻意不共享(`is` → False),`max_workers=1` 让 close 串行(对 native 释放是更安全的选择)。
- **问题 A(Minor):串行 + 每个 5.0s ⇒ 句柄释放窗口 = 队列深度 × 5s。** 实测:2 个会话(4 个真探测器)一次 `/reset` 之后,后台要 **20.0s** 才把它们全放掉(请求路径不等,但句柄在这 20s 里仍活着)。突发多次 `/reset` 会让待放句柄堆积。修复报告的顾虑 1 已承认这一点,判断一致。
- **问题 B(Minor,且**修复报告的说法不准**):修复报告顾虑 2 说「`ThreadPoolExecutor` 自己注册的 `atexit` 会 join 后台线程,所以排在队列里的 close 在正常退出时仍会跑完」。实测**不成立**:造 4 个真探测器 → 全提交 → 立刻正常退出,进程在 **5.67s** 后死亡,4 个 close 里**只跑完 1 个**,#1/#2/#3 只打印了「开始」就被进程死亡打断(marker 文件法也确认只剩 `wrapped_0.done`;退出码 0,不是崩溃)。结论上没造成泄漏(进程一死,OS 回收 native 句柄 —— 与报告自己认可的 `kill -9` 情形同性质),但**报告里那句推理是错的**,不该留在账本里当依据。
  - 附带一条反直觉的实测:退出**不会**被队列拖慢(5.67s / 0.69s),因为 finalization 期间 native close 会立刻返回。我先前用 `time.sleep` 假件做的对照实验里退出确实被拖了 8s(`sleep` 不可被打断)—— 所以这条必须以真对象为准。

**(iii)「把测试从原地断言改成轮询等条件,有没有变松(比如改成恒真)?」→ 没有变松。**

`_wait_until`(`tests/test_gesture_detector_wiring.py:73-85`)的返回是 `predicate()` —— 循环只是「等」,超时后仍返回 False,断言照样红;它不是恒真。

实测(我做的变异):把 `gesture_analysis/api/app.py:179` 的 `close_detached(d)` 整个换成 `pass`(即回到「回收不 close」) →

```
FAILED tests/test_gesture_detector_wiring.py::test_expired_sessions_close_their_detectors
1 failed in 5.28s      ← 5s 是超时,之后判 False → 红
```

所以语义强度与改前一致(改前是「原地读到还没跑」的假红,现在是真等),没有把有约束力的断言改成恒真。

### 6.2 新发现

**N1 —— Important(低概率;一行可修):后台 `close()` 抛出的异常现在被完全静默吞掉。**

- 位置:`face_expression/pipeline/detector.py:36`、`gesture_analysis/core/detectors.py:38`(`_CLOSER.submit(detector.close)` 的返回值被丢弃)。
- `concurrent.futures` **不会**汇报未被取回的 Future 异常(不像 asyncio 会给 warning)。实测:造一个 `close()` 直接抛 `RuntimeError` 的探测器 → **无 traceback、无 warning、退出码 0**。
- 改前的可见性是有的:`gesture_analysis/api/app.py:493-494` 的 `except Exception → HTTPException(500, "重置失败: …")`、同文件 `/analyze` 里 TTL 回收的 close 也会经 `:401-403` 变成 500、`face_expression/api/app.py` 的 `reset_session` 无 try/except → 直接 500。
- 改后的后果:`/reset` 无论释放成功与否都返回 `{"status":"success"}`(`gesture_analysis/api/app.py:486`、`face_expression/api/app.py:368`);而 `close()` 只在成功后才把 `_landmarker` 置 None(`detector.py:93-95`),所以失败 = **native 句柄静默泄漏** —— 正好是 spec §6.3 要防的东西,现在以「零信号」的形式回来。本仓一路在杀的正是这个形态(「测试通过 ≠ 有约束力」「报告里印着一个什么都不代表的数」)。
- 概率不高(需要 `_landmarker.close()` 自己抛;跨线程 close 本身我实测能正常跑完,见 6.1),所以我给 Important 而不是 Critical。
- 一行修法:`fut = _CLOSER.submit(...)` + `fut.add_done_callback(lambda f: f.exception() and logger.error(...))`(或直接 `logger.exception`)。

**N2 —— Minor:`max_workers=1` 让句柄释放窗口 = 队列深度 × 5s(实测 2 会话 = 20.0s)。** 见 6.1(ii) 问题 A。与修复报告的顾虑 1 一致,记档即可。

**N3 —— Minor:修复报告顾虑 2 关于退出时 atexit join 的说法不成立**(见 6.1(ii) 问题 B,实测 4 个 close 只跑完 1 个)。结论无害,但账本里那句话需要改口。

**N4 —— Minor:计划里那段内嵌测试代码已经过期。** `docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md:951` 仍写着原地断言 `assert hands.closed and pose.closed`,而实际测试已改成 `_wait_until(...)`;该计划文件里 `_wait_until` 与 `close_detached` 的提及次数都是 **0**。这份快照是在**同一波次**里被更新成「修正版」的,却没跟上 I1 对测试的改动 —— 照它重跑的人会写出一条**假红**测试(正是修复报告 1.3 节记录的那个坑)。计划是历史快照,不构成缺口,但值得一行标注。

### 6.3 明确「没有」的东西(复核过,不是 finding)

- **无 Critical。**没有发现错值、数据丢失、产物遮蔽一类形态;改动面只有「close 挪线程 + 补测试 + 文档 + 入 git」,没有新增端点/输入面/反序列化点。
- **import 期约束没被破坏**:新增的模块级 import 只到 `gesture_analysis.core.detectors`(`gesture_analysis/api/app.py:27`),该模块自身 import 期不碰 mediapipe(mediapipe 只在工厂函数体内);且刻意**没有**把 `HandDetector`/`PoseDetector` 提到模块级,保住了 `tests/test_analyze_session_fallback.py` 对 `detectors.HandDetector` 的 monkeypatch(`test_importing_the_app_does_not_touch_mediapipe` 仍绿)。
- **`VideoPipeline.close()` 转发语义未变**(`video_pipeline.py:188-190` 只 `self.detector.close()`),把整条 pipeline 交给 helper 是安全的 —— 它不碰别的共享状态。
- **唯一的 helper 引用是 import 期绑定的**:`close_detached` 以名字直接 import,若有测试想 monkeypatch `detectors.close_detached` 不会生效 —— 但当前没有任何测试这么做(新测试是直接 parametrize 那两个函数对象),不构成缺陷。

---

## 7. 实跑结果

### 7.1 全量测试(声称 198 passed)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q
198 passed in 14.57s      ← 我跑了 3 次:14.57s / 14.36s / 14.58s,均 198 passed
```

**与声称完全一致。**基线 193 + 本波次 +5 = 198(2 close_detached parametrize + 2 gesture verify_models + 1 同会话复用),加数也对得上。

### 7.2 独立验证 `close_detached` 不阻塞

见 1.3(a):真造一个 `close()` 里 `sleep(3.0)` 的假件 → helper **6.36 ms / 1.20 ms** 返回(两个模块各一次),底层 close 稍后**确实被调到**且跑在 `detector-close_0` 线程上;同一对象重复提交只释放一次。

### 7.3 变异测试(证明钉子有牙齿;全部已还原)

| # | 变异 | 结果 | 还原 |
|---|---|---|---|
| M1 | `gesture_analysis/api/app.py:181` `if session_id not in detectors:` → `if True:` | `test_same_session_reuses_the_same_detectors` 红(+ `test_no_assert_is_dead` 级联);196 passed | ✅ `git checkout` |
| M2 | `gesture_analysis/core/detectors.py:171` `if not path.exists():` → `if False:` | `test_gesture_verify_models_fails_loudly_when_a_file_is_missing` 红(`DID NOT RAISE RuntimeError`) | ✅ |
| M3 | 两个 helper 的 `_CLOSER.submit(detector.close)` → `detector.close()` | `test_close_detached_does_not_block_the_caller[face]` **和** `[gesture]` 双红 | ✅ |
| M4 | `gesture_analysis/api/app.py:179` TTL 的 `close_detached(d)` → `pass` | `test_expired_sessions_close_their_detectors` 红(5.28s = 走满超时) | ✅ |

**工作树已还原**:5 个被碰过的文件逐个 `sha256sum -c` 通过,`git diff --stat` 为空,复跑 198 passed。全程未提交、未改动 HEAD。

---

## 8. 结论

**这个修复波次能不能收:能收。**

4 条 Important(I1/I2/I3/I4)与相关 Minor(M3/M9)逐条 **ADDRESSED**,而且不是纸面落地:I1 我用自己的假件 + 真 uvicorn 服务双重复核(`/reset` 从 20.04s 变成 **0.0025s**,并发 `/health` 的最大耗时从 20.03s 变成 **2.68 ms**),6 处调用点无遗漏、`.close()` 保留、async 结构未动;I2 的账本用 `git ls-files` 确认**真的被跟踪**;I3 的两条测试我都做了变异复现,**都会红**;I4 的 `git ls-files` 命中。

修复 diff **没有引入 Critical**。唯一需要表态的新破坏是 **N1**(后台 `close()` 的异常被完全静默吞掉 → native 句柄可能静默泄漏,而 `/reset` 仍报 success):它概率低、影响面是资源释放的错误可见性,但正好落在本仓最忌讳的「静默失效」形态上,且修法只有一行(`add_done_callback` 里 `logger.exception`)。我的建议是**这一行可以在合并前顺手补掉**(顺带修一下 N3 那句关于 atexit 的错误推理);若不补,请把它按 Minor 记进账本而不是当作已经稳妥 —— 其余 N2/N4 记档即可。

> 本报告与 I2 指出的问题同类:文件落在**被 gitignore** 的 `.superpowers/sdd/` 下(应要求写在此处)。若要留证据,需按 M1.5 账本的既有做法复制进**被跟踪的** `docs/superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/`。
